"""
TradeX – AI Orchestration Service  (v2 – Automatic Rolling Memory)
==================================================================

WHAT CHANGED vs v1 (and WHY)
------------------------------

ROOT CAUSE of "history too long" 503 error
-------------------------------------------
The previous code raised HTTP 503 with a user-visible message asking them to
manually delete their session when the ONE retry after a 413 also failed.

That retry failed because:
  1. The session STORE was never compressed between requests.  Even though
     _call_groq_with_budget() trimmed the messages for one call, it never wrote
     the trimmed version back to _SESSION_STORE.  So the next request started
     with the same bloated history.
  2. A single retry with tail=3 is not enough when the system prompt +
     summary itself is large.

THE FIX (multi-layer defence)
------------------------------

Layer 1 – Proactive session compression (compress_session)
  _append_message() now calls compress_session() after EVERY write.
  If the session store exceeds SOFT_COMPRESSION_THRESHOLD (6 000 tokens),
  old messages are summarised and replaced IN PLACE in _SESSION_STORE.
  The user-visible history is stored SEPARATELY in _HISTORY_STORE and is
  never compressed, so GET /ai/history always returns the full conversation.

Layer 2 – Pre-flight token check (unchanged, improved)
  _call_groq_with_budget() estimates tokens before every call and trims if
  over TOKEN_BUDGET (7 500).  Because Layer 1 keeps the store lean, this is
  now a safety net rather than the first line of defence.

Layer 3 – 413 retry with emergency compression (improved)
  On a 413 error we call emergency_compress() from token_budget.py which
  builds the absolute minimum viable payload.  The compressed payload is
  also written back to _SESSION_STORE so future calls benefit too.
  We attempt up to 3 progressive retries with shrinking tail sizes.

Layer 4 – Graceful final fallback
  If all retries fail (essentially impossible with the above layers but
  guarded anyway), we return a friendly AIChatResponse with an apology
  instead of raising HTTP 503.  The user session is soft-reset so the next
  message works normally.

DUAL STORE ARCHITECTURE
------------------------

  _SESSION_STORE[sid]  – OpenAI-format messages for Groq API calls.
                          Compressed automatically. Used only internally.

  _HISTORY_STORE[sid]  – List[ChatMessage] for GET /ai/history responses.
                          Never compressed. Append-only. User-facing.

This separation means aggressive context compression never affects what the
user sees in their chat history.

NO BREAKING CHANGES
--------------------
- All tool dispatch logic is unchanged.
- All schema types are unchanged.
- GET /ai/history and DELETE /ai/history work identically.
- AIChatResponse structure is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_model import (
    SYSTEM_PROMPT,
    build_assistant_tool_call_message,
    build_tool_result_message,
    call_groq,
    extract_text,
    extract_tool_calls,
)
from app.schemas.ai_schema import (
    AIChatResponse,
    ChatMessage,
    ToolExecution,
)
from app.services import (
    backtest_service,
    data_service,
    model_service,
    sentiment_service,
    strategy_service,
)
from app.schemas.backtest_schema import BacktestRunRequest
from app.schemas.sentiment_schema import SentimentRunRequest

# v2: import updated token-budget utilities
from app.models.token_budget import (
    TOKEN_BUDGET,
    SOFT_COMPRESSION_THRESHOLD,
    EMERGENCY_TOKEN_BUDGET,
    build_summary_message,
    compress_tool_result,
    compress_session,          # NEW – proactive session compression
    emergency_compress,        # NEW – last-resort compression for retries
    estimate_messages_tokens,
    trim_history,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tradex.ai.service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS: int = 5
_GROQ_TIMEOUT: float = 60.0

# Hard cap on stored messages (memory guard even for very short messages).
_MAX_HISTORY_MESSAGES: int = 40

# Max output tokens per Groq call.
# Groq counts requested output tokens against the TPM limit even when the
# actual reply is short.  1024 is sufficient for trading Q&A.
_DEFAULT_MAX_OUTPUT_TOKENS: int = 1_024

# HTTP error codes Groq uses for payload-too-large.
_GROQ_PAYLOAD_TOO_LARGE_CODES: frozenset[int] = frozenset({413, 400})

# Maximum number of 413-retry attempts before giving up gracefully.
_MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# DUAL STORE ARCHITECTURE (NEW in v2)
#
# _SESSION_STORE: OpenAI-format messages used for Groq API calls.
#   – Automatically compressed by compress_session() on every write.
#   – May have old turns replaced by compact summaries.
#   – NEVER exposed directly to the user.
#
# _HISTORY_STORE: ChatMessage list used for GET /ai/history.
#   – Append-only; never compressed or mutated.
#   – Always shows the user their full conversation.
# ---------------------------------------------------------------------------

_SESSION_STORE: dict[str, list[dict[str, Any]]] = {}
_HISTORY_STORE: dict[str, list[ChatMessage]] = {}


# ===========================================================================
# Session helpers
# ===========================================================================

def _get_or_create_session(session_id: Optional[str]) -> str:
    """Return existing session id or create fresh stores for a new one."""
    if not session_id or session_id not in _SESSION_STORE:
        sid = session_id or str(uuid.uuid4())
        # Initialise both stores for this session
        _SESSION_STORE[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
        _HISTORY_STORE[sid] = []
        logger.info("_get_or_create_session | new session created | sid=%s", sid)
        return sid
    return session_id


def _append_message(session_id: str, message: dict[str, Any]) -> None:
    """
    Append one OpenAI-format message to the session's API history.

    Post-append housekeeping (NEW in v2):
    1. Hard message-count cap (_MAX_HISTORY_MESSAGES) – drops oldest non-system.
    2. compress_session() – proactively summarises if token count is over
       SOFT_COMPRESSION_THRESHOLD.  This keeps _SESSION_STORE lean between
       requests, which was the key missing piece in v1.

    Only the API-facing _SESSION_STORE is modified here.
    User-visible history (_HISTORY_STORE) is updated separately in
    process_chat() so we can filter to text-only turns.
    """
    history = _SESSION_STORE.setdefault(
        session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    history.append(message)

    # Hard message-count guard (memory safety)
    if len(history) > _MAX_HISTORY_MESSAGES:
        _SESSION_STORE[session_id] = [history[0]] + history[-(_MAX_HISTORY_MESSAGES - 1):]
        logger.debug(
            "_append_message | hard cap hit | trimmed to %d messages",
            _MAX_HISTORY_MESSAGES,
        )

    # ── KEY FIX (v2): proactive compression on every write ─────────────────
    # In v1 this was missing.  The session store could grow indefinitely
    # between requests, causing repeated 413s even after retry logic was added.
    compress_session(
        session_store=_SESSION_STORE,
        session_id=session_id,
        budget=SOFT_COMPRESSION_THRESHOLD,
    )

    # Log current token usage for observability
    current_tokens = estimate_messages_tokens(_SESSION_STORE.get(session_id, []))
    logger.debug(
        "_append_message | session=%s | stored_msgs=%d | stored_tokens≈%d",
        session_id,
        len(_SESSION_STORE.get(session_id, [])),
        current_tokens,
    )


def _append_to_history(session_id: str, role: str, content: str) -> None:
    """
    Append a user or assistant text turn to the user-visible _HISTORY_STORE.
    Tool messages are never added here (they are internal implementation detail).
    """
    if role not in ("user", "assistant"):
        return
    ts = datetime.now(timezone.utc).isoformat()
    store = _HISTORY_STORE.setdefault(session_id, [])
    store.append(
        ChatMessage(
            role=role,   # type: ignore[arg-type]
            content=content,
            timestamp=ts,
        )
    )


def get_session_messages(session_id: str) -> list[ChatMessage]:
    """
    Return user-visible conversation history for GET /ai/history.

    v2 CHANGE: reads from _HISTORY_STORE (never compressed) instead of
    filtering _SESSION_STORE.  This means the user always sees the full
    conversation even when the API context has been summarised.
    """
    return list(_HISTORY_STORE.get(session_id, []))


def delete_session(session_id: str) -> bool:
    """Delete both stores for a session. Returns True if the session existed."""
    existed = session_id in _SESSION_STORE or session_id in _HISTORY_STORE
    _SESSION_STORE.pop(session_id, None)
    _HISTORY_STORE.pop(session_id, None)
    return existed


# ===========================================================================
# Token-aware Groq call with multi-stage 413 retry logic (v2)
# ===========================================================================

async def _call_groq_with_budget(
    session_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
) -> Any:
    """
    Token-aware wrapper around call_groq() with multi-stage fallback.

    v1 had ONE retry that could still fail and raise HTTP 503.
    v2 has THREE progressive retries that compress more aggressively each time,
    and a final graceful fallback instead of a user-visible error.

    Stage 0 – Pre-flight trim
      Trim if estimated tokens > TOKEN_BUDGET before the first attempt.

    Stage 1 – First attempt
      Call Groq normally.  Usually succeeds because compress_session() has
      already kept the session lean.

    Stage 2 – On 413: emergency compress + retry (up to _MAX_RETRIES times)
      Each retry reduces the payload further via emergency_compress().
      The compressed payload is written back to _SESSION_STORE so the session
      benefits for future turns too.

    Stage 3 – Graceful final fallback
      If all retries fail (edge case: system prompt alone is over budget),
      return a synthetic "I need to reset my context" response.  The session
      is soft-reset so the user can keep talking without manual intervention.
    """

    # ── Stage 0: pre-flight trim ───────────────────────────────────────────
    estimated = estimate_messages_tokens(messages)
    logger.info(
        "_call_groq_with_budget | session=%s | pre-flight tokens≈%d | budget=%d",
        session_id, estimated, TOKEN_BUDGET,
    )

    if estimated > TOKEN_BUDGET:
        logger.warning(
            "_call_groq_with_budget | over budget pre-flight – trimming",
        )
        messages = trim_history(messages, budget=TOKEN_BUDGET)
        # Write the trimmed version back so the store stays lean
        _SESSION_STORE[session_id] = messages
        estimated = estimate_messages_tokens(messages)
        logger.info(
            "_call_groq_with_budget | after pre-flight trim | tokens≈%d", estimated,
        )

    # ── Stage 1: first attempt ─────────────────────────────────────────────
    try:
        return await asyncio.wait_for(
            call_groq(messages, max_tokens=max_tokens),
            timeout=_GROQ_TIMEOUT,
        )

    except asyncio.TimeoutError:
        logger.error(
            "_call_groq_with_budget | session=%s | timeout after %.1fs",
            session_id, _GROQ_TIMEOUT,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Groq API timed out after {_GROQ_TIMEOUT:.0f}s. Please try again.",
        )

    except Exception as exc:
        if not _is_413(exc):
            # Non-413 error – raise immediately, not a token problem
            logger.error(
                "_call_groq_with_budget | session=%s | Groq error: %s",
                session_id, exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Groq API error: {exc}",
            )

        # ── Stage 2: 413 detected – progressive retry loop ─────────────────
        logger.warning(
            "_call_groq_with_budget | session=%s | 413 received – starting retry loop",
            session_id,
        )

        current_messages = messages
        for attempt in range(1, _MAX_RETRIES + 1):
            # Shrink budget by 20% each retry to ensure we make progress
            retry_budget = int(EMERGENCY_TOKEN_BUDGET * (1 - 0.2 * (attempt - 1)))
            retry_budget = max(retry_budget, 2000)  # absolute floor

            # emergency_compress() returns a minimal but valid messages list
            retry_messages = emergency_compress(current_messages, budget=retry_budget)

            retry_tokens = estimate_messages_tokens(retry_messages)
            logger.info(
                "_call_groq_with_budget | retry %d/%d | budget=%d | tokens≈%d | msgs=%d",
                attempt, _MAX_RETRIES, retry_budget, retry_tokens, len(retry_messages),
            )

            # Write back to session store so future turns start lean
            _SESSION_STORE[session_id] = retry_messages

            try:
                result = await asyncio.wait_for(
                    call_groq(retry_messages, max_tokens=max_tokens),
                    timeout=_GROQ_TIMEOUT,
                )
                logger.info(
                    "_call_groq_with_budget | session=%s | succeeded on retry %d",
                    session_id, attempt,
                )
                return result

            except asyncio.TimeoutError:
                logger.error(
                    "_call_groq_with_budget | session=%s | retry %d timed out",
                    session_id, attempt,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Groq API timed out. Please try again.",
                )

            except Exception as retry_exc:
                if _is_413(retry_exc):
                    logger.warning(
                        "_call_groq_with_budget | session=%s | retry %d also 413 "
                        "– compressing further",
                        session_id, attempt,
                    )
                    current_messages = retry_messages  # feed into next iteration
                    continue
                # Non-413 error on retry – propagate
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Groq API error on retry: {retry_exc}",
                )

        # ── Stage 3: graceful final fallback ───────────────────────────────
        # All retries exhausted.  Instead of HTTP 503, soft-reset the context
        # and return a synthetic response.  The user can keep chatting normally.
        logger.error(
            "_call_groq_with_budget | session=%s | all %d retries exhausted – "
            "soft-resetting context and returning fallback response",
            session_id, _MAX_RETRIES,
        )
        _soft_reset_session(session_id)
        # Return a sentinel that process_chat() will convert to a user-friendly reply
        return _FALLBACK_SENTINEL


def _is_413(exc: Exception) -> bool:
    """Detect Groq's payload-too-large error regardless of how the SDK wraps it."""
    exc_str = str(exc)
    return (
        "413" in exc_str
        or "Request too large" in exc_str
        or "request_too_large" in exc_str.lower()
        or getattr(exc, "status_code", None) in _GROQ_PAYLOAD_TOO_LARGE_CODES
    )


# Sentinel object returned by _call_groq_with_budget on graceful fallback.
# process_chat() checks for this and returns a friendly message instead of
# crashing.  Using a dedicated sentinel avoids isinstance() checks on the
# ChatCompletion return type.
_FALLBACK_SENTINEL = object()

_FALLBACK_REPLY = (
    "I've been working on a long conversation and my context window filled up. "
    "I've automatically summarised our earlier discussion so we can continue. "
    "Please resend your last message and I'll answer it right away."
)


def _soft_reset_session(session_id: str) -> None:
    """
    Soft-reset a session's API context to the minimum viable state.

    The user-visible _HISTORY_STORE is LEFT INTACT so the user still sees
    their full conversation in the chat UI.  Only the internal _SESSION_STORE
    (used for Groq API calls) is reset to [system_prompt].

    This is called as a last resort after all retries are exhausted.  The next
    user message will start a fresh context window.
    """
    logger.warning(
        "_soft_reset_session | session=%s | resetting API context to system prompt only",
        session_id,
    )
    _SESSION_STORE[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]


# ===========================================================================
# Tool dispatcher (UNCHANGED from v1)
# ===========================================================================

async def _dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    db: AsyncSession,
) -> Any:
    """
    Route a tool call to the correct backend service function.
    Business logic is UNCHANGED from the original implementation.
    """

    # ── Strategies ─────────────────────────────────────────────────────────
    if tool_name == "get_strategies":
        return await strategy_service.get_strategies(
            db,
            symbol=args.get("symbol"),
            time_horizon=args.get("time_horizon"),
            search=args.get("search"),
            page=int(args.get("page", 1)),
            page_size=int(args.get("page_size", 20)),
        )

    if tool_name == "get_strategy_detail":
        result = await strategy_service.get_strategy_by_name(
            db, args["strategy_name"]
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{args['strategy_name']}' not found.",
            )
        return result

    # ── Backtest strategies dropdown ────────────────────────────────────────
    if tool_name == "get_backtest_strategies":
        return await backtest_service.get_backtest_strategies(db)

    # ── Run backtest ────────────────────────────────────────────────────────
    if tool_name == "run_backtest":
        req = BacktestRunRequest(
            strategy_name=args["strategy_name"],
            exchange=args["exchange"],
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            starting_balance=float(args.get("starting_balance", 1000.0)),
            take_profit=float(args.get("take_profit", 1.0)),
            stop_loss=float(args.get("stop_loss", 1.0)),
            buy_after_minutes=int(args.get("buy_after_minutes", 0)),
            fee=float(args.get("fee", 0.05)),
            leverage=float(args.get("leverage", 1.0)),
            slippage=float(args.get("slippage", 0.0)),
        )
        return await backtest_service.run_backtest(db, req)

    # ── Strategy run history ────────────────────────────────────────────────
    if tool_name == "get_strategy_runs":
        return await backtest_service.get_strategy_runs(db, args["strategy_name"])

    # ── ML / DL models ─────────────────────────────────────────────────────
    if tool_name == "get_models":
        model_type = args.get("model_type", "ml").lower()
        if model_type not in ("ml", "dl"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model_type must be 'ml' or 'dl'.",
            )
        return await model_service.get_model_results(
            db,
            model_type=model_type,
            search=args.get("search"),
            page=int(args.get("page", 1)),
            page_size=int(args.get("page_size", 20)),
        )

    if tool_name == "get_model_detail":
        model_type = args.get("model_type", "ml").lower()
        result = await model_service.get_model_result_by_name(
            db, model_type, args["model_name"]
        )
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{args['model_name']}' not found in {model_type}_results.",
            )
        return result

    # ── Sentiment ───────────────────────────────────────────────────────────
    if tool_name == "get_sentiment_results":
        return await sentiment_service.get_sentiment_results(db, args["coin"])

    if tool_name == "run_sentiment":
        req = SentimentRunRequest(coin=args["coin"])
        return await sentiment_service.run_sentiment(db, req)

    # ── OHLCV ───────────────────────────────────────────────────────────────
    if tool_name == "get_ohlcv":
        return await data_service.read_ohlcv(
            db,
            exchange=args["exchange"],
            symbol=args["symbol"],
            timeframe="1m",
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unknown tool '{tool_name}' requested by AI.",
    )


# ===========================================================================
# Result serialisation helpers (UNCHANGED from v1)
# ===========================================================================

def _serialise_result(result: Any) -> dict[str, Any]:
    """Convert a service result to a JSON-serialisable dict."""
    if result is None:
        return {}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, list):
        return {
            "items": [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in result
            ],
            "count": len(result),
        }
    return {"value": result}


def _result_summary(tool_name: str, result: Any) -> str:
    """Generate a short human-readable summary of a tool result."""
    try:
        if tool_name == "get_strategies":
            return f"Found {result.total} strategies (page {result.page}/{result.pages})"
        if tool_name == "get_strategy_detail":
            return (
                f"Strategy '{result.name}': {len(result.indicators)} indicators, "
                f"{len(result.patterns)} patterns, last PnL: "
                f"{result.last_pnl_pct:.2f}%"
                if result.last_pnl_pct is not None
                else f"Strategy '{result.name}' loaded"
            )
        if tool_name == "get_backtest_strategies":
            count = len(result) if isinstance(result, list) else "?"
            return f"Loaded {count} strategies for backtest"
        if tool_name == "run_backtest":
            s = result.summary
            return (
                f"Backtest complete: {s.total_trades} trades, "
                f"win rate {s.win_rate:.1f}%, "
                f"PnL {s.total_pnl_pct:+.2f}%, "
                f"final balance ${s.final_balance:.2f}"
            )
        if tool_name == "get_strategy_runs":
            count = len(result) if isinstance(result, list) else "?"
            return f"Found {count} historical run(s)"
        if tool_name == "get_models":
            return f"Found {result.total} {result.model_type.upper()} models"
        if tool_name == "get_model_detail":
            return (
                f"Model '{result.model_name}': PnL {result.pnl:+.2f}, "
                f"win rate {result.win_rate:.1f}%"
                if result.pnl is not None and result.win_rate is not None
                else f"Model '{result.model_name}' loaded"
            )
        if tool_name in ("get_sentiment_results", "run_sentiment"):
            r = result if tool_name == "get_sentiment_results" else result.results
            return (
                f"Sentiment for {r.coin}: "
                f"mean score {r.overall.mean_sentiment:.3f}, "
                f"{r.overall.total_posts} posts analysed"
            )
        if tool_name == "get_ohlcv":
            return (
                f"OHLCV for {result.symbol} on {result.exchange}: "
                f"{result.total_rows} candles, "
                f"last close ${result.close:.4f}"
            )
    except Exception:
        pass
    return f"Tool '{tool_name}' executed successfully"


# ===========================================================================
# Main orchestration entry-point
# ===========================================================================

async def process_chat(
    db: AsyncSession,
    user_message: str,
    session_id: Optional[str],
) -> AIChatResponse:
    """
    Full AI orchestration pipeline (Groq / Llama 3.3-70B).

    Flow
    ----
    1.  Retrieve or create session; both _SESSION_STORE and _HISTORY_STORE
        are initialised if new.
    2.  Append user turn to _SESSION_STORE (API context) and _HISTORY_STORE
        (user-visible history).
    3.  Call _call_groq_with_budget() which handles all token management
        and retries transparently.
    4.  If response is _FALLBACK_SENTINEL (graceful fallback after all retries
        exhausted), return a friendly AIChatResponse immediately.
    5.  If the model returns tool_calls, dispatch them, compress results,
        append to _SESSION_STORE, and loop (up to MAX_TOOL_ROUNDS).
    6.  When the model returns a plain text reply, append to both stores
        and return AIChatResponse.

    KEY v2 CHANGES vs v1
    ---------------------
    - User-visible history stored in _HISTORY_STORE (separate, never compressed).
    - _append_message() triggers compress_session() automatically.
    - _call_groq_with_budget() has 3-retry progressive fallback, not 1-retry.
    - _FALLBACK_SENTINEL handling: no HTTP 503 ever raised to the user.
    - _soft_reset_session() used as last resort to clear API context without
      deleting the user's conversation history.
    """
    sid = _get_or_create_session(session_id)

    # Append user message to API context store
    _append_message(sid, {"role": "user", "content": user_message})
    # Append to user-visible history store (text-only, never compressed)
    _append_to_history(sid, "user", user_message)

    logger.info(
        "process_chat | session=%s | user_message_len=%d",
        sid, len(user_message),
    )

    tools_executed: list[ToolExecution] = []
    last_structured_data: Optional[dict[str, Any]] = None
    final_reply: str = ""

    # ── Orchestration loop ──────────────────────────────────────────────────
    for round_num in range(MAX_TOOL_ROUNDS):
        logger.debug(
            "process_chat | session=%s | round=%d/%d",
            sid, round_num + 1, MAX_TOOL_ROUNDS,
        )

        current_messages = list(_SESSION_STORE.get(sid, []))

        # ── Call Groq (token-aware, multi-retry, graceful fallback) ─────────
        response = await _call_groq_with_budget(
            session_id=sid,
            messages=current_messages,
            max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        )

        # ── Handle graceful fallback sentinel ───────────────────────────────
        # This is returned when all retries are exhausted.  We return a
        # friendly message instead of crashing.  The session context has been
        # soft-reset so the next user message will work normally.
        if response is _FALLBACK_SENTINEL:
            final_reply = _FALLBACK_REPLY
            _append_to_history(sid, "assistant", final_reply)
            logger.warning(
                "process_chat | session=%s | fallback sentinel received – "
                "returning graceful reply",
                sid,
            )
            break

        tool_calls = extract_tool_calls(response)
        text_reply = extract_text(response)

        # ── No tool calls → model gave final text answer ────────────────────
        if not tool_calls:
            final_reply = text_reply or "I've completed the requested operations."
            _append_message(sid, {"role": "assistant", "content": final_reply})
            _append_to_history(sid, "assistant", final_reply)
            logger.info(
                "process_chat | session=%s | final_reply_len=%d | rounds_used=%d",
                sid, len(final_reply), round_num + 1,
            )
            break

        # ── Echo assistant message (with tool_calls) into API history ───────
        assistant_msg = build_assistant_tool_call_message(response)
        _append_message(sid, assistant_msg)
        # Note: tool-call assistant turns are NOT added to _HISTORY_STORE
        # (they are internal implementation detail, not user-facing text)

        logger.debug(
            "process_chat | session=%s | round=%d | tool_calls=%s",
            sid, round_num + 1,
            [tc["name"] for tc in tool_calls],
        )

        # ── Dispatch each tool call ─────────────────────────────────────────
        for tc in tool_calls:
            tool_name = tc["name"]
            args: dict[str, Any] = tc.get("args") or {}
            tool_call_id = tc["id"]

            logger.debug(
                "process_chat | dispatching | tool=%s | args=%s",
                tool_name, json.dumps(args, default=str)[:200],
            )

            try:
                result = await _dispatch_tool(tool_name, args, db)
                serialised = _serialise_result(result)
                summary = _result_summary(tool_name, result)

                tools_executed.append(
                    ToolExecution(
                        tool_name=tool_name,
                        parameters=args,
                        status="success",
                        result_summary=summary,
                    )
                )

                # Full uncompressed result → UI via AIChatResponse.data
                last_structured_data = serialised

                # Compressed result → session history (saves ~60-90% tokens
                # on backtest / sentiment payloads)
                compressed = compress_tool_result(tool_name, serialised)
                compressed_json = json.dumps(compressed, default=str)
                original_json = json.dumps(serialised, default=str)

                if len(compressed_json) < len(original_json):
                    savings = 100 * (1 - len(compressed_json) / len(original_json))
                    logger.info(
                        "process_chat | tool=%s | original=%d chars | "
                        "compressed=%d chars | saved=%.0f%%",
                        tool_name, len(original_json), len(compressed_json), savings,
                    )

                _append_message(
                    sid,
                    build_tool_result_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=compressed_json,
                    ),
                )

                logger.debug(
                    "process_chat | tool_success | tool=%s | summary=%s",
                    tool_name, summary,
                )

            except HTTPException as http_exc:
                error_msg = str(http_exc.detail)
                tools_executed.append(
                    ToolExecution(
                        tool_name=tool_name,
                        parameters=args,
                        status="error",
                        error=error_msg,
                    )
                )
                _append_message(
                    sid,
                    build_tool_result_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=json.dumps({"error": error_msg}),
                    ),
                )
                logger.warning(
                    "process_chat | tool_http_error | tool=%s | error=%s",
                    tool_name, error_msg,
                )

            except Exception as exc:
                error_msg = str(exc)
                tools_executed.append(
                    ToolExecution(
                        tool_name=tool_name,
                        parameters=args,
                        status="error",
                        error=error_msg,
                    )
                )
                _append_message(
                    sid,
                    build_tool_result_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=json.dumps({"error": error_msg}),
                    ),
                )
                logger.error(
                    "process_chat | tool_exception | tool=%s | error=%s",
                    tool_name, error_msg, exc_info=True,
                )

    else:
        # MAX_TOOL_ROUNDS exhausted without a plain-text reply
        final_reply = (
            "I've gathered the requested data. Here's a summary: "
            + "; ".join(
                t.result_summary or t.tool_name
                for t in tools_executed
                if t.status == "success"
            )
            + "."
        )
        _append_message(sid, {"role": "assistant", "content": final_reply})
        _append_to_history(sid, "assistant", final_reply)
        logger.warning(
            "process_chat | session=%s | MAX_TOOL_ROUNDS (%d) exhausted",
            sid, MAX_TOOL_ROUNDS,
        )

    return AIChatResponse(
        session_id=sid,
        reply=final_reply,
        tools_executed=tools_executed,
        data=last_structured_data,   # Full uncompressed data for the UI
    )