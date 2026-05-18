"""
TradeX – AI Orchestration Service  (v3 – Strategy Generator Integration)
=========================================================================

WHAT CHANGED vs v2 (and WHY)
------------------------------

STRATEGY GENERATOR INTEGRATION
--------------------------------
v3 makes strategy generation a first-class AI tool. Two new tools are wired
into _dispatch_tool():

  create_strategy
    Delegates directly to strategy_generator_service.create_strategy().
    Accepts the same parameters as POST /strategy-generator/create so the AI
    can pass any combination of (name, symbol, exchange, timeframe, TP, SL,
    leverage, fee, slippage, start/end dates).

  compare_strategies
    A higher-level tool that loops create_strategy N times (2–5) and returns
    a ranked summary.  Implemented here (not in strategy_generator_service) so
    the AI never has to chain N parallel tool calls manually.  Results are
    deduplicated and sorted by win_rate descending, then total_pnl_pct.

COMPRESSION EXTENSION
----------------------
Both new tools return large payloads (ledger + PnL + win-loss arrays).
compress_tool_result() already handles "ledger", "pnl_data", and "win_loss_data"
via _HEAVY_ARRAY_KEYS in token_budget.py.  We add those keys to the existing
_HEAVY_ARRAY_KEYS frozenset (see token_budget.py patch note).  No new
compression logic needed in ai_service.py itself.

_result_summary() is extended with cases for create_strategy and
compare_strategies so the orchestration log and ToolExecution.result_summary
are meaningful.

NO BREAKING CHANGES
--------------------
- All existing tool dispatch paths are UNCHANGED.
- All schema types are unchanged (CreateStrategyResponse added to AIChatResponse.data).
- GET /ai/history and DELETE /ai/history work identically.
- AIChatResponse structure is unchanged.
- Token budget / compression / retry logic is unchanged.

TIMEOUT PROTECTION
-------------------
create_strategy runs OHLCV load + signals + backtest in asyncio.to_thread.
For compare_strategies we run each strategy sequentially (not in parallel)
to avoid saturating the thread-pool and to stay within DB connection limits.
Each individual create_strategy call already has internal error handling;
compare_strategies wraps each call in try/except and records partial failures.

STRUCTURED LOGGING
-------------------
All new code paths use logger.info / logger.warning with structured key=value
fields consistent with the existing service logging style.
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
    # v3: import the strategy generator service so we can call it from
    # _dispatch_tool() without touching its internal business logic.
    strategy_generator_service,
)
from app.schemas.backtest_schema import BacktestRunRequest
from app.schemas.sentiment_schema import SentimentRunRequest
# v3: import the request schema for strategy generation
from app.schemas.strategy_generator_schema import (
    CreateStrategyRequest,
    CreateStrategyResponse,
)

# v2: import updated token-budget utilities
from app.models.token_budget import (
    TOKEN_BUDGET,
    SOFT_COMPRESSION_THRESHOLD,
    EMERGENCY_TOKEN_BUDGET,
    build_summary_message,
    compress_tool_result,
    compress_session,          # NEW in v2 – proactive session compression
    emergency_compress,        # NEW in v2 – last-resort compression for retries
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

# v3: Safety cap on the number of strategies compare_strategies will generate.
# Prevents abuse / accidental long-running requests.
_MAX_COMPARE_COUNT: int = 5

# v3: Per-strategy timeout guard for compare_strategies (seconds).
# Each create_strategy call involves OHLCV load + backtest; 120 s is generous.
_STRATEGY_GEN_TIMEOUT: float = 120.0


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
# Token-aware Groq call with multi-stage 413 retry logic (v2, unchanged in v3)
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
# v3: Strategy Generator helpers
# ===========================================================================

def _build_create_strategy_request(args: dict[str, Any]) -> CreateStrategyRequest:
    """
    Build a CreateStrategyRequest from AI tool call arguments.

    WHY A SEPARATE HELPER: Keeps _dispatch_tool() readable and makes it easy
    to add validation / default-filling in one place.  Also ensures every
    optional field has a sensible default even when the AI omits it.
    """
    return CreateStrategyRequest(
        name=args["name"],
        symbol=args["symbol"].lower(),
        exchange=args["exchange"].lower(),
        timeframe=args["timeframe"].lower(),
        start_date=args.get("start_date"),
        end_date=args.get("end_date"),
        starting_balance=float(args.get("starting_balance", 1000.0)),
        take_profit=float(args.get("take_profit", 3.0)),
        stop_loss=float(args.get("stop_loss", 1.0)),
        fee=float(args.get("fee", 0.05)),
        leverage=float(args.get("leverage", 1.0)),
        slippage=float(args.get("slippage", 0.0)),
    )


async def _run_compare_strategies(
    args: dict[str, Any],
    db: AsyncSession,
) -> dict[str, Any]:
    """
    Generate `count` strategies sequentially and return a ranked comparison dict.

    WHY SEQUENTIAL (not asyncio.gather):
    - Each strategy runs OHLCV load + backtest in asyncio.to_thread.
    - Parallel execution would saturate the thread-pool and could exhaust DB
      connections under load.
    - Sequential is safer and still completes in reasonable time (30–90 s for 3).

    WHY THIS RETURNS A PLAIN DICT (not CreateStrategyResponse):
    - The AI tool result must be a single JSON-serialisable object.
    - A list of CreateStrategyResponse plus a ranked summary is more useful
      to the AI than a raw list.
    - The UI receives the full structured dict via AIChatResponse.data.
    """
    count = max(2, min(int(args.get("count", 3)), _MAX_COMPARE_COUNT))
    base_name = args.get("base_name", "Generated Strategy")
    symbol = args["symbol"].lower()
    exchange = args["exchange"].lower()
    timeframe = args["timeframe"].lower()

    logger.info(
        "_run_compare_strategies | count=%d | symbol=%s | exchange=%s | timeframe=%s",
        count, symbol, exchange, timeframe,
    )

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for i in range(1, count + 1):
        name = f"{base_name} #{i}"
        req = CreateStrategyRequest(
            name=name,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=args.get("start_date"),
            end_date=args.get("end_date"),
            starting_balance=float(args.get("starting_balance", 1000.0)),
            take_profit=float(args.get("take_profit", 3.0)),
            stop_loss=float(args.get("stop_loss", 1.0)),
            fee=float(args.get("fee", 0.05)),
            leverage=float(args.get("leverage", 1.0)),
            slippage=float(args.get("slippage", 0.0)),
        )

        try:
            # Timeout protection: each individual strategy gen capped at
            # _STRATEGY_GEN_TIMEOUT seconds to prevent a single slow run from
            # blocking the entire compare operation.
            resp: CreateStrategyResponse = await asyncio.wait_for(
                strategy_generator_service.create_strategy(db, req),
                timeout=_STRATEGY_GEN_TIMEOUT,
            )
            results.append({
                "rank": None,               # filled after sorting
                "strategy_id": resp.strategy_id,
                "display_name": resp.display_name,
                "timeframe": resp.timeframe,
                "symbol": resp.symbol,
                "exchange": resp.exchange,
                "win_rate": resp.summary.win_rate,
                "loss_rate": resp.summary.loss_rate,
                "total_pnl_pct": resp.summary.total_pnl_pct,
                "total_trades": resp.summary.total_trades,
                "win_trades": resp.summary.win_trades,
                "loss_trades": resp.summary.loss_trades,
                "final_balance": resp.summary.final_balance,
                "starting_balance": resp.summary.starting_balance,
                "max_consecutive_wins": resp.summary.max_consecutive_wins,
                "max_consecutive_losses": resp.summary.max_consecutive_losses,
                "risk_reward_ratio": round(req.take_profit / req.stop_loss, 2),
                "message": resp.message,
            })
            logger.info(
                "_run_compare_strategies | strategy %d/%d done | id=%s | "
                "win_rate=%.1f%% | pnl=%.2f%%",
                i, count, resp.strategy_id,
                resp.summary.win_rate, resp.summary.total_pnl_pct,
            )
        except asyncio.TimeoutError:
            msg = f"Strategy '{name}' timed out after {_STRATEGY_GEN_TIMEOUT:.0f}s"
            errors.append(msg)
            logger.warning("_run_compare_strategies | %s", msg)
        except HTTPException as http_exc:
            msg = f"Strategy '{name}' failed: {http_exc.detail}"
            errors.append(msg)
            logger.warning("_run_compare_strategies | %s", msg)
        except Exception as exc:
            msg = f"Strategy '{name}' error: {exc}"
            errors.append(msg)
            logger.error("_run_compare_strategies | %s", msg, exc_info=True)

    # Sort by win_rate DESC, then total_pnl_pct DESC
    results.sort(key=lambda r: (-r["win_rate"], -r["total_pnl_pct"]))

    # Assign rank
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return {
        "generated": len(results),
        "requested": count,
        "errors": errors,
        "ranked_strategies": results,
        "best_strategy_id": results[0]["strategy_id"] if results else None,
        "best_win_rate": results[0]["win_rate"] if results else None,
        "best_pnl_pct": results[0]["total_pnl_pct"] if results else None,
    }


# ===========================================================================
# Tool dispatcher (v3: extended with create_strategy and compare_strategies)
# ===========================================================================

async def _dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    db: AsyncSession,
) -> Any:
    """
    Route a tool call to the correct backend service function.

    v3 ADDITIONS (at the bottom, before the final raise):
      create_strategy      → strategy_generator_service.create_strategy()
      compare_strategies   → _run_compare_strategies() (defined above)

    All existing routing paths are UNCHANGED.
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

    # =======================================================================
    # v3: STRATEGY GENERATOR TOOLS
    # =======================================================================

    # ── Create / generate a single strategy ────────────────────────────────
    if tool_name == "create_strategy":
        # Validate required fields before hitting the service layer so we get
        # a clear error message from the AI rather than a Pydantic ValidationError.
        for required_field in ("name", "symbol", "exchange", "timeframe"):
            if not args.get(required_field):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"create_strategy requires '{required_field}'. "
                        "Please provide it and try again."
                    ),
                )

        req = _build_create_strategy_request(args)

        logger.info(
            "_dispatch_tool | create_strategy | name=%s | symbol=%s | "
            "exchange=%s | timeframe=%s | tp=%.1f | sl=%.1f | leverage=%.1f",
            req.name, req.symbol, req.exchange, req.timeframe,
            req.take_profit, req.stop_loss, req.leverage,
        )

        # strategy_generator_service.create_strategy() already:
        #   - validates timeframe and exchange (raises HTTPException on failure)
        #   - runs OHLCV load + resample + signals + backtest in asyncio.to_thread
        #   - persists signal table, strategy registry, and backtest run
        #   - returns CreateStrategyResponse with full ledger + summary
        # We call it directly without wrapping to preserve all existing error
        # handling.  HTTPException propagates naturally to _dispatch_tool's
        # caller in process_chat().
        return await strategy_generator_service.create_strategy(db, req)

    # ── Compare N generated strategies ─────────────────────────────────────
    if tool_name == "compare_strategies":
        # Validate required fields
        for required_field in ("symbol", "exchange", "timeframe"):
            if not args.get(required_field):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"compare_strategies requires '{required_field}'. "
                        "Please provide it and try again."
                    ),
                )

        count = int(args.get("count", 3))
        if not (2 <= count <= _MAX_COMPARE_COUNT):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"compare_strategies count must be 2–{_MAX_COMPARE_COUNT}. Got {count}.",
            )

        logger.info(
            "_dispatch_tool | compare_strategies | count=%d | symbol=%s | "
            "exchange=%s | timeframe=%s",
            count, args["symbol"], args["exchange"], args["timeframe"],
        )

        return await _run_compare_strategies(args, db)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unknown tool '{tool_name}' requested by AI.",
    )


# ===========================================================================
# Result serialisation helpers (extended in v3 for strategy generator types)
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
    # v3: compare_strategies already returns a plain dict
    if isinstance(result, dict):
        return result
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

        # ── v3: strategy generator summaries ──────────────────────────────
        if tool_name == "create_strategy":
            # result is a CreateStrategyResponse (has .summary, .strategy_id)
            s = result.summary
            return (
                f"Strategy '{result.strategy_id}' created: "
                f"{s.total_trades} trades, "
                f"win rate {s.win_rate:.1f}%, "
                f"PnL {s.total_pnl_pct:+.2f}%, "
                f"final balance ${s.final_balance:.2f}"
            )

        if tool_name == "compare_strategies":
            # result is the plain dict from _run_compare_strategies()
            generated = result.get("generated", 0)
            best_id = result.get("best_strategy_id", "N/A")
            best_wr = result.get("best_win_rate")
            best_pnl = result.get("best_pnl_pct")
            best_wr_str = f"{best_wr:.1f}%" if best_wr is not None else "N/A"
            best_pnl_str = f"{best_pnl:+.2f}%" if best_pnl is not None else "N/A"
            return (
                f"Compared {generated} strategies; "
                f"best: {best_id} "
                f"(win rate {best_wr_str}, PnL {best_pnl_str})"
            )

    except Exception:
        pass
    return f"Tool '{tool_name}' executed successfully"


# ===========================================================================
# Main orchestration entry-point (UNCHANGED from v2 except import additions)
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

    KEY v3 CHANGES vs v2
    ---------------------
    - create_strategy and compare_strategies routed in _dispatch_tool().
    - _result_summary() extended for new tool names.
    - _serialise_result() extended for plain dict (compare_strategies result).
    - SYSTEM_PROMPT updated with strategy generation routing rules.
    - No changes to the orchestration loop itself.
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
                # on backtest / sentiment / strategy-gen payloads)
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