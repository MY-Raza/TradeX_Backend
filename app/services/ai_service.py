"""
TradeX – AI Orchestration Service (Groq / Llama 3.3-70B)
=========================================================

Responsibilities
----------------
1.  Maintain per-session conversation history in-memory (keyed by session_id)
2.  Receive a user prompt and build a full OpenAI-format messages list
3.  Pre-flight token estimation before every Groq call
4.  Send the (trimmed) messages to Groq via the OpenAI-compatible SDK
5.  Parse ``tool_calls`` from the response and dispatch to backend services
6.  Compress tool results before storing them in session history
7.  Inject tool results back as ``role="tool"`` messages and loop
8.  Return AIChatResponse with the final text reply + tool execution trace

TOKEN OPTIMISATION CHANGES (vs original)
-----------------------------------------
| Problem                            | Fix                                        |
|------------------------------------|--------------------------------------------|
| _MAX_HISTORY_MESSAGES=50 by count  | Budget managed in TOKENS not message count |
| Tool results stored raw (~15K tok) | Compressed via compress_tool_result()      |
| max_tokens=4096 per call           | Lowered to 1024 (overridable)              |
| No pre-flight token check          | estimate_messages_tokens() before each call|
| 413 crashes request                | Retry loop with trim → summarise → retry   |
| Verbose system prompt              | Compact SYSTEM_PROMPT (~110 tokens)        |

Groq 413 error handling
------------------------
When a 413 (request too large) is returned we:
  1. Trim history via trim_history() to drop old messages.
  2. If still too large, inject a compact summary and keep only the tail.
  3. Retry the call once.  If it fails again we raise a 503 with a clear msg.

This is transparent to the user – they never see the 413; they get the answer.
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

# NEW: import token-budget utilities
from app.models.token_budget import (
    TOKEN_BUDGET,
    build_summary_message,
    compress_tool_result,
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

MAX_TOOL_ROUNDS: int = 5       # maximum tool-call → result cycles per request
_GROQ_TIMEOUT: float = 60.0    # seconds per Groq API call

# CHANGED: history is now governed by TOKEN_BUDGET (from token_budget.py),
# not by a fixed message count.  _MAX_HISTORY_MESSAGES is kept as a hard cap
# to bound memory even if individual messages are very short.
_MAX_HISTORY_MESSAGES: int = 40

# Max tokens to request from the model.
# Lower = fewer TPM consumed = less risk of 413.
# For a trading Q&A assistant 1 024 output tokens is almost always enough.
_DEFAULT_MAX_OUTPUT_TOKENS: int = 1_024

# Groq HTTP error codes that indicate a payload-too-large condition.
_GROQ_PAYLOAD_TOO_LARGE_CODES: frozenset[int] = frozenset({413, 400})

# ---------------------------------------------------------------------------
# In-memory session store
# Maps session_id → list[dict] in OpenAI messages format.
# Replace with Redis for multi-instance production deployments.
# ---------------------------------------------------------------------------

_SESSION_STORE: dict[str, list[dict[str, Any]]] = {}


# ===========================================================================
# Session helpers
# ===========================================================================

def _get_or_create_session(session_id: Optional[str]) -> str:
    """Return existing session id or create a new one."""
    if not session_id or session_id not in _SESSION_STORE:
        sid = session_id or str(uuid.uuid4())
        _SESSION_STORE[sid] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return sid
    return session_id


def _append_message(session_id: str, message: dict[str, Any]) -> None:
    """
    Append one OpenAI-format message to the session history.

    Trimming strategy (CHANGED from original):
    - Hard cap: if message count exceeds _MAX_HISTORY_MESSAGES, drop oldest
      non-system messages first (same as before).
    - Token cap: after appending, if estimated tokens exceed TOKEN_BUDGET,
      run trim_history() to bring it back under budget.

    The system message at index 0 is always preserved.
    """
    history = _SESSION_STORE.setdefault(
        session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    history.append(message)

    # Hard message-count cap (memory guard)
    if len(history) > _MAX_HISTORY_MESSAGES:
        _SESSION_STORE[session_id] = [history[0]] + history[-(
            _MAX_HISTORY_MESSAGES - 1
        ):]
        history = _SESSION_STORE[session_id]

    # Token cap – run trim only when we're clearly over budget
    current_tokens = estimate_messages_tokens(history)
    if current_tokens > TOKEN_BUDGET:
        _SESSION_STORE[session_id] = trim_history(history, budget=TOKEN_BUDGET)


def get_session_messages(session_id: str) -> list[ChatMessage]:
    """
    Return conversation history as ChatMessage objects for GET /ai/history.
    Filters to user and assistant text turns; skips tool messages.
    """
    history = _SESSION_STORE.get(session_id, [])
    messages: list[ChatMessage] = []
    ts = datetime.now(timezone.utc).isoformat()
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            messages.append(
                ChatMessage(
                    role=role,  # type: ignore[arg-type]
                    content=content,
                    timestamp=ts,
                )
            )
    return messages


def delete_session(session_id: str) -> bool:
    """Delete a session from the store. Returns True if it existed."""
    if session_id in _SESSION_STORE:
        del _SESSION_STORE[session_id]
        return True
    return False


# ===========================================================================
# Token-aware Groq call with 413 retry logic
# ===========================================================================

async def _call_groq_with_budget(
    session_id: str,
    messages: list[dict[str, Any]],
    max_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
) -> Any:
    """
    Wrapper around call_groq() that:

    1. Estimates token count before sending.
    2. If estimate exceeds TOKEN_BUDGET, trims history pre-emptively.
    3. Calls Groq with asyncio.wait_for timeout.
    4. On 413/400 (payload too large): trims aggressively, injects a summary,
       and retries ONCE.
    5. On repeated failure: raises HTTP 503 with a human-readable message.

    This is the SINGLE place where all Groq calls go – ai_service never calls
    call_groq() directly anymore.

    Parameters
    ----------
    session_id : Used only for logging.
    messages   : The full messages list to send (may be mutated by retry).
    max_tokens : Passed to call_groq(); lower = safer for TPM budget.
    """

    # ── Pre-flight estimate ────────────────────────────────────────────────
    estimated = estimate_messages_tokens(messages)
    logger.info(
        "_call_groq_with_budget | session=%s | estimated_tokens=%d | budget=%d",
        session_id, estimated, TOKEN_BUDGET,
    )

    if estimated > TOKEN_BUDGET:
        logger.warning(
            "_call_groq_with_budget | pre-flight over budget – trimming before call",
        )
        messages = trim_history(messages, budget=TOKEN_BUDGET)
        estimated = estimate_messages_tokens(messages)
        logger.info(
            "_call_groq_with_budget | after pre-flight trim | tokens=%d", estimated,
        )

    # ── First attempt ──────────────────────────────────────────────────────
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
            detail=f"Groq API timed out after {_GROQ_TIMEOUT:.0f}s.",
        )

    except Exception as exc:
        # ── 413 / payload-too-large detection ─────────────────────────────
        # The openai SDK wraps HTTP errors; the status_code lives on the
        # exception body or as exc.status_code / exc.response.status_code.
        exc_str = str(exc)
        is_413 = (
            "413" in exc_str
            or "Request too large" in exc_str
            or "request_too_large" in exc_str.lower()
            or getattr(exc, "status_code", None) in _GROQ_PAYLOAD_TOO_LARGE_CODES
        )

        if not is_413:
            logger.error(
                "_call_groq_with_budget | session=%s | Groq error: %s",
                session_id, exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Groq API error: {exc}",
            )

        # ── Retry path: aggressive trim + summary injection ────────────────
        logger.warning(
            "_call_groq_with_budget | session=%s | 413 received – retrying with "
            "aggressive trim + summary",
            session_id,
        )

        # Build a compact summary of the old history before we discard it
        system_msg = messages[0]
        non_system = messages[1:]

        summary_msg = build_summary_message(non_system)

        # Keep only the last 3 messages (most recent user + assistant exchange)
        recent_tail = non_system[-3:] if len(non_system) >= 3 else non_system

        # Reconstruct: system → summary → recent tail
        retry_messages = [system_msg, summary_msg] + recent_tail

        retry_tokens = estimate_messages_tokens(retry_messages)
        logger.info(
            "_call_groq_with_budget | retry | tokens=%d | messages=%d",
            retry_tokens, len(retry_messages),
        )

        try:
            return await asyncio.wait_for(
                call_groq(retry_messages, max_tokens=max_tokens),
                timeout=_GROQ_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq API timed out on retry.",
            )
        except Exception as retry_exc:
            logger.error(
                "_call_groq_with_budget | session=%s | retry also failed: %s",
                session_id, retry_exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The conversation history is too long for the current Groq plan. "
                    "Please start a new session or delete this session's history "
                    "via DELETE /ai/history/{session_id}."
                ),
            )


# ===========================================================================
# Tool dispatcher
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
# Result serialisation helpers (unchanged from original)
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
    1.  Retrieve or create a session; load existing OpenAI-format history.
    2.  Append the new user turn to history.
    3.  Snapshot history + run token pre-flight check via _call_groq_with_budget().
    4.  If the model returns ``tool_calls``:
          a.  Echo the assistant message (with tool_calls) back into history.
          b.  Dispatch each tool call to the correct backend service.
          c.  Compress the tool result (strip chart arrays) before storing.
          d.  Append each result as a ``role="tool"`` message.
          e.  Repeat from step 3 (up to MAX_TOOL_ROUNDS times).
    5.  When the model returns a plain text reply, persist history and return.

    KEY CHANGES vs original
    -----------------------
    - call_groq() is replaced by _call_groq_with_budget() everywhere.
    - Tool results are compressed via compress_tool_result() before being
      stored in session history (the uncompressed version is still returned
      to the UI via last_structured_data / AIChatResponse.data).
    - Token estimation + trim happens proactively on every round, not just
      reactively on error.
    """
    sid = _get_or_create_session(session_id)

    # Append the incoming user message
    _append_message(sid, {"role": "user", "content": user_message})
    logger.info(
        "process_chat | session=%s | user_message_len=%d",
        sid, len(user_message),
    )

    tools_executed: list[ToolExecution] = []
    last_structured_data: Optional[dict[str, Any]] = None
    final_reply: str = ""

    # ── Orchestration loop ─────────────────────────────────────────────────
    for round_num in range(MAX_TOOL_ROUNDS):
        logger.debug(
            "process_chat | session=%s | round=%d/%d",
            sid, round_num + 1, MAX_TOOL_ROUNDS,
        )

        # Snapshot current history for this round's API call
        current_messages = list(_SESSION_STORE.get(sid, []))

        # ── Call Groq (token-aware, with 413 retry) ────────────────────────
        response = await _call_groq_with_budget(
            session_id=sid,
            messages=current_messages,
            max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
        )

        tool_calls = extract_tool_calls(response)
        text_reply = extract_text(response)

        # ── No tool calls → model gave final text answer ───────────────────
        if not tool_calls:
            final_reply = text_reply or "I've completed the requested operations."
            _append_message(sid, {"role": "assistant", "content": final_reply})
            logger.info(
                "process_chat | session=%s | final_reply_len=%d | rounds_used=%d",
                sid, len(final_reply), round_num + 1,
            )
            break

        # ── Echo assistant message (with tool_calls) into history ──────────
        assistant_msg = build_assistant_tool_call_message(response)
        _append_message(sid, assistant_msg)

        logger.debug(
            "process_chat | session=%s | round=%d | tool_calls=%s",
            sid, round_num + 1,
            [tc["name"] for tc in tool_calls],
        )

        # ── Dispatch each tool call ────────────────────────────────────────
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

                # ── CRITICAL: store UNCOMPRESSED result for the UI ─────────
                # The UI receives the full data via AIChatResponse.data.
                last_structured_data = serialised

                # ── CRITICAL: store COMPRESSED result in session history ────
                # This is the main fix for 413 errors from backtest/sentiment
                # payloads.  The model gets key stats; chart arrays are stripped.
                compressed = compress_tool_result(tool_name, serialised)
                compressed_json = json.dumps(compressed, default=str)

                original_json = json.dumps(serialised, default=str)
                if len(compressed_json) < len(original_json):
                    savings_pct = 100 * (1 - len(compressed_json) / len(original_json))
                    logger.info(
                        "process_chat | compressed tool result | tool=%s | "
                        "original=%d chars | compressed=%d chars | saved=%.0f%%",
                        tool_name, len(original_json), len(compressed_json), savings_pct,
                    )

                _append_message(
                    sid,
                    build_tool_result_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=compressed_json,   # CHANGED: was original_json
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
            "I've gathered the requested data. Here's a summary of what was executed: "
            + "; ".join(
                t.result_summary or t.tool_name
                for t in tools_executed
                if t.status == "success"
            )
            + "."
        )
        _append_message(sid, {"role": "assistant", "content": final_reply})
        logger.warning(
            "process_chat | session=%s | MAX_TOOL_ROUNDS (%d) exhausted",
            sid, MAX_TOOL_ROUNDS,
        )

    return AIChatResponse(
        session_id=sid,
        reply=final_reply,
        tools_executed=tools_executed,
        data=last_structured_data,   # FULL uncompressed data for the UI
    )