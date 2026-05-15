"""
TradeX – AI Orchestration Service (Groq / Llama 3.3-70B)
=========================================================

Responsibilities
----------------
1.  Maintain per-session conversation history in-memory (keyed by session_id)
2.  Receive a user prompt and build a full OpenAI-format messages list
3.  Send the prompt to Groq via the OpenAI-compatible SDK
4.  Parse ``tool_calls`` from the response and dispatch to backend services
5.  Inject tool results back as ``role="tool"`` messages and loop
6.  Return AIChatResponse with the final text reply + tool execution trace

Migration notes vs. Gemini implementation
------------------------------------------
| Gemini concept                     | OpenAI/Groq replacement              |
|------------------------------------|--------------------------------------|
| role="model"                       | role="assistant"                     |
| role="user" (function_response)    | role="tool"  (tool_call_id required) |
| parts=[{"function_call": ...}]     | message.tool_calls=[...]             |
| parts=[{"function_response": ...}] | {"role":"tool","tool_call_id":...}   |
| chat.send_message(msg)             | client.chat.completions.create(...)  |
| asyncio.to_thread(chat.send_msg)   | native await (Groq SDK is async)     |
| model.start_chat(history=...)      | full messages[] list each request    |

Key architectural improvements over Gemini version
---------------------------------------------------
- No thread-executor overhead: Groq client is fully async.
- Tool results carry ``tool_call_id`` for strict protocol compliance.
- History format is standard OpenAI messages[] – portable to any compatible LLM.
- ``parallel_tool_calls=True`` enables fan-out in a single round (same as Gemini).
- Structured debug logging for every tool call and LLM round.
- Timeout guard via ``asyncio.wait_for`` on every Groq call.
- Malformed tool-argument recovery: bad JSON is caught, logged, and the tool
  receives an empty args dict rather than crashing the whole request.
- Token budget awareness: history is pruned before it overflows context.
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tradex.ai.service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS: int = 5       # maximum tool-call → result cycles per request
_GROQ_TIMEOUT: float = 60.0    # seconds to wait for each Groq API call
_MAX_HISTORY_MESSAGES: int = 50  # trim window to prevent context overflow

# ---------------------------------------------------------------------------
# In-memory session store
# Maps session_id → list[dict] in OpenAI messages format
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
        # Seed with system prompt so every conversation starts with it
        _SESSION_STORE[sid] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        return sid
    return session_id


def _append_message(session_id: str, message: dict[str, Any]) -> None:
    """
    Append one OpenAI-format message dict to the session history and trim if
    the history exceeds _MAX_HISTORY_MESSAGES.

    The system message at index 0 is always preserved during trimming so the
    model never loses its instructions.
    """
    history = _SESSION_STORE.setdefault(
        session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    history.append(message)
    if len(history) > _MAX_HISTORY_MESSAGES:
        # Keep system message + most recent (_MAX_HISTORY_MESSAGES - 1) turns
        _SESSION_STORE[session_id] = [history[0]] + history[-(
            _MAX_HISTORY_MESSAGES - 1
        ):]


def get_session_messages(session_id: str) -> list[ChatMessage]:
    """
    Return conversation history as ChatMessage objects for GET /ai/history.
    Filters to user and assistant text turns only; skips tool messages.
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
# Tool dispatcher
# Routes OpenAI tool_call names → backend service functions
# ===========================================================================

async def _dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    db: AsyncSession,
) -> Any:
    """
    Route a tool call to the correct backend service function.

    Returns the Pydantic model or primitive from the service.
    Raises HTTPException (propagated to caller) on service-level errors.
    Raises ValueError for unknown tool names.
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
# Result serialisation helpers (unchanged from Gemini version)
# ===========================================================================

def _serialise_result(result: Any) -> dict[str, Any]:
    """
    Convert a service result to a JSON-serialisable dict.
    Pydantic models → .model_dump(); lists → element-wise; primitives wrapped.
    """
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
    3.  POST the full messages list to Groq.
    4.  If the model returns ``tool_calls``:
          a.  Echo the assistant message (with tool_calls) back into history.
          b.  Dispatch each tool call to the correct backend service.
          c.  Append each result as a ``role="tool"`` message.
          d.  Repeat from step 3 (up to MAX_TOOL_ROUNDS times).
    5.  When the model returns a plain text reply, persist history and return.

    Differences from the Gemini orchestration loop
    -----------------------------------------------
    - Messages list is passed whole every round (no start_chat abstraction).
    - Tool results use role="tool" + tool_call_id (not role="user" + parts).
    - The assistant message that contains tool_calls is echoed back verbatim
      (required by the OpenAI protocol so the model tracks its own decisions).
    - Groq call is wrapped in asyncio.wait_for for timeout safety.
    - Debug logging records every round, every tool call, and every result.
    """
    sid = _get_or_create_session(session_id)

    # Append the incoming user message
    _append_message(sid, {"role": "user", "content": user_message})
    logger.info("process_chat | session=%s | user_message_len=%d", sid, len(user_message))

    tools_executed: list[ToolExecution] = []
    last_structured_data: Optional[dict[str, Any]] = None
    final_reply: str = ""

    # ── Orchestration loop ─────────────────────────────────────────────────
    for round_num in range(MAX_TOOL_ROUNDS):
        logger.debug("process_chat | session=%s | round=%d/%d", sid, round_num + 1, MAX_TOOL_ROUNDS)

        # Snapshot current history for this round's API call
        current_messages = list(_SESSION_STORE.get(sid, []))

        # ── Call Groq (with timeout) ───────────────────────────────────────
        try:
            response = await asyncio.wait_for(
                call_groq(current_messages),
                timeout=_GROQ_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("process_chat | session=%s | Groq timeout after %.1fs", sid, _GROQ_TIMEOUT)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Groq API timed out after {_GROQ_TIMEOUT:.0f}s.",
            )
        except Exception as exc:
            logger.error("process_chat | session=%s | Groq error: %s", sid, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Groq API error: {exc}",
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
        # This is mandatory in the OpenAI protocol – the model must see its
        # own tool_calls before it receives the tool results.
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
                last_structured_data = serialised

                # Inject successful tool result into history
                _append_message(
                    sid,
                    build_tool_result_message(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=json.dumps(serialised, default=str),
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
        data=last_structured_data,
    )