"""
TradeX – AI Orchestration Service

Responsibilities
----------------
1.  Maintain per-session conversation history (in-memory, keyed by session_id)
2.  Receive a user prompt and build the Gemini conversation context
3.  Send the prompt to Gemini; parse any function_call parts in the response
4.  Dispatch function calls to the EXISTING backend service functions
5.  Feed tool results back to Gemini for a final natural-language reply
6.  Return AIChatResponse with structured data + the tool execution trace

Design notes
------------
- All calls to existing services go through the same async functions already
  used by the FastAPI routes – no duplication, no re-implementation.
- History is stored as a list of Gemini-format dicts so the same list can be
  passed directly to model.start_chat(history=...).
- Tool result payloads are serialised to JSON-compatible dicts before being
  sent back to Gemini as function_response parts.
- The orchestration loop runs at most MAX_TOOL_ROUNDS rounds to prevent
  infinite loops when the model keeps requesting tools.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import google.generativeai as genai
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.ai_model import (
    call_gemini,
    extract_function_calls,
    extract_text,
    get_gemini_model,
)
from app.ai.ai_schema import (
    AIChatResponse,
    ChatMessage,
    ToolExecution,
)

# Existing service imports – called directly, no wrappers needed
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
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS: int = 5  # maximum tool-call → result cycles per request

# ---------------------------------------------------------------------------
# In-memory session store
# Maps session_id → list of Gemini-format history dicts
# Replace with Redis/DB persistence for production multi-instance deployments.
# ---------------------------------------------------------------------------

_SESSION_STORE: dict[str, list[dict[str, Any]]] = {}

# Maximum messages kept per session (prevents unbounded growth)
_MAX_HISTORY_MESSAGES: int = 50


# ===========================================================================
# Session helpers
# ===========================================================================

def _get_or_create_session(session_id: Optional[str]) -> str:
    """Return an existing session id or create a new one."""
    if not session_id or session_id not in _SESSION_STORE:
        sid = session_id or str(uuid.uuid4())
        _SESSION_STORE[sid] = []
        return sid
    return session_id


def _append_to_history(
    session_id: str,
    role: str,           # "user" | "model"
    parts: list[Any],
) -> None:
    history = _SESSION_STORE.setdefault(session_id, [])
    history.append({"role": role, "parts": parts})
    # Trim to keep only the most recent messages
    if len(history) > _MAX_HISTORY_MESSAGES:
        _SESSION_STORE[session_id] = history[-_MAX_HISTORY_MESSAGES:]


def get_session_messages(session_id: str) -> list[ChatMessage]:
    """
    Return conversation history as ChatMessage objects for GET /ai/history.
    Filters to text-only parts; skips function_call / function_response parts.
    """
    history = _SESSION_STORE.get(session_id, [])
    messages: list[ChatMessage] = []
    for turn in history:
        role = turn.get("role", "user")
        for part in turn.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                messages.append(
                    ChatMessage(
                        role="user" if role == "user" else "assistant",
                        content=part["text"],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )
            elif isinstance(part, str):
                messages.append(
                    ChatMessage(
                        role="user" if role == "user" else "assistant",
                        content=part,
                        timestamp=datetime.now(timezone.utc).isoformat(),
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
# Maps Gemini function_call names → actual backend service calls
# ===========================================================================

async def _dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    db: AsyncSession,
) -> Any:
    """
    Route a Gemini function_call to the correct backend service function.
    Returns the Pydantic model or primitive from the service.
    Raises HTTPException (propagated) or returns an error dict on service error.
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
        result = await strategy_service.get_strategy_by_name(db, args["strategy_name"])
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{args['strategy_name']}' not found.",
            )
        return result

    # ── Backtest strategies dropdown ───────────────────────────────────────
    if tool_name == "get_backtest_strategies":
        return await backtest_service.get_backtest_strategies(db)

    # ── Run backtest ───────────────────────────────────────────────────────
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

    # ── Strategy run history ───────────────────────────────────────────────
    if tool_name == "get_strategy_runs":
        return await backtest_service.get_strategy_runs(db, args["strategy_name"])

    # ── Models ────────────────────────────────────────────────────────────
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

    # ── Sentiment ─────────────────────────────────────────────────────────
    if tool_name == "get_sentiment_results":
        return await sentiment_service.get_sentiment_results(db, args["coin"])

    if tool_name == "run_sentiment":
        req = SentimentRunRequest(coin=args["coin"])
        return await sentiment_service.run_sentiment(db, req)

    # ── OHLCV ─────────────────────────────────────────────────────────────
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


def _serialise_result(result: Any) -> dict[str, Any]:
    """
    Convert a service result to a JSON-serialisable dict.
    Pydantic models are serialised via .model_dump(); lists are handled
    element-wise; primitives are wrapped in {"value": ...}.
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
    """
    Generate a short human-readable summary of a tool result.
    Used to populate ToolExecution.result_summary for the response envelope.
    """
    try:
        if tool_name == "get_strategies":
            return f"Found {result.total} strategies (page {result.page}/{result.pages})"
        if tool_name == "get_strategy_detail":
            return (
                f"Strategy '{result.name}': {len(result.indicators)} indicators, "
                f"{len(result.patterns)} patterns, last PnL: "
                f"{result.last_pnl_pct:.2f}%" if result.last_pnl_pct is not None
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
        if tool_name in ("get_models",):
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
    Full AI orchestration pipeline.

    1. Retrieve / create session
    2. Send user message to Gemini (with history)
    3. If Gemini returns function_calls → dispatch to backend services
    4. Feed results back to Gemini as function_response → get final text reply
    5. Persist updated history
    6. Return AIChatResponse

    The loop runs up to MAX_TOOL_ROUNDS times to support multi-step workflows
    such as: fetch strategies → pick best → run backtest → summarise.
    """
    sid = _get_or_create_session(session_id)
    model = get_gemini_model()

    # Conversation history for this session (Gemini format)
    history = list(_SESSION_STORE.get(sid, []))

    tools_executed: list[ToolExecution] = []
    last_structured_data: Optional[dict[str, Any]] = None
    final_reply: str = ""

    # ── Append user turn to history before sending ─────────────────────────
    _append_to_history(sid, "user", [{"text": user_message}])
    # Rebuild history reference after append
    history = list(_SESSION_STORE.get(sid, []))
    # The last item is the new user turn; pass everything before it as history
    send_history = history[:-1]

    # ── Orchestration loop ────────────────────────────────────────────────
    current_message: str = user_message

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            response = await call_gemini(model, send_history, current_message)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Gemini API error: {exc}",
            )

        function_calls = extract_function_calls(response)
        text_reply = extract_text(response)

        # No tool calls → model gave a final text answer
        if not function_calls:
            final_reply = text_reply or "I've completed the requested operations."
            # Record the model reply in history
            _append_to_history(sid, "model", [{"text": final_reply}])
            break

        # Process each function call in this round
        tool_results_for_gemini: list[dict[str, Any]] = []

        for fc in function_calls:
            tool_name = fc["name"]
            args = fc.get("args", {})

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

                tool_results_for_gemini.append({
                    "function_response": {
                        "name": tool_name,
                        "response": {"content": json.dumps(serialised, default=str)},
                    }
                })

            except HTTPException as http_exc:
                error_msg = http_exc.detail
                tools_executed.append(
                    ToolExecution(
                        tool_name=tool_name,
                        parameters=args,
                        status="error",
                        error=str(error_msg),
                    )
                )
                tool_results_for_gemini.append({
                    "function_response": {
                        "name": tool_name,
                        "response": {"error": str(error_msg)},
                    }
                })

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
                tool_results_for_gemini.append({
                    "function_response": {
                        "name": tool_name,
                        "response": {"error": error_msg},
                    }
                })

        # Append the model's tool-call turn + our results to history so
        # the next round has full context
        _append_to_history(
            sid,
            "model",
            [{"function_call": fc} for fc in function_calls],
        )
        _append_to_history(
            sid,
            "user",  # function_response parts go in the "user" role in Gemini API
            tool_results_for_gemini,
        )

        # Rebuild history and prepare for next round (model sees its own
        # function_calls + our function_responses as the new "current_message")
        send_history = list(_SESSION_STORE.get(sid, []))[:-1]
        # Feed the function responses as the next user message content
        current_message = json.dumps(
            [r.get("function_response", {}) for r in tool_results_for_gemini],
            default=str,
        )

    else:
        # Reached MAX_TOOL_ROUNDS without a final text reply
        final_reply = (
            "I've gathered the requested data. Here's a summary of what was executed: "
            + "; ".join(t.result_summary or t.tool_name for t in tools_executed if t.status == "success")
            + "."
        )
        _append_to_history(sid, "model", [{"text": final_reply}])

    return AIChatResponse(
        session_id=sid,
        reply=final_reply,
        tools_executed=tools_executed,
        data=last_structured_data,
    )