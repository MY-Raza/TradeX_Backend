"""
TradeX – Groq / Llama 3.3-70B AI Model Layer
=============================================

Responsibilities
----------------
- Initialise the Groq client via the OpenAI-compatible SDK
- Declare every TradeX backend function as an OpenAI-format tool (JSON Schema)
- Provide ``call_groq()`` – fully async, no thread-executor needed
- Provide ``extract_tool_calls()`` and ``extract_text()`` to parse responses
- Expose ``SYSTEM_PROMPT`` and ``TRADEX_TOOLS`` for use in ai_service.py

Environment variables
---------------------
GROQ_API_KEY   – required; obtain from https://console.groq.com
GROQ_MODEL     – optional; defaults to "llama-3.3-70b-versatile"

Architecture notes
------------------
- The Groq API is 100% OpenAI-compatible, so we use the ``openai`` SDK with a
  custom ``base_url``.  No Groq-specific SDK is needed.
- Tool declarations use plain Python dicts (OpenAI JSON-Schema format) instead
  of ``genai.protos.*`` objects, making them easier to read, test, and extend.
- ``call_groq()`` is a true coroutine (uses ``await client.chat.completions.create``)
  so it never blocks the FastAPI event loop – unlike the previous Gemini path
  that required ``asyncio.to_thread``.
- We set ``tool_choice="auto"`` so the model decides whether to call a tool or
  respond directly, matching Gemini's automatic function-calling behaviour.
- ``parallel_tool_calls=True`` allows Llama 3.3 to fan out multiple tool calls
  in one round (same as Gemini's multi-part function_call responses).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tradex.ai.model")

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not _API_KEY:
    import warnings
    warnings.warn(
        "[ai_model] GROQ_API_KEY is not set – AI endpoints will fail at runtime.",
        RuntimeWarning,
        stacklevel=1,
    )

# AsyncOpenAI pointed at Groq's OpenAI-compatible endpoint
client = AsyncOpenAI(
    api_key=_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

_MODEL_NAME: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ===========================================================================
# Tool declarations (OpenAI JSON-Schema format)
#
# Each entry is a dict that matches the ``tools`` parameter accepted by
# ``client.chat.completions.create()``.  The structure is:
#
#   {
#       "type": "function",
#       "function": {
#           "name":        str,
#           "description": str,
#           "parameters": {          # standard JSON Schema object
#               "type": "object",
#               "properties": { ... },
#               "required": [ ... ],
#           },
#       },
#   }
#
# These replace all ``genai.protos.Tool`` / ``genai.protos.FunctionDeclaration``
# / ``genai.protos.Schema`` objects from the old Gemini implementation.
# ===========================================================================

TRADEX_TOOLS: list[dict[str, Any]] = [

    # ── Strategy catalogue ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_strategies",
            "description": (
                "List trading strategies stored in the database. "
                "Supports optional filtering by symbol (e.g. 'btc', 'eth'), "
                "time_horizon (e.g. '1h', '15m', '5m'), and a name search string. "
                "Use this when the user asks about available strategies, "
                "wants to browse strategies, or asks to find / select a strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Coin symbol filter, e.g. 'btc'. Optional.",
                    },
                    "time_horizon": {
                        "type": "string",
                        "description": "Timeframe filter: '1h' | '15m' | '5m'. Optional.",
                    },
                    "search": {
                        "type": "string",
                        "description": "Partial match on strategy name. Optional.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-based). Default 1.",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Results per page. Default 20, max 100.",
                    },
                },
                "required": [],
            },
        },
    },

    # ── Strategy detail ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_strategy_detail",
            "description": (
                "Fetch full detail for a single strategy by its exact name "
                "(primary key), e.g. 'sig_1h_btc_1'. Returns active indicators, "
                "candlestick patterns, period parameters, TP/SL, and latest run stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Exact strategy identifier, e.g. 'sig_1h_btc_1'.",
                    },
                },
                "required": ["strategy_name"],
            },
        },
    },

    # ── Backtest strategies dropdown ───────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_backtest_strategies",
            "description": (
                "Return all strategies available for backtesting, with their "
                "symbol, timeframe, default TP/SL, and last run stats. "
                "Use this to help the user choose the best strategy before running "
                "a backtest. Call it before 'run_backtest' when the user says "
                "'best strategy', 'top strategy', or has not specified an exact name."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    # ── Run backtest ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "Execute the backtest engine for a specific strategy and exchange. "
                "Returns a full ledger of trades, win/loss chart data, PnL-per-trade "
                "data, and summary stats (win rate, final balance, max drawdown streaks). "
                "Required: strategy_name, exchange. "
                "Optional: start_date, end_date (ISO format), starting_balance, "
                "take_profit (%), stop_loss (%), fee, leverage, slippage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Exact strategy identifier, e.g. 'sig_1h_btc_1'.",
                    },
                    "exchange": {
                        "type": "string",
                        "description": "Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start datetime (inclusive), e.g. '2024-01-01'. Optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End datetime (inclusive), e.g. '2024-12-31'. Optional.",
                    },
                    "starting_balance": {
                        "type": "number",
                        "description": "Starting capital in USD. Default 1000.",
                    },
                    "take_profit": {
                        "type": "number",
                        "description": "Take-profit as a percentage, e.g. 1.5 means 1.5%. Default 1.0.",
                    },
                    "stop_loss": {
                        "type": "number",
                        "description": "Stop-loss as a percentage, e.g. 1.0 means 1%. Default 1.0.",
                    },
                    "buy_after_minutes": {
                        "type": "integer",
                        "description": "Delay buying N minutes after signal. Default 0.",
                    },
                    "fee": {
                        "type": "number",
                        "description": "Trading fee percentage. Default 0.05.",
                    },
                    "leverage": {
                        "type": "number",
                        "description": "Leverage multiplier. Default 1.0.",
                    },
                    "slippage": {
                        "type": "number",
                        "description": "Slippage percentage. Default 0.0.",
                    },
                },
                "required": ["strategy_name", "exchange"],
            },
        },
    },

    # ── Strategy run history ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_strategy_runs",
            "description": (
                "List all previously saved backtest runs for a strategy, "
                "ordered newest first. Each item contains exchange, date range, "
                "TP/SL, win-rate, and PnL. Use when the user asks about past runs "
                "or run history for a strategy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {
                        "type": "string",
                        "description": "Exact strategy identifier.",
                    },
                },
                "required": ["strategy_name"],
            },
        },
    },

    # ── ML / DL model list ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_models",
            "description": (
                "List machine-learning (ml) or deep-learning (dl) model backtest "
                "results stored in the database. Supports search by model name and "
                "pagination. Use when the user asks about ML/DL models, model "
                "performance, or wants to compare models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "'ml' for machine-learning or 'dl' for deep-learning.",
                    },
                    "search": {
                        "type": "string",
                        "description": "Partial match on model name. Optional.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number. Default 1.",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Items per page. Default 20.",
                    },
                },
                "required": ["model_type"],
            },
        },
    },

    # ── ML / DL model detail ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_model_detail",
            "description": (
                "Return full metrics for a single ML or DL model run: "
                "PnL, trade counts, win/loss rates, risk metrics (Sharpe, Sortino, "
                "max drawdown), and streak data. Use when the user asks for details "
                "about a specific model by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "'ml' or 'dl'.",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Exact model run identifier.",
                    },
                },
                "required": ["model_type", "model_name"],
            },
        },
    },

    # ── Sentiment results ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_sentiment_results",
            "description": (
                "Retrieve cached Reddit sentiment analysis results for a coin "
                "from the database without re-running the pipeline. Returns "
                "per-post sentiment, hourly aggregated sentiment chart data, "
                "and overall stats. Use when the user asks about market sentiment "
                "or social-media mood for a coin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {
                        "type": "string",
                        "description": "Coin id: 'btc' | 'eth' | 'sol'.",
                    },
                },
                "required": ["coin"],
            },
        },
    },

    # ── Run sentiment pipeline ────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_sentiment",
            "description": (
                "Scrape Reddit for the latest posts and comments, run FinBERT "
                "sentiment analysis for the specified coin, persist the results, "
                "and return them. This is a long-running operation. "
                "Use only when the user explicitly asks to run or refresh the "
                "sentiment pipeline, NOT just to view existing results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {
                        "type": "string",
                        "description": "Coin id: 'btc' | 'eth' | 'sol'.",
                    },
                },
                "required": ["coin"],
            },
        },
    },

    # ── OHLCV data ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_ohlcv",
            "description": (
                "Retrieve saved OHLCV (candlestick) price data for a coin "
                "from a specific exchange. Returns candle stats: open, high, low, "
                "close, volume, and total rows. Use when the user asks about price "
                "data, market data, or wants to know OHLCV stats for a coin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {
                        "type": "string",
                        "description": "Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'.",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Coin symbol key, e.g. 'btc'.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Filter start date, e.g. '2024-01-01'. Optional.",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Filter end date, e.g. '2024-12-31'. Optional.",
                    },
                },
                "required": ["exchange", "symbol"],
            },
        },
    },
]


# ===========================================================================
# System prompt
# ===========================================================================

SYSTEM_PROMPT: str = """You are TradeX AI, an intelligent assistant integrated into the TradeX algorithmic trading platform.

You have access to the following real-time backend tools:
- get_strategies          – browse and filter trading strategies
- get_strategy_detail     – inspect a specific strategy's indicators and parameters
- get_backtest_strategies – list all strategies available for backtesting
- run_backtest            – execute the backtest engine for a strategy
- get_strategy_runs       – view historical backtest runs for a strategy
- get_models              – list ML/DL model performance results
- get_model_detail        – inspect a specific model's full metrics
- get_sentiment_results   – retrieve cached Reddit/FinBERT sentiment for a coin
- run_sentiment           – trigger a fresh Reddit scrape + FinBERT sentiment run
- get_ohlcv               – retrieve OHLCV price data for a coin and exchange

IMPORTANT RULES:
1. Always call tools to retrieve live data before answering quantitative questions.
2. When the user asks for the "best" strategy, call get_backtest_strategies first,
   rank by last_pnl_pct (highest), then call run_backtest for the winner.
3. Parse dates mentioned in natural language (e.g. "11/11/2025") to ISO format "YYYY-MM-DD".
4. Default exchange is "binance" when not specified.
5. When executing multi-step operations (select best strategy → run backtest),
   execute them in order and report results from each step.
6. Be concise and data-driven. Lead with numbers, then explain.
7. Never fabricate data. If a tool fails, report the error clearly.
8. Format financial numbers with 2 decimal places and include % signs on rates.
"""


# ===========================================================================
# Public API
# ===========================================================================

def get_groq_client() -> AsyncOpenAI:
    """
    Return the shared AsyncOpenAI client configured for Groq.

    The client is module-level singleton; this function exists so callers
    have a stable import surface and tests can monkeypatch it easily.
    """
    return client


async def call_groq(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> ChatCompletion:
    """
    Send a full OpenAI-format messages list to Groq and return the raw response.

    Parameters
    ----------
    messages    : Full conversation history in OpenAI format, including the
                  system message, all prior user/assistant/tool turns, and the
                  latest user turn.  The caller (ai_service) builds this list.
    temperature : Sampling temperature.  0.1 gives near-deterministic, factual
                  responses suitable for trading data queries.
    max_tokens  : Upper bound on response length.

    Returns
    -------
    ChatCompletion – caller inspects .choices[0].message for tool_calls / content.

    Notes
    -----
    This is a true async coroutine – no ``asyncio.to_thread`` wrapper needed,
    unlike the previous Gemini implementation which used a sync SDK.
    ``parallel_tool_calls=True`` allows Llama 3.3 to fan out multiple tool
    invocations in a single response, preserving Gemini's multi-part behaviour.
    """
    logger.debug(
        "call_groq | model=%s | messages=%d | max_tokens=%d",
        _MODEL_NAME, len(messages), max_tokens,
    )

    response: ChatCompletion = await client.chat.completions.create(
        model=_MODEL_NAME,
        messages=messages,           # type: ignore[arg-type]
        tools=TRADEX_TOOLS,          # type: ignore[arg-type]
        tool_choice="auto",
        parallel_tool_calls=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.debug(
        "call_groq | finish_reason=%s | tool_calls=%d",
        response.choices[0].finish_reason,
        len(response.choices[0].message.tool_calls or []),
    )
    return response


def extract_tool_calls(response: ChatCompletion) -> list[dict[str, Any]]:
    """
    Extract all tool calls from a Groq/OpenAI ChatCompletion response.

    Returns a list of dicts:
        [{"id": str, "name": str, "args": dict}, ...]

    The ``id`` field is the tool_call_id required when submitting tool results
    back to the model.  This replaces ``extract_function_calls()`` from the
    Gemini implementation (which had no concept of tool_call_id).

    Returns an empty list when the model replied with pure text.
    """
    import json as _json

    calls: list[dict[str, Any]] = []
    message = response.choices[0].message
    if not message.tool_calls:
        return calls

    for tc in message.tool_calls:
        raw_args = getattr(tc.function, "arguments", None)
        args: dict[str, Any] = {}
        if raw_args:
            try:
                parsed = _json.loads(raw_args)
                # Llama 3.3 occasionally returns a JSON null or non-dict
                args = parsed if isinstance(parsed, dict) else {}
            except (_json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "extract_tool_calls | malformed args for tool '%s': %r",
                    tc.function.name, raw_args,
                )

        calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "args": args,   # always a dict, never None
        })

    return calls


def extract_text(response: ChatCompletion) -> str:
    """
    Extract the text content from a Groq/OpenAI ChatCompletion response.
    Returns an empty string when the response contains only tool calls.
    """
    content = response.choices[0].message.content
    return (content or "").strip()


def build_assistant_tool_call_message(
    response: ChatCompletion,
) -> dict[str, Any]:
    """
    Build the assistant message dict to append to conversation history
    after the model requests tool calls.

    In the OpenAI protocol the assistant message that contains tool_calls
    must be echoed back verbatim in the next request so the model can see
    its own prior decisions.  This replaces the Gemini pattern of appending
    a ``{"role": "model", "parts": [{"function_call": ...}]}`` turn.
    """
    msg = response.choices[0].message
    return {
        "role": "assistant",
        "content": msg.content,          # may be None – that's fine
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    }


def build_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    content: str,
) -> dict[str, Any]:
    """
    Build a ``tool`` role message to inject a tool result into the conversation.

    In the OpenAI protocol tool results use role="tool" (not "user") and must
    reference the tool_call_id from the assistant's tool_call request.
    This replaces the Gemini ``function_response`` part injected under the
    "user" role.

    Parameters
    ----------
    tool_call_id : The ``id`` field from the original tool_call request.
    tool_name    : The function name (for clarity / debugging).
    content      : JSON-serialised string of the tool result.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": content,
    }