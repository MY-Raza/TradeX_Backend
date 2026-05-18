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

TOKEN OPTIMISATION CHANGES (vs original)
-----------------------------------------
1. SYSTEM_PROMPT reduced from ~250 tokens → ~110 tokens.
   The model already knows how to use tools; verbose instructions waste budget.
2. ``max_tokens`` default lowered from 4 096 → 1 024.
   Groq counts *requested* output tokens against the TPM limit even when the
   actual reply is short.  1 024 is sufficient for trading Q&A; caller can
   override for complex summaries.
3. ``call_groq()`` now accepts a ``max_tokens`` override so ai_service can
   pass lower values for simple one-shot queries.
4. Tool descriptions tightened: removed redundant filler sentences that the
   model doesn't need to select the right tool.
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
# TOKEN OPTIMISATION: description strings have been tightened to remove
# repetitive preamble.  Each description is now a single focused sentence
# that tells the model WHEN to call the tool.  This saves ~200-300 tokens
# that were wasted on prose the model never uses for routing decisions.
# ===========================================================================

TRADEX_TOOLS: list[dict[str, Any]] = [

    # ── Strategy catalogue ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_strategies",
            "description": (
                "List/filter trading strategies by symbol (e.g. 'btc'), "
                "time_horizon ('1h','15m','5m'), or name search string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Coin filter, e.g. 'btc'."},
                    "time_horizon": {"type": "string", "description": "Timeframe: '1h'|'15m'|'5m'."},
                    "search": {"type": "string", "description": "Partial name match."},
                    "page": {"type": "integer", "description": "Page (1-based). Default 1."},
                    "page_size": {"type": "integer", "description": "Per page. Default 20."},
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
                "Fetch full detail (indicators, patterns, TP/SL, last run stats) "
                "for one strategy by exact name, e.g. 'sig_1h_btc_1'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {"type": "string", "description": "Exact strategy id."},
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
                "Return all strategies available for backtesting with symbol, "
                "timeframe, default TP/SL, and last run stats. "
                "Call before run_backtest when user says 'best strategy'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ── Run backtest ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": (
                "Execute backtest for a strategy+exchange. "
                "Required: strategy_name, exchange. "
                "Optional: start_date, end_date (ISO), starting_balance, "
                "take_profit(%), stop_loss(%), fee, leverage, slippage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {"type": "string", "description": "Exact strategy id."},
                    "exchange": {"type": "string", "description": "'binance'|'bybit'|'kraken'|'metatrader5'."},
                    "start_date": {"type": "string", "description": "ISO date, e.g. '2024-01-01'."},
                    "end_date": {"type": "string", "description": "ISO date, e.g. '2024-12-31'."},
                    "starting_balance": {"type": "number", "description": "USD. Default 1000."},
                    "take_profit": {"type": "number", "description": "% e.g. 1.5. Default 1.0."},
                    "stop_loss": {"type": "number", "description": "% e.g. 1.0. Default 1.0."},
                    "buy_after_minutes": {"type": "integer", "description": "Signal delay. Default 0."},
                    "fee": {"type": "number", "description": "Fee %. Default 0.05."},
                    "leverage": {"type": "number", "description": "Multiplier. Default 1.0."},
                    "slippage": {"type": "number", "description": "Slippage %. Default 0.0."},
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
            "description": "List previously saved backtest runs for a strategy, newest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy_name": {"type": "string", "description": "Exact strategy id."},
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
            "description": "List ML or DL model backtest results; supports search and pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {"type": "string", "description": "'ml' or 'dl'."},
                    "search": {"type": "string", "description": "Partial model name match."},
                    "page": {"type": "integer", "description": "Page. Default 1."},
                    "page_size": {"type": "integer", "description": "Per page. Default 20."},
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
                "Full metrics for one ML/DL model: PnL, win rate, Sharpe, "
                "Sortino, max drawdown, streak data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {"type": "string", "description": "'ml' or 'dl'."},
                    "model_name": {"type": "string", "description": "Exact model run id."},
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
                "Retrieve cached Reddit/FinBERT sentiment for a coin "
                "('btc'|'eth'|'sol') without re-running the pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "'btc'|'eth'|'sol'."},
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
                "Scrape Reddit + run FinBERT sentiment for a coin. "
                "Long-running – call only when user explicitly asks to refresh."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "'btc'|'eth'|'sol'."},
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
            "description": "Retrieve OHLCV candlestick data for a coin+exchange with optional date filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "exchange": {"type": "string", "description": "'binance'|'bybit'|'kraken'|'metatrader5'."},
                    "symbol": {"type": "string", "description": "Coin key, e.g. 'btc'."},
                    "start_date": {"type": "string", "description": "Filter start, e.g. '2024-01-01'."},
                    "end_date": {"type": "string", "description": "Filter end, e.g. '2024-12-31'."},
                },
                "required": ["exchange", "symbol"],
            },
        },
    },
]


# ===========================================================================
# System prompt
#
# TOKEN OPTIMISATION: Reduced from ~250 tokens to ~110 tokens.
#
# WHY: The system prompt is sent with EVERY request.  In a 5-round tool loop
# the original 250-token prompt consumed 1 250 tokens just for instructions.
# The model already understands tool calling; it only needs:
#   (a) its persona / domain context
#   (b) the non-obvious routing rules (best strategy → get_backtest_strategies first)
#   (c) output format constraints
# Everything else ("You have access to the following tools: …") is redundant
# because the tool declarations themselves convey that information.
# ===========================================================================

SYSTEM_PROMPT: str = (
    "You are TradeX AI, an assistant for an algorithmic trading platform. "
    "Rules:\n"
    "1. Always call tools for live data before answering quantitative questions.\n"
    "2. For 'best strategy': call get_backtest_strategies first, rank by last_pnl_pct, "
    "then run_backtest for the winner.\n"
    "3. Parse natural-language dates to ISO YYYY-MM-DD.\n"
    "4. Default exchange: binance.\n"
    "5. Be concise: lead with numbers, then explain.\n"
    "6. Never fabricate data. Report tool errors clearly.\n"
    "7. Format numbers: 2 decimal places, % on rates."
)


# ===========================================================================
# Public API
# ===========================================================================

def get_groq_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client configured for Groq."""
    return client


async def call_groq(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,   # CHANGED: was 4096 – see module docstring
) -> ChatCompletion:
    """
    Send a full OpenAI-format messages list to Groq and return the raw response.

    Parameters
    ----------
    messages    : Full conversation history (system + all turns + latest user).
    temperature : 0.1 = near-deterministic, good for trading data.
    max_tokens  : Output cap.  Default 1 024 (was 4 096).
                  Groq charges requested output tokens against the TPM limit,
                  so keeping this low is important for free-tier accounts.
                  Pass 2 048 for summaries, 512 for simple yes/no answers.

    Returns
    -------
    ChatCompletion
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
                args = parsed if isinstance(parsed, dict) else {}
            except (_json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "extract_tool_calls | malformed args for tool '%s': %r",
                    tc.function.name, raw_args,
                )

        calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "args": args,
        })

    return calls


def extract_text(response: ChatCompletion) -> str:
    """Extract text content from a ChatCompletion. Returns '' for tool-only responses."""
    content = response.choices[0].message.content
    return (content or "").strip()


def build_assistant_tool_call_message(
    response: ChatCompletion,
) -> dict[str, Any]:
    """
    Build the assistant message dict to append to conversation history
    after the model requests tool calls.
    """
    msg = response.choices[0].message
    return {
        "role": "assistant",
        "content": msg.content,
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

    Parameters
    ----------
    tool_call_id : The ``id`` from the original tool_call request.
    tool_name    : Function name (for debugging).
    content      : JSON-serialised string of the tool result.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": content,
    }