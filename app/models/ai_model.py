"""
TradeX – Gemini AI Model

Responsibilities
----------------
- Initialise the Gemini 2.5 Flash client from GEMINI_API_KEY env var
- Declare every TradeX backend function as a Gemini tool (function calling)
- Provide a single `call_gemini()` coroutine that sends a conversation turn
  to the model and returns the raw response object
- Provide `extract_function_calls()` to parse tool-use parts from the response

Environment
-----------
GEMINI_API_KEY   – required; Gemini API key from Google AI Studio
GEMINI_MODEL     – optional; defaults to "gemini-2.5-flash-preview-04-17"
"""

from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not _API_KEY:
    import warnings
    warnings.warn(
        "[ai_model] GEMINI_API_KEY is not set – AI endpoints will fail at runtime.",
        RuntimeWarning,
        stacklevel=1,
    )

genai.configure(api_key=_API_KEY)

_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ===========================================================================
# Tool declarations
# Every function here maps 1-to-1 to a function in ai_service.py.
# Gemini uses these JSON schemas to decide which tool to call and with
# what arguments. Keep descriptions precise – they guide extraction accuracy.
# ===========================================================================

_TRADEX_TOOLS = [
    # ── Strategy catalogue ─────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_strategies",
                description=(
                    "List trading strategies stored in the database. "
                    "Supports optional filtering by symbol (e.g. 'btc', 'eth'), "
                    "time_horizon (e.g. '1h', '15m', '5m'), and a name search string. "
                    "Use this when the user asks about available strategies, "
                    "wants to browse strategies, or asks to find / select a strategy."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "symbol": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Coin symbol filter, e.g. 'btc'. Optional.",
                        ),
                        "time_horizon": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Timeframe filter: '1h' | '15m' | '5m'. Optional.",
                        ),
                        "search": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Partial match on strategy name. Optional.",
                        ),
                        "page": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Page number (1-based). Default 1.",
                        ),
                        "page_size": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Results per page. Default 20, max 100.",
                        ),
                    },
                    required=[],
                ),
            ),
        ]
    ),

    # ── Strategy detail ────────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_strategy_detail",
                description=(
                    "Fetch full detail for a single strategy by its exact name "
                    "(primary key), e.g. 'sig_1h_btc_1'. Returns active indicators, "
                    "candlestick patterns, period parameters, TP/SL, and latest run stats."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "strategy_name": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exact strategy identifier, e.g. 'sig_1h_btc_1'.",
                        ),
                    },
                    required=["strategy_name"],
                ),
            ),
        ]
    ),

    # ── Backtest strategies dropdown ───────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_backtest_strategies",
                description=(
                    "Return all strategies available for backtesting, with their "
                    "symbol, timeframe, default TP/SL, and last run stats. "
                    "Use this to help the user choose the best strategy before running "
                    "a backtest. Call it before 'run_backtest' when the user says "
                    "'best strategy', 'top strategy', or has not specified an exact name."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={},
                    required=[],
                ),
            ),
        ]
    ),

    # ── Run backtest ───────────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="run_backtest",
                description=(
                    "Execute the backtest engine for a specific strategy and exchange. "
                    "Returns a full ledger of trades, win/loss chart data, PnL-per-trade "
                    "data, and summary stats (win rate, final balance, max drawdown streaks). "
                    "Required: strategy_name, exchange. "
                    "Optional: start_date, end_date (ISO format), starting_balance, "
                    "take_profit (%), stop_loss (%), fee, leverage, slippage."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "strategy_name": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exact strategy identifier, e.g. 'sig_1h_btc_1'.",
                        ),
                        "exchange": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'.",
                        ),
                        "start_date": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Start datetime (inclusive), e.g. '2024-01-01'. Optional.",
                        ),
                        "end_date": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="End datetime (inclusive), e.g. '2024-12-31'. Optional.",
                        ),
                        "starting_balance": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Starting capital in USD. Default 1000.",
                        ),
                        "take_profit": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Take-profit as a percentage, e.g. 1.5 means 1.5%. Default 1.0.",
                        ),
                        "stop_loss": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Stop-loss as a percentage, e.g. 1.0 means 1%. Default 1.0.",
                        ),
                        "buy_after_minutes": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Delay buying N minutes after signal. Default 0.",
                        ),
                        "fee": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Trading fee percentage. Default 0.05.",
                        ),
                        "leverage": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Leverage multiplier. Default 1.0.",
                        ),
                        "slippage": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER,
                            description="Slippage percentage. Default 0.0.",
                        ),
                    },
                    required=["strategy_name", "exchange"],
                ),
            ),
        ]
    ),

    # ── List saved backtest runs ───────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_strategy_runs",
                description=(
                    "List all previously saved backtest runs for a strategy, "
                    "ordered newest first. Each item contains exchange, date range, "
                    "TP/SL, win-rate, and PnL. Use when the user asks about past runs "
                    "or run history for a strategy."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "strategy_name": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exact strategy identifier.",
                        ),
                    },
                    required=["strategy_name"],
                ),
            ),
        ]
    ),

    # ── ML / DL models ────────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_models",
                description=(
                    "List machine-learning (ml) or deep-learning (dl) model backtest "
                    "results stored in the database. Supports search by model name and "
                    "pagination. Use when the user asks about ML/DL models, model "
                    "performance, or wants to compare models."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "model_type": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="'ml' for machine-learning or 'dl' for deep-learning.",
                        ),
                        "search": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Partial match on model name. Optional.",
                        ),
                        "page": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Page number. Default 1.",
                        ),
                        "page_size": genai.protos.Schema(
                            type=genai.protos.Type.INTEGER,
                            description="Items per page. Default 20.",
                        ),
                    },
                    required=["model_type"],
                ),
            ),
        ]
    ),

    # ── ML / DL model detail ──────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_model_detail",
                description=(
                    "Return full metrics for a single ML or DL model run: "
                    "PnL, trade counts, win/loss rates, risk metrics (Sharpe, Sortino, "
                    "max drawdown), and streak data. Use when the user asks for details "
                    "about a specific model by name."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "model_type": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="'ml' or 'dl'.",
                        ),
                        "model_name": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exact model run identifier.",
                        ),
                    },
                    required=["model_type", "model_name"],
                ),
            ),
        ]
    ),

    # ── Sentiment results ─────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_sentiment_results",
                description=(
                    "Retrieve cached Reddit sentiment analysis results for a coin "
                    "from the database without re-running the pipeline. Returns "
                    "per-post sentiment, hourly aggregated sentiment chart data, "
                    "and overall stats. Use when the user asks about market sentiment "
                    "or social-media mood for a coin."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "coin": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Coin id: 'btc' | 'eth' | 'sol'.",
                        ),
                    },
                    required=["coin"],
                ),
            ),
        ]
    ),

    # ── Run sentiment pipeline ────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="run_sentiment",
                description=(
                    "Scrape Reddit for the latest posts and comments, run FinBERT "
                    "sentiment analysis for the specified coin, persist the results, "
                    "and return them. This is a long-running operation. "
                    "Use only when the user explicitly asks to run or refresh the "
                    "sentiment pipeline, NOT just to view existing results."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "coin": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Coin id: 'btc' | 'eth' | 'sol'.",
                        ),
                    },
                    required=["coin"],
                ),
            ),
        ]
    ),

    # ── OHLCV data ────────────────────────────────────────────────────────
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="get_ohlcv",
                description=(
                    "Retrieve saved OHLCV (candlestick) price data for a coin "
                    "from a specific exchange. Returns candle stats: open, high, low, "
                    "close, volume, and total rows. Use when the user asks about price "
                    "data, market data, or wants to know OHLCV stats for a coin."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "exchange": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'.",
                        ),
                        "symbol": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Coin symbol key, e.g. 'btc'.",
                        ),
                        "start_date": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Filter start date, e.g. '2024-01-01'. Optional.",
                        ),
                        "end_date": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="Filter end date, e.g. '2024-12-31'. Optional.",
                        ),
                    },
                    required=["exchange", "symbol"],
                ),
            ),
        ]
    ),
]


# ===========================================================================
# System prompt – sets the AI's role and operational constraints
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

def get_gemini_model() -> genai.GenerativeModel:
    """
    Return a configured GenerativeModel instance with all TradeX tools attached.
    Called once per request (stateless).
    """
    return genai.GenerativeModel(
        model_name=_MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        tools=_TRADEX_TOOLS,
    )


async def call_gemini(
    model: genai.GenerativeModel,
    history: list[dict[str, Any]],
    user_message: str,
) -> GenerateContentResponse:
    """
    Send a conversation turn to Gemini and return the raw response.

    Parameters
    ----------
    model        : GenerativeModel from get_gemini_model()
    history      : Previous conversation turns in Gemini format
                   [{"role": "user"|"model", "parts": [{"text": "..."}]}]
    user_message : The new user turn to send

    Returns
    -------
    GenerateContentResponse – caller inspects .candidates[0].content.parts
    """
    chat = model.start_chat(history=history)
    # Use asyncio executor to avoid blocking the event loop on the sync SDK call
    import asyncio
    response: GenerateContentResponse = await asyncio.to_thread(
        chat.send_message, user_message
    )
    return response


def extract_function_calls(
    response: GenerateContentResponse,
) -> list[dict[str, Any]]:
    """
    Extract all function_call parts from a Gemini response.

    Returns a list of dicts:  [{"name": str, "args": dict}, ...]
    Returns an empty list when the model replied with pure text.
    """
    calls: list[dict[str, Any]] = []
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                calls.append({
                    "name": fc.name,
                    "args": dict(fc.args),
                })
    except (IndexError, AttributeError):
        pass
    return calls


def extract_text(response: GenerateContentResponse) -> str:
    """
    Extract the concatenated text from a Gemini response.
    Returns an empty string when the response contains only tool calls.
    """
    parts: list[str] = []
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
    except (IndexError, AttributeError):
        pass
    return "".join(parts).strip()