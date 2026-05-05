"""
TradeX – ORM models for backtest data sources.

Dynamic tables resolved at runtime:
  1. Price data       → <exchange_schema>.<symbol>_1m
  2. Predictions      → strategy_signals.<strategy_name>
  3. Saved ledger run → backtest_runs.<strategy_name>_run_<i>
  4. Run registry     → backtest_runs.run_registry  (one row per completed run)
"""

from __future__ import annotations

from sqlalchemy import (
    Column, DateTime, Float, Integer, SmallInteger,
    String, Text, MetaData, Table, BigInteger, Numeric,
)

# ---------------------------------------------------------------------------
# Known coin keywords for symbol extraction from a strategy name
# ---------------------------------------------------------------------------

KNOWN_SYMBOLS: list[str] = [
    "btc", "eth", "bnb", "ada", "xrp", "doge", "sol",
    "dot", "matic", "link", "xbt", "avax", "bch", "eos",
    "lnk", "ltc", "neth", "xlm",
]


def extract_symbol_from_strategy(strategy_name: str) -> str:
    parts = strategy_name.lower().replace("-", "_").split("_")
    for part in parts:
        if part in KNOWN_SYMBOLS:
            return part
    name_lower = strategy_name.lower()
    for sym in KNOWN_SYMBOLS:
        if sym in name_lower:
            return sym
    return ""


# ---------------------------------------------------------------------------
# Dynamic table factories
# ---------------------------------------------------------------------------

def get_price_table(schema: str, symbol: str) -> Table:
    """<schema>.<symbol>_1m – OHLCV candles."""
    meta = MetaData(schema=schema)
    return Table(
        f"{symbol}_1m",
        meta,
        Column("datetime", DateTime(timezone=True), primary_key=True, nullable=False),
        Column("open",     Float, nullable=False),
        Column("high",     Float, nullable=False),
        Column("low",      Float, nullable=False),
        Column("close",    Float, nullable=False),
        Column("volume",   Float, nullable=True),
    )


def get_predictions_table(strategy_name: str) -> Table:
    """strategy_signals.<strategy_name> – pre-computed signals."""
    meta = MetaData(schema="strategy_signals")
    return Table(
        strategy_name,
        meta,
        Column("datetime", DateTime(timezone=True), primary_key=True, nullable=False),
        Column("signals",  SmallInteger, nullable=False),
    )


# ---------------------------------------------------------------------------
# Saved-run registry  (one row per completed backtest run)
# Schema: backtest_runs   Table: run_registry
# ---------------------------------------------------------------------------

_RUN_REGISTRY_META = MetaData(schema="backtest_runs")

RUN_REGISTRY_TABLE = Table(
    "run_registry",
    _RUN_REGISTRY_META,
    Column("id",            Integer,  primary_key=True, autoincrement=True),
    Column("table_name",    String,   nullable=False, unique=True),   # e.g. sig_1h_btc_1_run_3
    Column("strategy_name", String,   nullable=False),
    Column("exchange",      String,   nullable=False),
    Column("start_date",    String,   nullable=True),
    Column("end_date",      String,   nullable=True),
    Column("take_profit",   Float,    nullable=False),
    Column("stop_loss",     Float,    nullable=False),
    Column("total_trades",  Integer,  nullable=False),
    Column("win_rate",      Float,    nullable=False),
    Column("total_pnl_pct", Float,    nullable=False),
    Column("final_balance", Float,    nullable=False),
    Column("created_at",    DateTime, nullable=False),
)


def get_ledger_run_table(table_name: str) -> Table:
    """
    backtest_runs.<table_name>  – one row per trade event (buy/sell).
    table_name follows the pattern  <strategy_name>_run_<i>.
    """
    meta = MetaData(schema="backtest_runs")
    return Table(
        table_name,
        meta,
        Column("id",                 Integer,  primary_key=True, autoincrement=True),
        Column("datetime",           DateTime, nullable=False),
        Column("action",             String,   nullable=False),   # "buy" | "sell - take_profit" …
        Column("buy_price",          Float,    nullable=True),
        Column("sell_price",         Float,    nullable=True),
        Column("pnl",                Float,    nullable=True),
        Column("pnl_sum",            Float,    nullable=True),
        Column("balance",            Float,    nullable=False),
        Column("predicted_direction",String,   nullable=False),
    )