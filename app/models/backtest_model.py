"""
TradeX – ORM models for backtest data sources.

Two dynamic tables are resolved at runtime:
  1. Price data   → <exchange_schema>.<symbol>_1m   (same tables as DataTab)
  2. Predictions  → strategies.<strategy_name>      (one table per strategy)

Both use SQLAlchemy Table() so the table name can be injected at call time.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, SmallInteger, String, MetaData, Table

# ---------------------------------------------------------------------------
# Known coin keywords for symbol extraction from a strategy name
# e.g.  "sig_1h_btc_1"  →  "btc"
# ---------------------------------------------------------------------------

KNOWN_SYMBOLS: list[str] = [
    "btc", "eth", "bnb", "ada", "xrp", "doge", "sol",
    "dot", "matic", "link", "xbt", "avax", "bch", "eos",
    "lnk", "ltc", "neth", "xlm",
]


def extract_symbol_from_strategy(strategy_name: str) -> str:
    """
    Scan the strategy name for a known coin token.

    Examples
    --------
    "sig_1h_btc_1"       → "btc"
    "random_forest_eth"  → "eth"
    "my_bnb_strategy_v2" → "bnb"

    Falls back to empty string if nothing matches.
    """
    parts = strategy_name.lower().replace("-", "_").split("_")
    for part in parts:
        if part in KNOWN_SYMBOLS:
            return part
    # second pass: substring match for compound tokens like "btcusdt"
    name_lower = strategy_name.lower()
    for sym in KNOWN_SYMBOLS:
        if sym in name_lower:
            return sym
    return ""


# ---------------------------------------------------------------------------
# Dynamic table factories (no fixed ORM class needed)
# ---------------------------------------------------------------------------

def get_price_table(schema: str, symbol: str) -> Table:
    """
    Return a Table descriptor for  <schema>.<symbol>_1m.
    Columns: datetime (PK), open, high, low, close, volume
    """
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
    """
    Return a Table descriptor for  strategy_signals.<strategy_name>.
    Columns: datetime (PK), signals
    """
    meta = MetaData(schema="strategy_signals")
    return Table(
        strategy_name,
        meta,
        Column("datetime", DateTime(timezone=True), primary_key=True, nullable=False),
        Column("signals",  SmallInteger, nullable=False),
    )