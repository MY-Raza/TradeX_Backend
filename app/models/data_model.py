"""
TradeX – Dynamic OHLCV ORM model

Each exchange stores candles in its own schema with per-symbol tables:
    <exchange_schema>.<symbol>_1m

Because the table name is only known at runtime, we use SQLAlchemy's
`Table` reflection helper rather than a fixed mapped class.

Usage
-----
    from app.models.data_model import get_ohlcv_table, EXCHANGE_SCHEMA_MAP
    table = get_ohlcv_table(schema="binance_data", table_name="btc_1m")
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
)

# ---------------------------------------------------------------------------
# Exchange → DB schema mapping
# Must mirror TradeX.utils.common.constants.EXCHANGE_SCHEMA_MAP
# ---------------------------------------------------------------------------

EXCHANGE_SCHEMA_MAP: dict[str, str] = {
    "binance":      "data_binance",
    "bybit":        "data_bybit",
    "kraken":       "data_kraken",
    "metatrader5":  "data_mt5",
}

# ---------------------------------------------------------------------------
# Per-exchange coin lists
# Keys match the exchange id; values are (symbol_key, display_label) tuples.
# These should mirror the config.yml files of each fetcher.
# ---------------------------------------------------------------------------

EXCHANGE_COINS: dict[str, list[tuple[str, str]]] = {
    "binance": [
        ("btc",   "BTC/USDT"),
        ("eth",   "ETH/USDT"),
        ("bnb",   "BNB/USDT"),
        ("ada",   "ADA/USDT"),
        ("xrp",   "XRP/USDT"),
        ("doge",  "DOGE/USDT"),
        ("sol",   "SOL/USDT"),
        ("dot",   "DOT/USDT"),
        ("matic", "MATIC/USDT"),
        ("link",  "LINK/USDT"),
    ],
    "bybit": [
        ("btc",   "BTC/USDT"),
        ("eth",   "ETH/USDT"),
        ("bnb",   "BNB/USDT"),
        ("ada",   "ADA/USDT"),
        ("xrp",   "XRP/USDT"),
        ("doge",  "DOGE/USDT"),
        ("sol",   "SOL/USDT"),
        ("dot",   "DOT/USDT"),
        ("matic", "MATIC/USDT"),
        ("link",  "LINK/USDT"),
    ],
    "kraken": [
        ("xbt",  "XBT/USD"),
        ("eth",  "ETH/USD"),
        ("bnb",  "BNB/USD"),
        ("ada",  "ADA/USD"),
        ("xrp",  "XRP/USD"),
        ("doge", "DOGE/USD"),
        ("sol",  "SOL/USD"),
        ("dot",  "DOT/USD"),
        ("link", "LINK/USD"),
        ("ltc",  "LTC/USD"),
    ],
    "metatrader5": [
        ("ada",   "ADA/USD"),
        ("avax",  "AVAX/USD"),
        ("bch",   "BCH/USD"),
        ("bnb",   "BNB/USD"),
        ("btc",   "BTC/USD"),
        ("doge",  "DOGE/USD"),
        ("dot",   "DOT/USD"),
        ("eos",   "EOS/USD"),
        ("eth",   "ETH/USD"),
        ("lnk",   "LNK/USD"),
        ("ltc",   "LTC/USD"),
        ("matic", "MATIC/USD"),
        ("neth",  "NETH/USD"),
        ("sol",   "SOL/USD"),
        ("xlm",   "XLM/USD"),
    ],
}

# ---------------------------------------------------------------------------
# Human-readable exchange labels (for the dropdown)
# ---------------------------------------------------------------------------

EXCHANGE_LABELS: dict[str, str] = {
    "binance":     "Binance",
    "bybit":       "Bybit",
    "kraken":      "Kraken",
    "metatrader5": "Mt5 Trader",
}


# ---------------------------------------------------------------------------
# Dynamic Table factory
# Returns a SQLAlchemy Table object for the given schema + table name.
# A fresh MetaData is used so tables don't collide across calls.
# ---------------------------------------------------------------------------

def get_ohlcv_table(schema: str, table_name: str) -> Table:
    """
    Build a SQLAlchemy Table descriptor for a dynamic OHLCV table.

    Parameters
    ----------
    schema     : DB schema name, e.g. 'binance_data'
    table_name : table name, e.g. 'btc_1m'

    Returns
    -------
    sqlalchemy.Table – ready for use in select() / execute()
    """
    meta = MetaData(schema=schema)
    return Table(
        table_name,
        meta,
        Column("datetime", DateTime(timezone=True), primary_key=True, nullable=False),
        Column("open",     Float,                nullable=False),
        Column("high",     Float,                nullable=False),
        Column("low",      Float,                nullable=False),
        Column("close",    Float,                nullable=False),
        Column("volume",   Float,                nullable=True),
    )