"""
TradeX – Data Service

Responsibilities
----------------
1. Return exchange list and per-exchange coin list (static catalogue)
2. Run the correct fetcher in a thread-pool worker (non-blocking)
3. Read OHLCV candles from the DB and resample to the requested timeframe
4. Return chart-ready OHLCVCandle objects to the router
"""

from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.data_model import (
    EXCHANGE_COINS,
    EXCHANGE_LABELS,
    EXCHANGE_SCHEMA_MAP,
    get_ohlcv_table,
)
from app.schemas.data_schema import (
    CoinInfo,
    ExchangeInfo,
    FetchRequest,
    FetchResponse,
    OHLCVCandle,
    OHLCVResponse,
)

# ---------------------------------------------------------------------------
# Project root on sys.path so fetcher imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Timeframe → pandas resample rule
# ---------------------------------------------------------------------------
TIMEFRAME_RESAMPLE: dict[str, str] = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}

# ---------------------------------------------------------------------------
# Timeframe → X-axis time format for frontend labels
# ---------------------------------------------------------------------------
TIMEFRAME_FMT: dict[str, str] = {
    "1m":  "%H:%M",
    "5m":  "%H:%M",
    "15m": "%H:%M",
    "1h":  "%H:%M",
    "4h":  "%b %d %H:%M",
    "1d":  "%b %d",
}

# ---------------------------------------------------------------------------
# How many candles to return per timeframe (keeps chart readable)
# ---------------------------------------------------------------------------
TIMEFRAME_LIMIT: dict[str, int] = {
    "1m":  120,
    "5m":  120,
    "15m": 96,
    "1h":  72,
    "4h":  60,
    "1d":  90,
}


# ===========================================================================
# Static catalogue helpers
# ===========================================================================

def get_exchanges() -> list[ExchangeInfo]:
    """Return all supported exchanges for the dropdown."""
    return [
        ExchangeInfo(id=eid, label=label)
        for eid, label in EXCHANGE_LABELS.items()
    ]


def get_coins(exchange: str) -> list[CoinInfo]:
    """Return coins available for a given exchange."""
    exchange = exchange.lower()
    if exchange not in EXCHANGE_COINS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'. "
                   f"Valid options: {list(EXCHANGE_LABELS.keys())}",
        )
    return [
        CoinInfo(symbol=sym, label=label)
        for sym, label in EXCHANGE_COINS[exchange]
    ]


# ===========================================================================
# Fetcher runner  (executes in a thread-pool via asyncio.to_thread)
# ===========================================================================

def _run_fetcher_sync(req: FetchRequest) -> int:
    """
    Synchronous fetcher runner – called inside asyncio.to_thread so it
    does not block the event loop.

    Returns the number of rows saved (best-effort; falls back to 0).
    """
    exchange = req.exchange.lower()
    # Normalize symbol: "btc/usdt" → "btc"
    symbol = req.symbol.lower().split("/")[0].split("-")[0].strip()

    # ── Resolve fetcher class ────────────────────────────────────────────
    if exchange == "binance":
        from TradeX.data.binance.binance_fetcher import BinanceFuturesFetcher
        from TradeX.utils.db.utils import save_df_to_db, get_last_date
        from TradeX.utils.data.data_cleaner import clean_df

        schema = EXCHANGE_SCHEMA_MAP["binance"]
        last = get_last_date(f"{symbol}_1m", schema, "datetime")
        start = (
            (last + pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            if last else req.start_date
        )
        fetcher = BinanceFuturesFetcher(
            symbol=f"{symbol.upper()}USDT",
            start_date=start,
            end_date=req.end_date,
            interval="1m",
        )
        raw_df = fetcher.fetch_data()
        if raw_df.empty:
            return 0
        df = clean_df(raw_df)
        save_df_to_db(df, f"{symbol}_1m", schema, "datetime", is_timeseries=True)
        return len(df)

    elif exchange == "bybit":
        from TradeX.data.bybit.bybit_fetcher import BybitFuturesFetcher
        from TradeX.utils.db.utils import save_df_to_db, get_last_date
        from TradeX.utils.data.data_cleaner import clean_df

        schema = EXCHANGE_SCHEMA_MAP["bybit"]
        last = get_last_date(f"{symbol}_1m", schema, "datetime")
        start = (
            (last + pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            if last else req.start_date
        )
        fetcher = BybitFuturesFetcher(
            symbol=f"{symbol.upper()}USDT",
            start_date=start,
            end_date=req.end_date,
            interval="1",
        )
        raw_df = fetcher.fetch_data()
        if raw_df.empty:
            return 0
        df = clean_df(raw_df)
        save_df_to_db(df, f"{symbol}_1m", schema, "datetime", is_timeseries=True)
        return len(df)

    elif exchange == "kraken":
        from TradeX.data.kraken.kraken_fetcher import KrakenFuturesFetcher
        from TradeX.utils.db.utils import save_df_to_db, get_last_date
        from TradeX.utils.data.data_cleaner import clean_df

        schema = EXCHANGE_SCHEMA_MAP["kraken"]
        last = get_last_date(f"{symbol}_1m", schema, "datetime")
        start = (
            (last + pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d %H:%M:%S")
            if last else req.start_date
        )
        kraken_symbol = f"PF_{symbol.upper()}USD"
        fetcher = KrakenFuturesFetcher(symbol=kraken_symbol, interval="1m")
        raw_df = fetcher.fetch_data(start_date=start, end_date=req.end_date)
        if raw_df.empty:
            return 0
        df = clean_df(raw_df)
        save_df_to_db(df, f"{symbol}_1m", schema, "datetime", is_timeseries=True)
        return len(df)

    elif exchange == "metatrader5":
        import MetaTrader5 as mt5_lib
        from TradeX.data.mt5.metatrader5_fetcher import MetaTrader5FutureFetcher
        from TradeX.utils.db.utils import save_df_to_db, get_last_date
        from TradeX.utils.data.data_cleaner import clean_df
        from dotenv import load_dotenv

        load_dotenv()
        login    = int(os.getenv("MT5_LOGIN", "0"))
        password = os.getenv("MT5_PASSWORD", "")
        server   = os.getenv("MT5_SERVER", "")
        if not mt5_lib.initialize(login=login, password=password, server=server):
            raise RuntimeError(f"MT5 init failed: {mt5_lib.last_error()}")

        schema = EXCHANGE_SCHEMA_MAP["metatrader5"]
        last   = get_last_date(f"{symbol}_1m", schema, "datetime")
        utc_from = (
            (last + pd.Timedelta(milliseconds=1)).to_pydatetime()
            if last
            else datetime.fromisoformat(req.start_date)
        )
        utc_to = (
            datetime.now(timezone.utc)
            if req.end_date.lower() == "now"
            else datetime.fromisoformat(req.end_date)
        )
        fetcher = MetaTrader5FutureFetcher(
            symbols=[symbol],
            utc_from=utc_from,
            utc_to=utc_to,
            timeframe=mt5_lib.TIMEFRAME_M1,
        )
        raw_df = fetcher.fetch(symbol)
        mt5_lib.shutdown()
        if raw_df is None or raw_df.empty:
            return 0
        df = clean_df(raw_df, "1m")
        save_df_to_db(df, f"{symbol}_1m", schema, "datetime", is_timeseries=True)
        return len(df)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported exchange '{exchange}'.",
        )


# ===========================================================================
# OHLCV reader
# ===========================================================================

async def read_ohlcv(
    db: AsyncSession,
    exchange: str,
    symbol: str,
    timeframe: str = "1h",
) -> OHLCVResponse:
    """
    Read candles from the DB, resample to the requested timeframe,
    and return chart-ready data.
    """
    exchange = exchange.lower()
    # Normalize: "btc/usdt" or "BTC/USDT" → "btc"
    symbol = symbol.lower().split("/")[0].split("-")[0].strip()

    # Validate inputs
    if exchange not in EXCHANGE_SCHEMA_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'.",
        )
    if timeframe not in TIMEFRAME_RESAMPLE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown timeframe '{timeframe}'. "
                   f"Valid: {list(TIMEFRAME_RESAMPLE.keys())}",
        )

    schema     = EXCHANGE_SCHEMA_MAP[exchange]
    table_name = f"{symbol}_1m"
    limit      = TIMEFRAME_LIMIT[timeframe]

    # ── Build dynamic table and query ────────────────────────────────────
    tbl = get_ohlcv_table(schema, table_name)

    try:
        # Fetch the last N*resample_factor rows to have enough data after resampling.
        # For 1d resampling we need ~90 days * 1440 min = 129 600 raw rows.
        # We cap raw fetch at 200k rows for performance.
        raw_limit = min(limit * 1440, 200_000)
        stmt = (
            select(tbl)
            .order_by(tbl.c.datetime.desc())
            .limit(raw_limit)
        )
        result = await db.execute(stmt)
        rows = result.fetchall()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{schema}.{table_name}' not found or unreadable. "
                   f"Please fetch data first. Detail: {exc}",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for {exchange}/{symbol}. "
                   "Please fetch data first.",
        )

    # ── Build DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")

    # ── Resample ──────────────────────────────────────────────────────────
    rule = TIMEFRAME_RESAMPLE[timeframe]
    ohlcv = df.resample(rule).agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().tail(limit)

    # ── Build chart candles ───────────────────────────────────────────────
    fmt = TIMEFRAME_FMT[timeframe]
    candles: list[OHLCVCandle] = [
        OHLCVCandle(
            time=ts.strftime(fmt),
            open=round(row.open, 4),
            high=round(row.high, 4),
            low=round(row.low, 4),
            close=round(row.close, 4),
            volume=round(row.volume, 2),
        )
        for ts, row in ohlcv.iterrows()
    ]

    # ── Summary stats ─────────────────────────────────────────────────────
    return OHLCVResponse(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        total_rows=len(candles),
        open=candles[0].open if candles else 0.0,
        high=max(c.high for c in candles) if candles else 0.0,
        low=min(c.low for c in candles) if candles else 0.0,
        close=candles[-1].close if candles else 0.0,
        total_volume=round(sum(c.volume for c in candles), 2),
    )


# ===========================================================================
# Public fetch entry-point  (called by the router)
# ===========================================================================

async def fetch_and_store(req: FetchRequest) -> FetchResponse:
    """
    Run the exchange-specific fetcher in a thread-pool worker so it
    does not block the FastAPI event loop, then return a summary.
    """
    import asyncio

    exchange = req.exchange.lower()
    if exchange not in EXCHANGE_SCHEMA_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'.",
        )

    try:
        rows_saved: int = await asyncio.to_thread(_run_fetcher_sync, req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fetcher failed for {exchange}/{req.symbol}: {exc}",
        )

    return FetchResponse(
        exchange=exchange,
        symbol=req.symbol,
        rows_saved=rows_saved,
        message=(
            f"Successfully fetched and stored {rows_saved} candles "
            f"for {req.symbol.upper()} from {EXCHANGE_LABELS[exchange]}."
            if rows_saved > 0
            else f"No new candles to store for {req.symbol.upper()} "
                 f"from {EXCHANGE_LABELS[exchange]}. Data may already be up to date."
        ),
    )