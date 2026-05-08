"""
TradeX – Data Service

Responsibilities
----------------
1. Return exchange list and per-exchange coin list (static catalogue)
2. Run the correct fetcher in a thread-pool worker (non-blocking)
3. Read OHLCV candles from the DB filtered by an optional date range
   (no resampling on the Data tab – 1-min candles only)
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
    LastDateResponse,
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
# Timeframe → X-axis time format for frontend labels
# (Only 1m is used on the Data tab, but kept for completeness)
# ---------------------------------------------------------------------------
TIMEFRAME_FMT: dict[str, str] = {
    "1m":  "%b %d, %H:%M",
    "5m":  "%b %d, %H:%M",
    "15m": "%b %d, %H:%M",
    "1h":  "%b %d, %H:%M",
    "4h":  "%b %d, %H:%M",
    "1d":  "%b %d, %Y",
}

# Default cap when neither start_date nor end_date is provided
_DEFAULT_CANDLE_LIMIT = 120


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
            detail=f"MetaTrader5 is not supported in the cloud deployment. "
               "Run the backend locally to use MT5.",
        )


# ===========================================================================
# Last-date query  (used by frontend to pre-fill start_date)
# ===========================================================================

async def get_last_date_for_coin(
    db: AsyncSession,
    exchange: str,
    symbol: str,
) -> LastDateResponse:
    """
    Return the most recent stored datetime for a given exchange/symbol pair.
    If no data exists, last_date is None (frontend treats this as "user can pick date").
    """
    exchange = exchange.lower()
    symbol   = symbol.lower().split("/")[0].split("-")[0].strip()

    if exchange not in EXCHANGE_SCHEMA_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'.",
        )

    schema     = EXCHANGE_SCHEMA_MAP[exchange]
    table_name = f"{symbol}_1m"
    tbl        = get_ohlcv_table(schema, table_name)

    try:
        stmt   = select(tbl.c.datetime).order_by(tbl.c.datetime.desc()).limit(1)
        result = await db.execute(stmt)
        row    = result.fetchone()
    except Exception:
        # Table doesn't exist yet — normal case before first fetch
        return LastDateResponse(exchange=exchange, symbol=symbol, last_date=None)

    last_date = row[0].isoformat() if row else None
    return LastDateResponse(exchange=exchange, symbol=symbol, last_date=last_date)


# ===========================================================================
# OHLCV reader  –  SQL-level date filtering, no resampling
# ===========================================================================

async def read_ohlcv(
    db: AsyncSession,
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> OHLCVResponse:
    """
    Read 1-minute candles from the DB, filtered by the requested date range,
    and return chart-ready data.  No resampling is performed on the Data tab.

    When neither start_date nor end_date is given, the most recent
    _DEFAULT_CANDLE_LIMIT candles are returned (same behaviour as before).

    Parameters
    ----------
    start_date : ISO date/datetime string, inclusive (e.g. "2024-01-01")
    end_date   : ISO date/datetime string, inclusive (e.g. "2024-03-31")
    """
    exchange = exchange.lower()
    symbol   = symbol.lower().split("/")[0].split("-")[0].strip()

    if exchange not in EXCHANGE_SCHEMA_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'.",
        )

    schema     = EXCHANGE_SCHEMA_MAP[exchange]
    table_name = f"{symbol}_1m"
    tbl        = get_ohlcv_table(schema, table_name)

    # ── Build the query with optional date filters ────────────────────────
    try:
        stmt = select(tbl).order_by(tbl.c.datetime.asc())

        if start_date:
            # Cast the date string to timestamptz inside SQL (mirrors backtest_service)
            stmt = stmt.where(
                tbl.c.datetime >= text(f"'{start_date}'::timestamptz")
            )
        if end_date:
            # Include the full end day: push end to end-of-day if only a date is given
            end_ts = end_date if "T" in end_date or " " in end_date else f"{end_date}T23:59:59"
            stmt = stmt.where(
                tbl.c.datetime <= text(f"'{end_ts}'::timestamptz")
            )

        # Cap to _DEFAULT_CANDLE_LIMIT when no explicit range is supplied
        if not start_date and not end_date:
            stmt = stmt.order_by(tbl.c.datetime.desc()).limit(_DEFAULT_CANDLE_LIMIT)

        result = await db.execute(stmt)
        rows   = result.fetchall()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{schema}.{table_name}' not found or unreadable. "
                   f"Please fetch data first. Detail: {exc}",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for {exchange}/{symbol} in the requested range. "
                   "Please fetch data first or adjust the date filter.",
        )

    # ── Build DataFrame and sort ascending ───────────────────────────────
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ── Build chart candles (1m, no resampling) ──────────────────────────
    fmt = TIMEFRAME_FMT.get(timeframe, "%b %d, %H:%M")
    candles: list[OHLCVCandle] = [
        OHLCVCandle(
            time=row.datetime.strftime(fmt),
            date=row.datetime.strftime("%Y-%m-%d"),   # ISO date for any residual client filtering
            open=round(row.open, 4),
            high=round(row.high, 4),
            low=round(row.low, 4),
            close=round(row.close, 4),
            volume=round(row.volume, 2),
        )
        for row in df.itertuples()
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