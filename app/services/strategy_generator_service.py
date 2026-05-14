"""
TradeX – Strategy Generator Service

Pipeline
--------
1.  Validate timeframe and exchange
2.  Load 1-minute OHLCV for the symbol from the exchange DB schema
3.  Resample to the requested timeframe
4.  Randomise indicator flags  (signals_combiner.randomize_indicators)
5.  Run signals combiner with majority voting
6.  Run BackTest engine on 1m price data with the signals
7.  Generate unique strategy ID  (strategy_counter.generate_strategy_id)
8.  Save signals to strategy_signals.<strategy_id>
9.  Save strategy metadata to strategies.strategy_registry
10. Return CreateStrategyResponse with full ledger + summary
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.data_model import EXCHANGE_SCHEMA_MAP, EXCHANGE_COINS
from app.schemas.backtest_schema import (
    BacktestSummary,
    LedgerEntry,
    PnLPoint,
    WinLossPoint,
)
from app.schemas.strategy_generator_schema import (
    CreateStrategyRequest,
    CreateStrategyResponse,
)

# Reuse the exact same ledger-persistence helpers from backtest_service.
# This ensures create_strategy writes to backtest_runs in an identical way
# to POST /backtest/run, so the Run History panel works for both flows.
from app.services.backtest_service import (
    _next_run_index,
    _persist_run,
    _update_strategy_stats,
)

# ---------------------------------------------------------------------------
# Ensure TradeX project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Valid options
# ---------------------------------------------------------------------------
VALID_TIMEFRAMES = {"1h", "15m", "5m"}


# ===========================================================================
# Streak helper (reuse same logic as backtest_service)
# ===========================================================================

def _streaks(wins: list[bool]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


# ===========================================================================
# Synchronous heavy lifting – runs in asyncio.to_thread
# ===========================================================================

def _generate_and_run_sync(req: CreateStrategyRequest) -> dict:
    """
    Pure-Python, synchronous work:
      - Load OHLCV via psycopg2
      - Resample to timeframe
      - Randomise + run signals combiner
      - Run BackTest engine
      - Generate strategy ID, save signals + metadata via TradeX DB utils

    Returns a plain dict with all data needed to build the response.
    Raises ValueError / RuntimeError on failure.
    """
    import re
    import numpy as np
    import psycopg2

    from TradeX.indicators.talib.indicators import ALL_INDICATORS
    from TradeX.backtest.backtest import BackTest
    from TradeX.utils.data.data_cleaner import resample_ohlcv
    from TradeX.utils.db.utils import save_df_to_db
    from TradeX.strategy_generator.signals_combiner import (
        randomize_indicators,
        run_active_signals_with_voting,
    )
    from TradeX.strategy_generator.strategy_counter import generate_strategy_id

    # ── 1. Validate exchange / symbol ────────────────────────────────────
    exchange = req.exchange.lower()
    schema = EXCHANGE_SCHEMA_MAP.get(exchange)
    if not schema:
        raise ValueError(
            f"Unknown exchange '{req.exchange}'. "
            f"Valid: {list(EXCHANGE_SCHEMA_MAP.keys())}"
        )

    symbol = req.symbol.lower()
    valid_symbols = [s for s, _ in EXCHANGE_COINS.get(exchange, [])]
    if symbol not in valid_symbols:
        raise ValueError(
            f"Symbol '{symbol}' not available on exchange '{exchange}'. "
            f"Available: {valid_symbols}"
        )

    timeframe = req.timeframe.lower()
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe '{req.timeframe}'. Valid: {sorted(VALID_TIMEFRAMES)}"
        )

    # ── 2. Load 1-min OHLCV ──────────────────────────────────────────────
    dsn = (
        os.environ.get("SYNC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("DB_URL")
    )
    if not dsn:
        raise ValueError(
            "No database URL found in environment. "
            "Set DATABASE_URL, DB_URL, or SYNC_DATABASE_URL."
        )

    psycopg2_dsn = re.sub(r"^(postgresql|postgres)\+\w+://", r"\1://", dsn)

    conditions: list[str] = []
    params: list[str] = []
    if req.start_date:
        conditions.append("datetime >= %s::timestamptz")
        params.append(str(req.start_date))
    if req.end_date:
        conditions.append("datetime <= %s::timestamptz")
        params.append(str(req.end_date))

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_clause = "" if (req.start_date or req.end_date) else "LIMIT 43200"

    sql = (
        f"SELECT datetime, open, high, low, close, volume "
        f'FROM {schema}."{symbol}_1m" '
        f"{where_clause} "
        f"ORDER BY datetime ASC "
        f"{limit_clause}"
    ).strip()

    conn = psycopg2.connect(psycopg2_dsn)
    try:
        df_1m = pd.read_sql(sql, conn, params=params if params else None)
    finally:
        conn.close()

    if df_1m.empty:
        raise ValueError(
            f"No price data found for {symbol} on {exchange} "
            "in the selected date range. Fetch data first from the Data tab."
        )

    # Use tz_convert(None) — not tz_localize(None) — to strip timezone.
    # tz_localize(None) raises TypeError on tz-aware series (pandas >= 2.0);
    # tz_convert(None) safely removes tz whether or not it is present.
    _dt = pd.to_datetime(df_1m["datetime"])
    df_1m["datetime"] = _dt.dt.tz_convert(None) if _dt.dt.tz is not None else _dt

    # ── 3. Resample ──────────────────────────────────────────────────────
    df_tf = resample_ohlcv(df_1m, timeframe)
    if df_tf.empty:
        raise ValueError(
            f"Resampling 1m data to '{timeframe}' produced an empty DataFrame. "
            "Ensure enough historical data exists for the selected date range."
        )

    open_  = df_tf["open"].values
    high   = df_tf["high"].values
    low    = df_tf["low"].values
    close_ = df_tf["close"].values
    volume = df_tf["volume"].values
    # Pass .values (numpy array) not the raw Series so that tz info
    # is stripped before entering signals_combiner. backtest_service
    # does the same thing. Passing the Series keeps tz alive and
    # causes BackTest's np.searchsorted to crash.
    timestamps = df_tf["datetime"].values

    # ── 4. Randomise indicators ──────────────────────────────────────────
    flags = randomize_indicators(ALL_INDICATORS)

    # ── 5. Run signals combiner ──────────────────────────────────────────
    signals, windows_dict = run_active_signals_with_voting(
        flags, open_, high, low, close_, volume, timestamps,
    )

    if signals.empty:
        raise ValueError(
            "Signal combiner returned no signals for the selected date range. "
            "Try a wider date range or a different timeframe."
        )

    # ── 6. Generate strategy ID ──────────────────────────────────────────
    strategy_id = generate_strategy_id(symbol, flags, timeframe=timeframe)

    # ── 7. Save signals to DB ────────────────────────────────────────────
    save_df_to_db(
        df=signals,
        schema="strategy_signals",
        table_name=strategy_id,
        time_column="datetime",
        is_timeseries=True,
    )

    # ── 8. Run BackTest on 1m price data ─────────────────────────────────
    # Strip tz from signals["datetime"] so it matches the tz-naive df_1m.
    # run_active_signals_with_voting internally calls pd.to_datetime() which
    # can return tz-aware timestamps (e.g. datetime64[ns, UTC]) even when the
    # input numpy array looked tz-naive — causing "Cannot compare tz-naive and
    # tz-aware timestamps" inside BackTest's np.searchsorted calls.
    _sig_dt = pd.to_datetime(signals["datetime"])
    signals["datetime"] = (
        _sig_dt.dt.tz_convert(None) if _sig_dt.dt.tz is not None else _sig_dt
    )

    bt = BackTest(
        df_price=df_1m,
        df_predictions=signals,
        starting_balance=req.starting_balance,
        take_profit=req.take_profit,
        stop_loss=req.stop_loss,
        fee=req.fee,
        leverage=req.leverage,
        slippage=req.slippage,
    )
    df_ledger, final_balance, total_pnl_pct = bt.run()

    if df_ledger.empty:
        raise ValueError(
            "BackTest completed but produced no trades. "
            "Try a wider date range."
        )

    # ── 9. Save strategy metadata ────────────────────────────────────────
    row_data: dict = {**flags}

    for ind_name, params in windows_dict.items():
        for param_name, value in params.items():
            row_data[f"{ind_name}_{param_name}"] = value

    strategy_df = pd.DataFrame([row_data])
    strategy_df.insert(0, "pnl_sum", total_pnl_pct)
    strategy_df.insert(0, "timehorizon", timeframe)
    strategy_df.insert(0, "symbol", symbol)
    strategy_df.insert(0, "sl", str(req.stop_loss))
    strategy_df.insert(0, "tp", str(req.take_profit))
    strategy_df.insert(0, "strategy", req.name)  # user-entered display name
    strategy_df.columns = strategy_df.columns.str.lower()

    save_df_to_db(
        df=strategy_df,
        table_name="strategy_registry",
        schema="strategies",
        time_column=None,
        is_timeseries=False,
    )

    return {
        "strategy_id": strategy_id,
        "df_ledger": df_ledger,
        "final_balance": final_balance,
        "total_pnl_pct": total_pnl_pct,
    }


# ===========================================================================
# Build response from ledger DataFrame
# ===========================================================================

def _build_response(
    result: dict,
    req: CreateStrategyRequest,
    run_table_name: Optional[str] = None,
) -> CreateStrategyResponse:
    df_ledger: pd.DataFrame = result["df_ledger"]
    final_balance: float = result["final_balance"]
    total_pnl_pct: float = result["total_pnl_pct"]
    strategy_id: str = result["strategy_id"]

    ledger_entries: list[LedgerEntry] = []
    for _, row in df_ledger.iterrows():
        action: str = str(row["action"])
        is_buy = action == "buy"
        reason = None
        if not is_buy:
            parts = action.split(" - ", 1)
            reason = parts[1] if len(parts) == 2 else action

        ledger_entries.append(LedgerEntry(
            date=str(row["datetime"]),
            type="Buy" if is_buy else "Sell",
            price=float(row["buy_price"] if is_buy else row["sell_price"]),
            pnl=float(row["pnl"]) if pd.notna(row.get("pnl")) else None,
            pnl_sum=float(row["pnl_sum"]) if "pnl_sum" in row.index and pd.notna(row["pnl_sum"]) else None,
            balance=float(row["balance"]),
            direction=str(row["predicted_direction"]),
            reason=reason,
        ))

    sell_rows = df_ledger[df_ledger["action"].str.startswith("sell")]
    total_trades = len(sell_rows)
    win_mask = (sell_rows["pnl"] > 0).tolist()
    win_trades = sum(win_mask)
    loss_trades = total_trades - win_trades
    win_rate = round(win_trades / total_trades * 100, 2) if total_trades else 0.0
    loss_rate = round(100 - win_rate, 2)
    max_w, max_l = _streaks(win_mask)

    summary = BacktestSummary(
        strategy_name=req.name,        # user display name, not the internal sig_* ID
        exchange=req.exchange,
        symbol=req.symbol.lower(),
        starting_balance=req.starting_balance,
        final_balance=final_balance,
        total_pnl_pct=total_pnl_pct,
        total_trades=total_trades,
        win_trades=win_trades,
        loss_trades=loss_trades,
        win_rate=win_rate,
        loss_rate=loss_rate,
        max_consecutive_wins=max_w,
        max_consecutive_losses=max_l,
        run_table_name=run_table_name,  # set after DB persist
    )

    win_loss_data = [
        WinLossPoint(name="Trades Won", value=win_trades),
        WinLossPoint(name="Trades Lost", value=loss_trades),
    ]
    pnl_data = [
        PnLPoint(trade=i + 1, pnl=round(float(pnl), 2))
        for i, pnl in enumerate(sell_rows["pnl"].tolist())
    ]

    return CreateStrategyResponse(
        strategy_id=strategy_id,
        display_name=req.name,
        timeframe=req.timeframe,
        symbol=req.symbol.lower(),
        exchange=req.exchange,
        summary=summary,
        ledger=ledger_entries,
        win_loss_data=win_loss_data,
        pnl_data=pnl_data,
        message=f"Strategy '{req.name}' created successfully.",
    )


# ===========================================================================
# Public entry-point
# ===========================================================================

async def create_strategy(
    db: AsyncSession,
    req: CreateStrategyRequest,
) -> CreateStrategyResponse:
    """Full pipeline – validate, generate, backtest, persist ledger, return."""

    # Validate timeframe early (fast, no I/O)
    if req.timeframe.lower() not in VALID_TIMEFRAMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid timeframe '{req.timeframe}'. Valid options: {sorted(VALID_TIMEFRAMES)}",
        )

    # Validate exchange
    if req.exchange.lower() not in EXCHANGE_SCHEMA_MAP:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown exchange '{req.exchange}'. Valid: {list(EXCHANGE_SCHEMA_MAP.keys())}",
        )

    # ── Heavy sync work in thread-pool (OHLCV load, signals, backtest) ───────
    try:
        result = await asyncio.to_thread(_generate_and_run_sync, req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy generation failed: {exc}",
        )

    # ── Persist ledger to backtest_runs (identical to POST /backtest/run) ─────
    df_ledger: pd.DataFrame = result["df_ledger"]
    final_balance: float    = result["final_balance"]
    total_pnl_pct: float    = result["total_pnl_pct"]

    sell_rows    = df_ledger[df_ledger["action"].str.startswith("sell")]
    total_trades = len(sell_rows)
    win_trades   = int((sell_rows["pnl"] > 0).sum())
    win_rate     = round(win_trades / total_trades * 100, 2) if total_trades else 0.0

    # Index runs under the user display name so Run History shows it correctly.
    run_index = await _next_run_index(db, req.name)

    # _persist_run / _update_strategy_stats expect a request-like object that
    # exposes exchange, start_date, end_date, take_profit, stop_loss.
    # We build a minimal adapter rather than coupling to BacktestRunRequest.
    class _ReqAdapter:
        exchange    = req.exchange
        start_date  = req.start_date
        end_date    = req.end_date
        take_profit = req.take_profit
        stop_loss   = req.stop_loss

    run_table_name = await _persist_run(
        db            = db,
        df_ledger     = df_ledger,
        req           = _ReqAdapter(),
        strategy_name = req.name,     # user display name, not the internal sig_* ID
        run_index     = run_index,
        final_balance = final_balance,
        total_pnl_pct = total_pnl_pct,
        total_trades  = total_trades,
        win_rate      = win_rate,
    )

    await _update_strategy_stats(
        db            = db,
        strategy_name = req.name,
        total_pnl_pct = total_pnl_pct,
        take_profit   = req.take_profit,
        stop_loss     = req.stop_loss,
    )

    return _build_response(result, req, run_table_name=run_table_name)