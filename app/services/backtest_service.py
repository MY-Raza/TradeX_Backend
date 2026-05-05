"""
TradeX – Backtest Service  (v2)

Pipeline for POST /backtest/run
--------------------------------
1.  Fetch strategy metadata from strategy_registry
      → symbol, tp, sl, indicators, patterns, indicator_params
2.  Build flags dict  { indicator/pattern: True }  for every active signal
3.  Build windows dict from DB-stored indicator parameters
      (slowperiod, fastperiod, timeperiod, period, fastk, slowk …)
4.  Fetch OHLCV price data from <exchange_schema>.<symbol>_1m
      filtered to [start_date, end_date] if supplied
5.  Run signals_combiner.run_active_signals_with_voting()
      → returns (df_signals, _windows_dict)
6.  Instantiate BackTest with df_price + df_signals + tp/sl from request
7.  Persist ledger rows to backtest_runs.<strategy_name>_run_<i>
8.  Insert row into backtest_runs.run_registry
9.  Update strategy_registry with last_pnl_pct / last_run_tp / last_run_sl
10. Return BacktestResponse

Additional endpoints
---------------------
GET  /backtest/strategies                           → dropdown list
GET  /backtest/runs/{strategy_name}                 → list of saved runs
GET  /backtest/runs/{strategy_name}/{run_id}/ledger → paginated ledger
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy_model import StrategyRegistry
from app.models.backtest_model import (
    KNOWN_SYMBOLS,
    RUN_REGISTRY_TABLE,
    extract_symbol_from_strategy,
    get_ledger_run_table,
)
from app.models.data_model import EXCHANGE_SCHEMA_MAP
from app.schemas.backtest_schema import (
    BacktestResponse,
    BacktestRunRequest,
    BacktestStrategyOption,
    BacktestSummary,
    LedgerEntry,
    LedgerRunMeta,
    PaginatedLedger,
    PnLPoint,
    WinLossPoint,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so BackTest / signals_combiner resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===========================================================================
# Helper: streak counter
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
# Helper: build indicator windows from DB-stored strategy params
# ===========================================================================

# All parameter column names that may exist on StrategyRegistry rows
_WINDOW_PARAM_KEYS = ("slowperiod", "fastperiod", "timeperiod", "period",
                      "fastk", "slowk", "signalperiod")


def _build_windows_from_db(
    strategy_row: StrategyRegistry,
    indicator_names: list[str],
) -> dict:
    """
    Read the window/parameter columns from the strategy_registry row and map
    them back to indicator names.

    The strategy_registry is expected to carry JSONB / hstore columns like
    ``indicator_params`` (dict[str, dict[str, int|None]]) built at strategy-
    creation time.  If that attribute is absent we fall back to individual
    columns (slowperiod, fastperiod, …).

    Returns:  { indicator_name: { param_key: value, … }, … }
    """
    windows: dict = {}

    # ── Preferred path: indicator_params is a JSON column ─────────────────
    raw_params = getattr(strategy_row, "indicator_params", None)
    if raw_params and isinstance(raw_params, dict):
        for ind in indicator_names:
            if ind in raw_params and raw_params[ind]:
                filtered = {
                    k: v for k, v in raw_params[ind].items()
                    if v not in (None, 0)
                }
                if filtered:
                    windows[ind] = filtered
        return windows

    # ── Fallback: flat scalar columns on the row ───────────────────────────
    flat_params: dict = {}
    for key in _WINDOW_PARAM_KEYS:
        val = getattr(strategy_row, key, None)
        if val is not None and val != 0:
            flat_params[key] = val

    if flat_params:
        # Apply the same flat params to every indicator that doesn't already
        # have a dedicated entry (best-effort when metadata is minimal).
        for ind in indicator_names:
            if ind not in windows:
                windows[ind] = flat_params

    return windows


# ===========================================================================
# 1. Strategy dropdown
# ===========================================================================

async def get_backtest_strategies(db: AsyncSession) -> list[BacktestStrategyOption]:
    stmt = select(
        StrategyRegistry.strategy,
        StrategyRegistry.symbol,
        StrategyRegistry.timehorizon,
        StrategyRegistry.tp,
        StrategyRegistry.sl,
        StrategyRegistry.last_pnl_pct,
        StrategyRegistry.last_run_tp,
        StrategyRegistry.last_run_sl,
    ).order_by(StrategyRegistry.strategy)

    rows = (await db.execute(stmt)).all()
    return [
        BacktestStrategyOption(
            name=row.strategy,
            symbol=row.symbol,
            time_horizon=row.timehorizon,
            tp=row.tp,
            sl=row.sl,
            last_pnl_pct=float(row.last_pnl_pct) if row.last_pnl_pct is not None else None,
            last_run_tp=float(row.last_run_tp) if row.last_run_tp is not None else None,
            last_run_sl=float(row.last_run_sl) if row.last_run_sl is not None else None,
        )
        for row in rows
    ]


# ===========================================================================
# 2. Load price DataFrame (date-filtered) – synchronous, runs in thread-pool
#
# The previous implementation used fetch_ohlcv_df which does SELECT * on the
# full table and filters in Python — this kills the DB connection for large
# tables.  We instead build a targeted query with SQL-level WHERE / LIMIT so
# only the required rows travel over the wire.  Date strings are cast to
# timestamptz inside the SQL string itself so psycopg2 never needs to convert
# a Python Timestamp, avoiding the tz-naive asyncpg error entirely.
# ===========================================================================

def _load_price_df_sync(
    exchange: str,
    symbol: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    import os
    import psycopg2

    schema = EXCHANGE_SCHEMA_MAP.get(exchange.lower())
    if not schema:
        raise ValueError(
            f"Unknown exchange '{exchange}'. "
            f"Valid: {list(EXCHANGE_SCHEMA_MAP.keys())}"
        )

    table = f"{symbol}_1m"

    # Build WHERE clauses — date strings are cast inside SQL so psycopg2
    # never has to serialise a Python datetime/Timestamp object.
    conditions: list[str] = []
    params: list[str] = []

    if start_date:
        conditions.append("datetime >= %s::timestamptz")
        params.append(str(start_date))
    if end_date:
        conditions.append("datetime <= %s::timestamptz")
        params.append(str(end_date))

    where_clause  = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    # Cap at 30 days of 1-min candles only when no explicit range is given
    limit_clause  = "" if (start_date or end_date) else "LIMIT 43200"

    sql = (
        f"SELECT datetime, open, high, low, close, volume "
        f'FROM {schema}."{table}" '
        f"{where_clause} "
        f"ORDER BY datetime ASC "
        f"{limit_clause}"
    ).strip()

    # Resolve sync connection string — prefer an explicit plain-psycopg2 URL,
    # fall back to DATABASE_URL / DB_URL which may carry a SQLAlchemy dialect.
    dsn = (
        os.environ.get("SYNC_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("DB_URL")
    )
    if not dsn:
        raise ValueError(
            "No sync database URL found in environment. "
            "Set DATABASE_URL, DB_URL, or SYNC_DATABASE_URL."
        )

    # Strip SQLAlchemy driver suffix so psycopg2 can parse the URL.
    # e.g. postgresql+asyncpg://... → postgresql://...
    import re
    psycopg2_dsn = re.sub(r"^(postgresql|postgres)\+\w+://", r"\1://", dsn)

    conn = psycopg2.connect(psycopg2_dsn)
    try:
        df = pd.read_sql(sql, conn, params=params if params else None)
    finally:
        conn.close()

    if df.empty:
        raise ValueError(
            f"No price data found for {symbol} on {exchange} "
            f"in the selected date range. Fetch data first from the Data tab."
        )

    # Drop timezone so downstream BackTest / numpy stays tz-naive
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


# ===========================================================================
# 3. Run BackTest engine (synchronous, called inside asyncio.to_thread)
# ===========================================================================

def _run_engine_with_combiner(
    df_price: pd.DataFrame,
    strategy_row: StrategyRegistry,
    req: BacktestRunRequest,
) -> tuple[pd.DataFrame, float, float]:
    """
    a. Build flags dict (all strategy indicators + patterns → True).
    b. Build windows dict from DB-stored indicator parameters.
    c. Call signals_combiner.run_active_signals_with_voting().
    d. Pass resulting signals + price data into BackTest engine.
    Returns (df_ledger, final_balance, total_pnl_pct).
    """
    import numpy as np
    from TradeX.backtest.backtest import BackTest
    from TradeX.signals_combiner import run_active_signals_with_voting

    # ── a. Build flags ──────────────────────────────────────────────────────
    indicators: list[str] = list(getattr(strategy_row, "indicators", None) or [])
    patterns:   list[str] = list(getattr(strategy_row, "patterns",   None) or [])
    all_signals = indicators + patterns

    if not all_signals:
        raise ValueError(
            f"Strategy '{strategy_row.strategy}' has no indicators or patterns configured."
        )

    flags: dict[str, bool] = {name: True for name in all_signals}

    # ── b. Build windows ────────────────────────────────────────────────────
    windows_from_db = _build_windows_from_db(strategy_row, indicators)

    # ── c. Run signals combiner ─────────────────────────────────────────────
    open_  = df_price["open"].to_numpy(dtype=np.float64)
    high   = df_price["high"].to_numpy(dtype=np.float64)
    low    = df_price["low"].to_numpy(dtype=np.float64)
    close_ = df_price["close"].to_numpy(dtype=np.float64)
    volume = df_price["volume"].to_numpy(dtype=np.float64)
    timestamps = df_price["datetime"].values

    df_signals, _returned_windows = run_active_signals_with_voting(
        flags=flags,
        open_=open_,
        high=high,
        low=low,
        close_=close_,
        volume=volume,
        timestamps=timestamps,
    )

    if df_signals.empty:
        raise ValueError("Signal combiner returned no signals for the selected date range.")

    # ── d. BackTest engine ──────────────────────────────────────────────────
    bt = BackTest(
        df_price=df_price,
        df_predictions=df_signals,
        starting_balance=req.starting_balance,
        take_profit=req.take_profit,
        stop_loss=req.stop_loss,
        buy_after_minutes=req.buy_after_minutes,
        fee=req.fee,
        leverage=req.leverage,
        slippage=req.slippage,
    )
    return bt.run()


# ===========================================================================
# 4. Next run index for a strategy
# ===========================================================================

async def _next_run_index(db: AsyncSession, strategy_name: str) -> int:
    """Count existing runs for this strategy and return next index (1-based)."""
    try:
        stmt = select(func.count()).select_from(RUN_REGISTRY_TABLE).where(
            RUN_REGISTRY_TABLE.c.strategy_name == strategy_name
        )
        count = (await db.execute(stmt)).scalar() or 0
        return count + 1
    except Exception:
        # run_registry table may not exist yet – will be created on first insert
        return 1


# ===========================================================================
# 5. Persist ledger + registry row
# ===========================================================================

async def _persist_run(
    db: AsyncSession,
    df_ledger: pd.DataFrame,
    req: BacktestRunRequest,
    strategy_name: str,
    run_index: int,
    final_balance: float,
    total_pnl_pct: float,
    total_trades: int,
    win_rate: float,
) -> str:
    """
    Create backtest_runs schema + tables if needed, insert ledger rows and a
    run_registry record.  Returns the generated table_name.
    """
    table_name = f"{strategy_name}_run_{run_index}"

    # Ensure schema exists
    await db.execute(text("CREATE SCHEMA IF NOT EXISTS backtest_runs"))

    # Create run_registry if absent
    await db.execute(text("""
        CREATE TABLE IF NOT EXISTS backtest_runs.run_registry (
            id            SERIAL PRIMARY KEY,
            table_name    TEXT NOT NULL UNIQUE,
            strategy_name TEXT NOT NULL,
            exchange      TEXT NOT NULL,
            start_date    TEXT,
            end_date      TEXT,
            take_profit   DOUBLE PRECISION NOT NULL,
            stop_loss     DOUBLE PRECISION NOT NULL,
            total_trades  INTEGER NOT NULL,
            win_rate      DOUBLE PRECISION NOT NULL,
            total_pnl_pct DOUBLE PRECISION NOT NULL,
            final_balance DOUBLE PRECISION NOT NULL,
            created_at    TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """))

    # Create the individual ledger table
    await db.execute(text(f"""
        CREATE TABLE IF NOT EXISTS backtest_runs."{table_name}" (
            id                  SERIAL PRIMARY KEY,
            datetime            TIMESTAMP NOT NULL,
            action              TEXT NOT NULL,
            buy_price           DOUBLE PRECISION,
            sell_price          DOUBLE PRECISION,
            pnl                 DOUBLE PRECISION,
            pnl_sum             DOUBLE PRECISION,
            balance             DOUBLE PRECISION NOT NULL,
            predicted_direction TEXT NOT NULL
        )
    """))

    # Insert ledger rows in bulk
    if not df_ledger.empty:
        rows_to_insert = []
        for _, row in df_ledger.iterrows():
            rows_to_insert.append({
                "datetime":            str(row["datetime"]),
                "action":              str(row["action"]),
                "buy_price":           float(row["buy_price"])  if pd.notna(row.get("buy_price"))  else None,
                "sell_price":          float(row["sell_price"]) if pd.notna(row.get("sell_price")) else None,
                "pnl":                 float(row["pnl"])        if pd.notna(row.get("pnl"))        else None,
                "pnl_sum":             float(row["pnl_sum"])    if "pnl_sum" in row.index and pd.notna(row["pnl_sum"]) else None,
                "balance":             float(row["balance"]),
                "predicted_direction": str(row["predicted_direction"]),
            })

        if rows_to_insert:
            ledger_tbl = get_ledger_run_table(table_name)
            await db.execute(ledger_tbl.insert(), rows_to_insert)

    # Insert run_registry row
    await db.execute(
        RUN_REGISTRY_TABLE.insert().values(
            table_name=table_name,
            strategy_name=strategy_name,
            exchange=req.exchange,
            start_date=req.start_date,
            end_date=req.end_date,
            take_profit=req.take_profit,
            stop_loss=req.stop_loss,
            total_trades=total_trades,
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            final_balance=final_balance,
            created_at=datetime.utcnow(),
        )
    )

    await db.commit()
    return table_name


# ===========================================================================
# 6. Update strategy_registry with latest run stats
# ===========================================================================

async def _update_strategy_stats(
    db: AsyncSession,
    strategy_name: str,
    total_pnl_pct: float,
    take_profit: float,
    stop_loss: float,
) -> None:
    """Persist last_pnl_pct, last_run_tp, last_run_sl back to strategy_registry."""
    try:
        # Ensure columns exist (idempotent)
        for col_def in [
            "ADD COLUMN IF NOT EXISTS last_pnl_pct  DOUBLE PRECISION",
            "ADD COLUMN IF NOT EXISTS last_run_tp   DOUBLE PRECISION",
            "ADD COLUMN IF NOT EXISTS last_run_sl   DOUBLE PRECISION",
        ]:
            await db.execute(text(f"ALTER TABLE strategy_registry {col_def}"))

        await db.execute(
            text("""
                UPDATE strategy_registry
                   SET last_pnl_pct = :pnl,
                       last_run_tp  = :tp,
                       last_run_sl  = :sl
                 WHERE strategy = :name
            """),
            {"pnl": total_pnl_pct, "tp": take_profit, "sl": stop_loss, "name": strategy_name},
        )
        await db.commit()
    except Exception:
        await db.rollback()   # non-fatal – stats update is best-effort


# ===========================================================================
# 7. Post-process ledger → BacktestResponse
# ===========================================================================

def _build_response(
    df_ledger: pd.DataFrame,
    final_balance: float,
    total_pnl_pct: float,
    req: BacktestRunRequest,
    symbol: str,
    run_table_name: Optional[str] = None,
) -> BacktestResponse:

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
    win_rate  = round(win_trades / total_trades * 100, 2) if total_trades else 0.0
    loss_rate = round(100 - win_rate, 2)
    max_w, max_l = _streaks(win_mask)

    win_loss_data = [
        WinLossPoint(name="Trades Won",  value=win_trades),
        WinLossPoint(name="Trades Lost", value=loss_trades),
    ]
    pnl_data = [
        PnLPoint(trade=i + 1, pnl=round(float(pnl), 2))
        for i, pnl in enumerate(sell_rows["pnl"].tolist())
    ]

    summary = BacktestSummary(
        strategy_name=req.strategy_name,
        exchange=req.exchange,
        symbol=symbol,
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
        run_table_name=run_table_name,
    )

    return BacktestResponse(
        summary=summary,
        ledger=ledger_entries,
        win_loss_data=win_loss_data,
        pnl_data=pnl_data,
    )


# ===========================================================================
# 8. Public entry-point: run backtest
# ===========================================================================

async def run_backtest(
    db: AsyncSession,
    req: BacktestRunRequest,
) -> BacktestResponse:
    """Full pipeline – see module docstring."""

    # ── Strategy metadata ─────────────────────────────────────────────────
    strategy_row: Optional[StrategyRegistry] = (
        await db.execute(
            select(StrategyRegistry).where(
                StrategyRegistry.strategy == req.strategy_name
            )
        )
    ).scalars().first()

    if strategy_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy '{req.strategy_name}' not found in strategy_registry.",
        )

    symbol: str = (
        strategy_row.symbol or extract_symbol_from_strategy(req.strategy_name)
    ).lower()

    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot determine coin symbol for strategy '{req.strategy_name}'.",
        )

    # Use DB defaults for tp/sl only when request carries the model defaults (1.0/1.0)
    # and the DB has explicit values stored.
    if strategy_row.tp and req.take_profit == 1.0:
        req.take_profit = float(strategy_row.tp)
    if strategy_row.sl and req.stop_loss == 1.0:
        req.stop_loss = float(strategy_row.sl)

    # ── Load price data via targeted SQL query (no full-table scan) ──────
    try:
        df_price = await asyncio.to_thread(
            _load_price_df_sync, req.exchange, symbol, req.start_date, req.end_date
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # ── Run engine (signals combiner + BackTest) in thread-pool ──────────
    try:
        df_ledger, final_balance, total_pnl_pct = await asyncio.to_thread(
            _run_engine_with_combiner, df_price, strategy_row, req
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BackTest engine failed: {exc}",
        )

    if df_ledger.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Backtest completed but produced no trades.",
        )

    # Compute summary stats needed for saving
    sell_rows = df_ledger[df_ledger["action"].str.startswith("sell")]
    total_trades = len(sell_rows)
    win_trades = int((sell_rows["pnl"] > 0).sum())
    win_rate = round(win_trades / total_trades * 100, 2) if total_trades else 0.0

    # ── Persist ledger & update stats ────────────────────────────────────
    run_index = await _next_run_index(db, req.strategy_name)

    run_table_name = await _persist_run(
        db=db,
        df_ledger=df_ledger,
        req=req,
        strategy_name=req.strategy_name,
        run_index=run_index,
        final_balance=final_balance,
        total_pnl_pct=total_pnl_pct,
        total_trades=total_trades,
        win_rate=win_rate,
    )

    await _update_strategy_stats(
        db=db,
        strategy_name=req.strategy_name,
        total_pnl_pct=total_pnl_pct,
        take_profit=req.take_profit,
        stop_loss=req.stop_loss,
    )

    return _build_response(df_ledger, final_balance, total_pnl_pct, req, symbol, run_table_name)


# ===========================================================================
# 9. List saved runs for a strategy
# ===========================================================================

async def get_strategy_runs(
    db: AsyncSession,
    strategy_name: str,
) -> list[LedgerRunMeta]:
    try:
        stmt = (
            select(RUN_REGISTRY_TABLE)
            .where(RUN_REGISTRY_TABLE.c.strategy_name == strategy_name)
            .order_by(RUN_REGISTRY_TABLE.c.id.desc())
        )
        rows = (await db.execute(stmt)).fetchall()
    except Exception:
        return []

    return [
        LedgerRunMeta(
            run_id=row.id,
            table_name=row.table_name,
            strategy_name=row.strategy_name,
            exchange=row.exchange,
            start_date=row.start_date,
            end_date=row.end_date,
            take_profit=row.take_profit,
            stop_loss=row.stop_loss,
            total_trades=row.total_trades,
            win_rate=row.win_rate,
            total_pnl_pct=row.total_pnl_pct,
            final_balance=row.final_balance,
            created_at=str(row.created_at),
        )
        for row in rows
    ]


# ===========================================================================
# 10. Paginated ledger for a specific run
# ===========================================================================

async def get_run_ledger(
    db: AsyncSession,
    strategy_name: str,
    run_id: int,
    page: int = 1,
    page_size: int = 50,
) -> PaginatedLedger:
    # Fetch run meta
    stmt = select(RUN_REGISTRY_TABLE).where(
        RUN_REGISTRY_TABLE.c.id == run_id,
        RUN_REGISTRY_TABLE.c.strategy_name == strategy_name,
    )
    row = (await db.execute(stmt)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found for strategy '{strategy_name}'.")

    run_meta = LedgerRunMeta(
        run_id=row.id,
        table_name=row.table_name,
        strategy_name=row.strategy_name,
        exchange=row.exchange,
        start_date=row.start_date,
        end_date=row.end_date,
        take_profit=row.take_profit,
        stop_loss=row.stop_loss,
        total_trades=row.total_trades,
        win_rate=row.win_rate,
        total_pnl_pct=row.total_pnl_pct,
        final_balance=row.final_balance,
        created_at=str(row.created_at),
    )

    # Count total
    count_stmt = text(
        f'SELECT COUNT(*) FROM backtest_runs."{row.table_name}"'
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    # Fetch page
    offset = (page - 1) * page_size
    page_stmt = text(
        f'SELECT * FROM backtest_runs."{row.table_name}" '
        f'ORDER BY datetime ASC LIMIT :lim OFFSET :off'
    )
    ledger_rows = (await db.execute(page_stmt, {"lim": page_size, "off": offset})).fetchall()

    entries: list[LedgerEntry] = []
    for r in ledger_rows:
        action = str(r.action)
        is_buy = action == "buy"
        reason = None
        if not is_buy:
            parts = action.split(" - ", 1)
            reason = parts[1] if len(parts) == 2 else action

        entries.append(LedgerEntry(
            date=str(r.datetime),
            type="Buy" if is_buy else "Sell",
            price=float(r.buy_price if is_buy else r.sell_price) if (r.buy_price or r.sell_price) else 0.0,
            pnl=float(r.pnl) if r.pnl is not None else None,
            pnl_sum=float(r.pnl_sum) if r.pnl_sum is not None else None,
            balance=float(r.balance),
            direction=str(r.predicted_direction),
            reason=reason,
        ))

    return PaginatedLedger(
        run_meta=run_meta,
        entries=entries,
        page=page,
        page_size=page_size,
        total=total,
        pages=max(1, math.ceil(total / page_size)),
    )