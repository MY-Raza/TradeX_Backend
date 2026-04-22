"""
TradeX – Backtest Service

Pipeline
--------
1. GET /backtest/strategies
   - Read all rows from strategy_registry
   - Extract coin symbol from each strategy name
   - Return dropdown-ready list

2. POST /backtest/run
   a. Fetch strategy metadata (symbol, tp, sl) from strategy_registry
   b. Resolve exchange DB schema → load price DataFrame from <schema>.<symbol>_1m
   c. Load predictions DataFrame from strategies.<strategy_name>
   d. Run BackTest engine in a thread-pool worker (non-blocking)
   e. Post-process ledger → LedgerEntry, WinLossPoint, PnLPoint
   f. Return BacktestResponse
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy_model import StrategyRegistry
from app.models.backtest_model import (
    KNOWN_SYMBOLS,
    extract_symbol_from_strategy,
    get_price_table,
    get_predictions_table,
)
from app.models.data_model import EXCHANGE_SCHEMA_MAP
from app.schemas.backtest_schema import (
    BacktestResponse,
    BacktestRunRequest,
    BacktestStrategyOption,
    BacktestSummary,
    LedgerEntry,
    PnLPoint,
    WinLossPoint,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so BackTest import resolves
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
    """Return (max_consecutive_wins, max_consecutive_losses)."""
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
# 1. Strategy dropdown
# ===========================================================================

async def get_backtest_strategies(db: AsyncSession) -> list[BacktestStrategyOption]:
    """Return all strategies from strategy_registry for the dropdown."""
    stmt = select(
        StrategyRegistry.strategy,
        StrategyRegistry.symbol,
        StrategyRegistry.timehorizon,
        StrategyRegistry.tp,
        StrategyRegistry.sl,
    ).order_by(StrategyRegistry.strategy)

    rows = (await db.execute(stmt)).all()
    return [
        BacktestStrategyOption(
            name=row.strategy,
            symbol=row.symbol,          # already stored in DB
            time_horizon=row.timehorizon,
            tp=row.tp,
            sl=row.sl,
        )
        for row in rows
    ]


# ===========================================================================
# 2. Load DataFrames from DB (async, called from the router)
# ===========================================================================

async def _load_price_df(
    db: AsyncSession,
    exchange: str,
    symbol: str,
) -> pd.DataFrame:
    """Read all 1-minute candles for symbol from the exchange schema."""
    schema = EXCHANGE_SCHEMA_MAP.get(exchange.lower())
    if not schema:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown exchange '{exchange}'. "
                   f"Valid: {list(EXCHANGE_SCHEMA_MAP.keys())}",
        )

    tbl = get_price_table(schema, symbol)
    try:
        result = await db.execute(select(tbl).order_by(tbl.c.datetime))
        rows = result.fetchall()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price table '{schema}.{symbol}_1m' not found. "
                   f"Fetch data first from the Data tab. Detail: {exc}",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price data found for {symbol} on {exchange}. "
                   "Fetch data first from the Data tab.",
        )

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


async def _load_predictions_df(
    db: AsyncSession,
    strategy_name: str,
) -> pd.DataFrame:
    """Read predictions from strategies.<strategy_name>."""
    tbl = get_predictions_table(strategy_name)
    try:
        result = await db.execute(select(tbl).order_by(tbl.c.datetime))
        rows = result.fetchall()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Predictions table 'strategy_signals.{strategy_name}' not found. "
                   f"Detail: {exc}",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for strategy '{strategy_name}'.",
        )

    df = pd.DataFrame(rows, columns=["datetime", "signals"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df


# ===========================================================================
# 3. Run BackTest engine (synchronous – called inside asyncio.to_thread)
# ===========================================================================

def _run_engine(
    df_price: pd.DataFrame,
    df_predictions: pd.DataFrame,
    req: BacktestRunRequest,
) -> tuple[pd.DataFrame, float, float]:
    """Instantiate and run BackTest. Returns (df_ledger, final_balance, total_pnl_pct)."""
    from TradeX.backtest.backtest import BackTest   # lazy import inside thread

    bt = BackTest(
        df_price=df_price,
        df_predictions=df_predictions,
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
# 4. Post-process ledger → response schemas
# ===========================================================================

def _build_response(
    df_ledger: pd.DataFrame,
    final_balance: float,
    total_pnl_pct: float,
    req: BacktestRunRequest,
    symbol: str,
) -> BacktestResponse:

    # ── Ledger entries ────────────────────────────────────────────────────
    ledger_entries: list[LedgerEntry] = []
    for _, row in df_ledger.iterrows():
        action: str = str(row["action"])
        is_buy = action == "buy"
        reason = None
        if not is_buy:
            # action is like "sell - take_profit"
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

    # ── Stats from sell rows only ─────────────────────────────────────────
    sell_rows = df_ledger[df_ledger["action"].str.startswith("sell")]
    total_trades = len(sell_rows)
    win_mask = (sell_rows["pnl"] > 0).tolist()
    win_trades = sum(win_mask)
    loss_trades = total_trades - win_trades
    win_rate  = round(win_trades / total_trades * 100, 2) if total_trades else 0.0
    loss_rate = round(100 - win_rate, 2)
    max_w, max_l = _streaks(win_mask)

    # ── Win/loss bar chart ────────────────────────────────────────────────
    win_loss_data = [
        WinLossPoint(name="Trades Won",  value=win_trades),
        WinLossPoint(name="Trades Lost", value=loss_trades),
    ]

    # ── PnL-per-trade line chart (sell rows only, numbered 1…N) ──────────
    pnl_data = [
        PnLPoint(trade=i + 1, pnl=round(float(pnl), 2))
        for i, pnl in enumerate(sell_rows["pnl"].tolist())
    ]

    # ── Summary ───────────────────────────────────────────────────────────
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
    )

    return BacktestResponse(
        summary=summary,
        ledger=ledger_entries,
        win_loss_data=win_loss_data,
        pnl_data=pnl_data,
    )


# ===========================================================================
# 5. Public entry-point called by the router
# ===========================================================================

async def run_backtest(
    db: AsyncSession,
    req: BacktestRunRequest,
) -> BacktestResponse:
    """
    Full pipeline:
      1. Fetch strategy metadata → get symbol, tp, sl
      2. Load price data from exchange DB schema
      3. Load predictions from strategies schema
      4. Run BackTest engine in thread-pool (non-blocking)
      5. Build and return BacktestResponse
    """

    # ── 1. Strategy metadata ──────────────────────────────────────────────
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

    # symbol is stored directly in the DB; also try name extraction as fallback
    symbol: str = (
        strategy_row.symbol
        or extract_symbol_from_strategy(req.strategy_name)
    ).lower()

    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot determine coin symbol for strategy '{req.strategy_name}'. "
                   "Ensure the strategy name contains a known coin (btc, eth, bnb…).",
        )

    # Override tp/sl from DB if not supplied in request defaults
    effective_tp = float(strategy_row.tp) if strategy_row.tp else req.take_profit
    effective_sl = float(strategy_row.sl) if strategy_row.sl else req.stop_loss
    req.take_profit = effective_tp
    req.stop_loss   = effective_sl

    # ── 2 & 3. Load DataFrames ────────────────────────────────────────────
    df_price, df_predictions = await asyncio.gather(
        _load_price_df(db, req.exchange, symbol),
        _load_predictions_df(db, req.strategy_name),
    )

    # ── 4. Run engine in thread-pool ──────────────────────────────────────
    try:
        df_ledger, final_balance, total_pnl_pct = await asyncio.to_thread(
            _run_engine, df_price, df_predictions, req
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BackTest engine failed: {exc}",
        )

    if df_ledger.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Backtest completed but produced no trades. "
                   "Check that predictions contain non-zero signals.",
        )

    # ── 5. Build response ─────────────────────────────────────────────────
    return _build_response(df_ledger, final_balance, total_pnl_pct, req, symbol)