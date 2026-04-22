"""
TradeX – Backtest Router

Endpoints
---------
GET  /backtest/strategies    All strategy names for the dropdown
POST /backtest/run           Run the backtest engine and return results
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.backtest_schema import (
    BacktestResponse,
    BacktestRunRequest,
    BacktestStrategyOption,
)
from app.services import backtest_service

router = APIRouter(prefix="/backtest", tags=["Backtest"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /backtest/strategies
# ===========================================================================

@router.get(
    "/strategies",
    response_model=list[BacktestStrategyOption],
    summary="List strategies for dropdown",
    description=(
        "Returns every strategy in strategy_registry with its symbol, "
        "time_horizon, and default TP/SL. Use `name` as `strategy_name` "
        "in the run request."
    ),
)
async def list_strategies(db: DB) -> list[BacktestStrategyOption]:
    return await backtest_service.get_backtest_strategies(db)


# ===========================================================================
# POST /backtest/run
# ===========================================================================

@router.post(
    "/run",
    response_model=BacktestResponse,
    summary="Run backtest",
    description=(
        "Runs the BackTest engine for the selected strategy and exchange. "
        "The coin is resolved automatically from the strategy's `symbol` field "
        "(or extracted from the strategy name as fallback). "
        "Price data is loaded from the exchange DB schema (`data_binance`, etc.) "
        "and predictions from `strategies.<strategy_name>`. "
        "Returns ledger rows, win/loss chart data, PnL-per-trade chart data, "
        "and summary stats."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Strategy, price data, or predictions not found.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Strategy has no coin symbol or backtest produced no trades.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "BackTest engine raised an exception.",
        },
    },
)
async def run_backtest(
    req: BacktestRunRequest,
    db: DB,
) -> BacktestResponse:
    return await backtest_service.run_backtest(db, req)