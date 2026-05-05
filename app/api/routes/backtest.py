"""
TradeX – Backtest Router

Endpoints
---------
GET  /backtest/strategies                              All strategy names for the dropdown
POST /backtest/run                                     Run the backtest engine and return results
GET  /backtest/runs/{strategy_name}                    List saved runs for a strategy
GET  /backtest/runs/{strategy_name}/{run_id}/ledger    Paginated ledger for a specific run
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.backtest_schema import (
    BacktestResponse,
    BacktestRunRequest,
    BacktestStrategyOption,
    LedgerRunMeta,
    PaginatedLedger,
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


# ===========================================================================
# GET /backtest/runs/{strategy_name}
# ===========================================================================

@router.get(
    "/runs/{strategy_name}",
    response_model=list[LedgerRunMeta],
    summary="List saved backtest runs for a strategy",
    description=(
        "Returns all previously saved backtest runs for the given strategy, "
        "ordered newest first. Each item contains the run metadata (exchange, "
        "date range, TP/SL, win-rate, PnL). Returns an empty list if no runs "
        "have been saved yet."
    ),
)
async def list_strategy_runs(
    strategy_name: str,
    db: DB,
) -> list[LedgerRunMeta]:
    return await backtest_service.get_strategy_runs(db, strategy_name)


# ===========================================================================
# GET /backtest/runs/{strategy_name}/{run_id}/ledger
# ===========================================================================

@router.get(
    "/runs/{strategy_name}/{run_id}/ledger",
    response_model=PaginatedLedger,
    summary="Get paginated ledger for a saved run",
    description=(
        "Returns a paginated list of trade events (buy/sell) for the given "
        "run_id. Includes the run metadata in `run_meta` and the trade entries "
        "in `entries`. Use `page` and `page_size` query params to paginate."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Run not found for the given strategy_name / run_id.",
        },
    },
)
async def get_run_ledger(
    strategy_name: str,
    run_id: int,
    db: DB,
    page: Annotated[int, Query(ge=1, description="Page number (1-based).")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, alias="page_size")] = 50,
) -> PaginatedLedger:
    return await backtest_service.get_run_ledger(db, strategy_name, run_id, page, page_size)