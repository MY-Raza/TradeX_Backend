"""
TradeX – Strategy Generator Router

Endpoints
---------
POST /strategy-generator/create   → CreateStrategyResponse
GET  /strategy-generator/coins/{exchange}  → list[CoinInfo]  (reuses data_service)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.strategy_generator_schema import (
    CreateStrategyRequest,
    CreateStrategyResponse,
)
from app.schemas.data_schema import CoinInfo
from app.services import strategy_generator_service, data_service

router = APIRouter(prefix="/strategy-generator", tags=["Strategy Generator"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /strategy-generator/coins/{exchange}
# ===========================================================================

@router.get(
    "/coins/{exchange}",
    response_model=list[CoinInfo],
    summary="List coins available on an exchange (for strategy creation form)",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unknown exchange id."},
    },
)
async def list_coins(exchange: str) -> list[CoinInfo]:
    """Reuse the data_service coin list so the form stays in sync with available data."""
    return data_service.get_coins(exchange)


# ===========================================================================
# POST /strategy-generator/create
# ===========================================================================

@router.post(
    "/create",
    response_model=CreateStrategyResponse,
    summary="Create a new trading strategy",
    description=(
        "Generates a randomised set of indicator/pattern flags for the selected "
        "timeframe and symbol, runs the signals combiner with majority voting, "
        "saves the resulting signal table to `strategy_signals.<strategy_id>`, "
        "runs a BackTest on the historical price data, saves strategy metadata "
        "to `strategies.strategy_registry`, and returns the full backtest ledger "
        "and summary statistics."
    ),
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Invalid timeframe / exchange / symbol, or no data / trades produced.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Strategy generation or backtest engine raised an exception.",
        },
    },
)
async def create_strategy(
    req: CreateStrategyRequest,
    db: DB,
) -> CreateStrategyResponse:
    return await strategy_generator_service.create_strategy(db, req)