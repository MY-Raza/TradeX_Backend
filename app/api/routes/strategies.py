from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.strategy_schema import (
    PaginatedStrategies,
    StrategyDetail,
    StrategyFilterOptions,
)
from app.services import strategy_service

router = APIRouter(prefix="/strategies", tags=["Strategies"])

# Reusable type alias for the DB dependency
DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /strategies/filters   ← MUST come before /{strategy_id}
# ===========================================================================

@router.get(
    "/filters",
    response_model=StrategyFilterOptions,
    summary="Get filter options",
    description=(
        "Returns all distinct **symbol** and **time_horizon** values "
        "available in the database. Use this to populate frontend dropdowns."
    ),
)
async def list_filter_options(db: DB) -> StrategyFilterOptions:
    return await strategy_service.get_filter_options(db)


# ===========================================================================
# GET /strategies
# ===========================================================================

@router.get(
    "",
    response_model=PaginatedStrategies,
    summary="List strategies",
    description=(
        "Returns a paginated list of trading strategies. "
        "All query parameters are optional and combinable."
    ),
)
async def list_strategies(
    db: DB,
    symbol: Annotated[
        Optional[str],
        Query(
            description="Filter by ticker symbol (case-insensitive). "
                        "E.g. `btc`, `eth`, `bnb`",
            examples={"btc": {"value": "btc"}},
        ),
    ] = None,
    time_horizon: Annotated[
        Optional[str],
        Query(
            alias="time_horizon",
            description="Filter by candle timeframe (case-insensitive). "
                        "One of: `1h`, `15m`, `5m`",
            examples={"1h": {"value": "1h"}},
        ),
    ] = None,
    search: Annotated[
        Optional[str],
        Query(description="Partial case-insensitive match on strategy name."),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)."),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=100, alias="page_size", description="Items per page (max 100)."),
    ] = 20,
) -> PaginatedStrategies:
    return await strategy_service.get_strategies(
        db,
        symbol=symbol,
        time_horizon=time_horizon,
        search=search,
        page=page,
        page_size=page_size,
    )


# ===========================================================================
# GET /strategies/{strategy_id}
# ===========================================================================

@router.get(
    "/{strategy_name}",
    response_model=StrategyDetail,
    summary="Get strategy detail",
    description=(
        "Returns the full detail record for a single strategy. "
        "Includes all active indicators with their period parameters, "
        "active candlestick patterns, and risk parameters (TP / SL)."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "No strategy found for the given ID.",
            "content": {
                "application/json": {
                    "example": {"detail": "Strategy with id=999 not found."}
                }
            },
        }
    },
)
async def get_strategy(
    strategy_name: str,
    db: DB,
) -> StrategyDetail:
    strategy = await strategy_service.get_strategy_by_name(db, strategy_name)
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy with id={strategy_name} not found.",
        )
    return strategy