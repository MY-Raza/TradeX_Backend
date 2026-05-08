"""
TradeX – Data Router

Endpoints
---------
GET  /data/exchanges              List all supported exchanges
GET  /data/coins/{exchange}       List coins available for an exchange
POST /data/fetch                  Trigger the exchange fetcher + save to DB
GET  /data/ohlcv                  Read saved candles back from DB (chart data)
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.data_schema import (
    CoinInfo,
    ExchangeInfo,
    FetchRequest,
    FetchResponse,
    LastDateResponse,
    OHLCVResponse,
)
from app.services import data_service

router = APIRouter(prefix="/data", tags=["Data"])

DB = Annotated[AsyncSession, Depends(get_db)]


# ===========================================================================
# GET /data/exchanges
# ===========================================================================

@router.get(
    "/exchanges",
    response_model=list[ExchangeInfo],
    summary="List supported exchanges",
)
async def list_exchanges() -> list[ExchangeInfo]:
    return data_service.get_exchanges()


# ===========================================================================
# GET /data/coins/{exchange}
# ===========================================================================

@router.get(
    "/coins/{exchange}",
    response_model=list[CoinInfo],
    summary="List coins for an exchange",
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Unknown exchange id."}},
)
async def list_coins(exchange: str) -> list[CoinInfo]:
    return data_service.get_coins(exchange)


# ===========================================================================
# GET /data/last-date
# ===========================================================================

@router.get(
    "/last-date",
    response_model=LastDateResponse,
    summary="Get the last stored candle datetime for a coin",
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Unknown exchange id."}},
)
async def get_last_date(
    db: DB,
    exchange: Annotated[str, Query(description="Exchange id, e.g. 'binance'")],
    symbol:   Annotated[str, Query(description="Coin symbol key, e.g. 'btc'")],
) -> LastDateResponse:
    return await data_service.get_last_date_for_coin(db, exchange, symbol)


# ===========================================================================
# POST /data/fetch
# ===========================================================================

@router.post(
    "/fetch",
    response_model=FetchResponse,
    summary="Fetch and store market data",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unknown exchange or symbol."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Fetcher error."},
    },
)
async def fetch_data(req: FetchRequest) -> FetchResponse:
    return await data_service.fetch_and_store(req)


# ===========================================================================
# GET /data/ohlcv
# ===========================================================================

@router.get(
    "/ohlcv",
    response_model=OHLCVResponse,
    summary="Get OHLCV chart data",
    description=(
        "Reads saved 1-minute candles from the database for the selected exchange "
        "and symbol, filtered by the optional start_date / end_date range. "
        "No resampling is performed — raw 1-minute candles are returned. "
        "When no date range is supplied the most recent 120 candles are returned."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Unknown exchange or timeframe."},
        status.HTTP_404_NOT_FOUND:   {"description": "No data found — fetch data first."},
    },
)
async def get_ohlcv(
    db: DB,
    exchange: Annotated[str, Query(description="Exchange id, e.g. 'binance'")],
    symbol:   Annotated[str, Query(description="Coin symbol key, e.g. 'btc'")],
    timeframe: Annotated[
        str,
        Query(description="Candle timeframe label (display only for 1m on Data tab)"),
    ] = "1m",
    start_date: Annotated[
        Optional[str],
        Query(description="Filter start – ISO date or datetime, e.g. '2024-01-01'"),
    ] = None,
    end_date: Annotated[
        Optional[str],
        Query(description="Filter end – ISO date or datetime, e.g. '2024-03-31'"),
    ] = None,
) -> OHLCVResponse:
    return await data_service.read_ohlcv(
        db,
        exchange,
        symbol,
        timeframe,
        start_date=start_date,
        end_date=end_date,
    )