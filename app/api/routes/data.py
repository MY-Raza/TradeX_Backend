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
    description=(
        "Returns all supported exchange identifiers and their display labels. "
        "Use the `id` field as the `exchange` value in subsequent requests."
    ),
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
    description=(
        "Returns all tradable coins available for the given exchange. "
        "Use `symbol` as the `symbol` value in fetch and OHLCV requests."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unknown exchange id supplied.",
        }
    },
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
    description=(
        "Returns the most recent stored candle datetime for the given exchange "
        "and symbol. If no data exists yet, `last_date` is null. "
        "The frontend uses this to pre-fill the start_date field and to "
        "decide whether to lock the date picker."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unknown exchange id supplied.",
        }
    },
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
    description=(
        "Runs the exchange-specific data fetcher for the selected coin, "
        "saves new 1-minute OHLCV candles to the database, "
        "and returns a summary of rows written. "
        "This is the endpoint triggered by the **Fetch Data** button."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unknown exchange or symbol.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Fetcher encountered an error.",
        },
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
        "Reads saved candles from the database for the selected exchange, "
        "symbol, and timeframe. Resamples 1-minute base data to the requested "
        "timeframe and returns chart-ready candles with summary stats. "
        "Call this after a successful `/data/fetch` request."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unknown exchange or timeframe.",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "No data found — fetch data first.",
        },
    },
)
async def get_ohlcv(
    db: DB,
    exchange: Annotated[
        str,
        Query(description="Exchange id, e.g. 'binance'"),
    ],
    symbol: Annotated[
        str,
        Query(description="Coin symbol key, e.g. 'btc'"),
    ],
    timeframe: Annotated[
        str,
        Query(
            description="Candle timeframe: 1m | 5m | 15m | 1h | 4h | 1d",
            examples={"1h": {"value": "1h"}},
        ),
    ] = "1h",
) -> OHLCVResponse:
    return await data_service.read_ohlcv(db, exchange, symbol, timeframe)