"""
TradeX – Data Tab Pydantic Schemas

Endpoints served
----------------
GET  /data/exchanges              → list[ExchangeInfo]
GET  /data/coins/{exchange}       → list[CoinInfo]
POST /data/fetch                  → FetchResponse   (triggers fetcher, saves to DB)
GET  /data/ohlcv                  → OHLCVResponse   (reads saved data back for chart)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Exchange catalogue
# ===========================================================================

class ExchangeInfo(BaseModel):
    id: str = Field(..., description="Internal key used in API calls, e.g. 'binance'")
    label: str = Field(..., description="Display label shown in the dropdown, e.g. 'Binance'")


# ===========================================================================
# Coin / symbol
# ===========================================================================

class CoinInfo(BaseModel):
    symbol: str = Field(..., description="Raw symbol key, e.g. 'btc'")
    label: str = Field(..., description="Display label, e.g. 'BTC/USDT'")


# ===========================================================================
# Fetch request  (POST /data/fetch)
# ===========================================================================

class FetchRequest(BaseModel):
    exchange: str = Field(
        ...,
        description="Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'",
    )
    symbol: str = Field(
        ...,
        description="Coin symbol key, e.g. 'btc'",
    )
    start_date: str = Field(
        default="2024-01-01",
        description="Start date for the fetch run. Format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
    )
    end_date: str = Field(
        default="now",
        description="End date for the fetch run. 'now' fetches up to current time.",
    )


# ===========================================================================
# Fetch response
# ===========================================================================

class FetchResponse(BaseModel):
    exchange: str
    symbol: str
    rows_saved: int = Field(..., description="Number of new candles written to the DB")
    message: str


# ===========================================================================
# OHLCV candle  (one row for the chart)
# ===========================================================================

class OHLCVCandle(BaseModel):
    time: str = Field(..., description="Formatted timestamp label shown on the X-axis")
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===========================================================================
# OHLCV query params / response  (GET /data/ohlcv)
# ===========================================================================

class OHLCVResponse(BaseModel):
    exchange: str
    symbol: str
    timeframe: str
    candles: list[OHLCVCandle]
    total_rows: int = Field(..., description="Number of candles returned")
    open: float = Field(..., description="Open of the first candle")
    high: float = Field(..., description="Highest high across all candles")
    low: float = Field(..., description="Lowest low across all candles")
    close: float = Field(..., description="Close of the last candle")
    total_volume: float = Field(..., description="Sum of volume across all candles")