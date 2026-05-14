"""
TradeX – Strategy Generator Pydantic Schemas

Endpoints served
----------------
POST /strategy-generator/create   → CreateStrategyResponse
GET  /strategy-generator/exchanges → list[ExchangeInfo]  (reuses data_schema)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.backtest_schema import LedgerEntry, BacktestSummary, WinLossPoint, PnLPoint


# ===========================================================================
# Create Strategy Request
# ===========================================================================

class CreateStrategyRequest(BaseModel):
    name: str = Field(
        ...,
        description="Human-readable strategy name (used only for display; actual ID is auto-generated)",
        min_length=1,
        max_length=100,
    )
    timeframe: str = Field(
        ...,
        description="Candle timeframe: '1h' | '15m' | '5m'",
    )
    exchange: str = Field(
        ...,
        description="Exchange id: 'binance' | 'bybit' | 'kraken' | 'metatrader5'",
    )
    symbol: str = Field(
        ...,
        description="Coin symbol key, e.g. 'btc'",
    )
    start_date: Optional[str] = Field(
        None,
        description="Start datetime for backtest (ISO date or datetime). Fetches all data if omitted.",
    )
    end_date: Optional[str] = Field(
        None,
        description="End datetime for backtest (ISO date or datetime). Fetches all data if omitted.",
    )
    starting_balance: float = Field(default=1000.0, gt=0)
    take_profit: float = Field(default=3.0, gt=0, description="TP as a percentage")
    stop_loss: float = Field(default=1.0, gt=0, description="SL as a percentage")
    fee: float = Field(default=0.05, ge=0)
    leverage: float = Field(default=1.0, gt=0)
    slippage: float = Field(default=0.0, ge=0)


# ===========================================================================
# Create Strategy Response
# ===========================================================================

class CreateStrategyResponse(BaseModel):
    strategy_id: str = Field(..., description="Auto-generated strategy ID, e.g. sig_1h_btc_42")
    display_name: str = Field(..., description="Human-readable name provided by user")
    timeframe: str
    symbol: str
    exchange: str
    summary: BacktestSummary
    ledger: list[LedgerEntry]
    win_loss_data: list[WinLossPoint]
    pnl_data: list[PnLPoint]
    message: str = Field(..., description="Success message with saved table info")