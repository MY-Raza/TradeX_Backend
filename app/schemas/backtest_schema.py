"""
TradeX – Backtest Pydantic Schemas

Endpoints served
----------------
GET  /backtest/strategies          → list[BacktestStrategyOption]   (dropdown)
POST /backtest/run                 → BacktestResponse               (run engine)
GET  /backtest/runs/{strategy}     → list[LedgerRunMeta]            (saved runs list)
GET  /backtest/runs/{strategy}/{run_id}/ledger → PaginatedLedger    (paginated ledger)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Strategy dropdown item  (GET /backtest/strategies)
# ===========================================================================

class BacktestStrategyOption(BaseModel):
    name:         str           = Field(..., description="Strategy primary key, e.g. sig_1h_btc_1")
    symbol:       str           = Field(..., description="Coin extracted from strategy, e.g. btc")
    time_horizon: str           = Field(..., description="Candle timeframe: 1h | 15m | 5m")
    tp:           Optional[str] = None
    sl:           Optional[str] = None
    # Most-recent run stats (populated from strategy_registry after a run)
    last_pnl_pct: Optional[float] = None
    last_run_tp:  Optional[float] = None
    last_run_sl:  Optional[float] = None


# ===========================================================================
# Run request  (POST /backtest/run)
# ===========================================================================

class BacktestRunRequest(BaseModel):
    strategy_name:     str            = Field(..., description="Strategy primary key from strategy_registry")
    exchange:          str            = Field(..., description="Exchange id: binance | bybit | kraken | metatrader5")
    # Date range – ISO strings like "2024-01-01" or "2024-01-01T00:00:00"
    start_date:        Optional[str]  = Field(None, description="Start datetime (inclusive). Fetches all data if omitted.")
    end_date:          Optional[str]  = Field(None, description="End datetime (inclusive). Fetches all data if omitted.")
    starting_balance:  float          = Field(default=1000.0, gt=0)
    take_profit:       float          = Field(default=1.0,    gt=0,  description="TP as a percentage, e.g. 1 = 1%")
    stop_loss:         float          = Field(default=1.0,    gt=0,  description="SL as a percentage, e.g. 1 = 1%")
    buy_after_minutes: int            = Field(default=0,      ge=0)
    fee:               float          = Field(default=0.05,   ge=0)
    leverage:          float          = Field(default=1.0,    gt=0)
    slippage:          float          = Field(default=0.0,    ge=0)


# ===========================================================================
# Ledger row  – one buy or sell event
# ===========================================================================

class LedgerEntry(BaseModel):
    date:      str             = Field(..., description="ISO timestamp of the trade event")
    type:      str             = Field(..., description="Buy | Sell")
    price:     float           = Field(..., description="Entry price for buys, exit price for sells")
    pnl:       Optional[float] = None
    pnl_sum:   Optional[float] = None
    balance:   float
    direction: str             = Field(..., description="long | short")
    reason:    Optional[str]   = None


# ===========================================================================
# Chart-ready data points
# ===========================================================================

class WinLossPoint(BaseModel):
    name:  str   # "Trades Won" | "Trades Lost"
    value: int

class PnLPoint(BaseModel):
    trade: int
    pnl:   float


# ===========================================================================
# Summary stats
# ===========================================================================

class BacktestSummary(BaseModel):
    strategy_name:          str
    exchange:               str
    symbol:                 str
    starting_balance:       float
    final_balance:          float
    total_pnl_pct:          float
    total_trades:           int
    win_trades:             int
    loss_trades:            int
    win_rate:               float
    loss_rate:              float
    max_consecutive_wins:   int
    max_consecutive_losses: int
    run_table_name:         Optional[str] = None   # e.g. "sig_1h_btc_1_run_3"


# ===========================================================================
# Full response
# ===========================================================================

class BacktestResponse(BaseModel):
    summary:       BacktestSummary
    ledger:        list[LedgerEntry]
    win_loss_data: list[WinLossPoint]
    pnl_data:      list[PnLPoint]


# ===========================================================================
# Saved-run metadata  (GET /backtest/runs/{strategy})
# ===========================================================================

class LedgerRunMeta(BaseModel):
    run_id:        int
    table_name:    str
    strategy_name: str
    exchange:      str
    start_date:    Optional[str] = None
    end_date:      Optional[str] = None
    take_profit:   float
    stop_loss:     float
    total_trades:  int
    win_rate:      float
    total_pnl_pct: float
    final_balance: float
    created_at:    str


# ===========================================================================
# Paginated ledger  (GET /backtest/runs/{strategy}/{run_id}/ledger)
# ===========================================================================

class PaginatedLedger(BaseModel):
    run_meta:  LedgerRunMeta
    entries:   list[LedgerEntry]
    page:      int
    page_size: int
    total:     int
    pages:     int