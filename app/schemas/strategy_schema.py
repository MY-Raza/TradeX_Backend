"""
TradeX – Strategy Pydantic Schemas

Schema hierarchy
----------------
StrategyBase        common identity fields
  └─ StrategyListItem   lightweight card for the table view
       └─ StrategyDetail    full record including periods and risk params

PaginatedStrategies  envelope used by GET /strategies
StrategyFilterOptions  returned by GET /strategies/filters (dropdown data)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# Shared base
# ===========================================================================

class StrategyBase(BaseModel):
    name:         str   = Field(..., description="Strategy identifier, e.g. sig_1h_btc_1")
    symbol:       str   = Field(..., description="Ticker, e.g. btc / eth")
    time_horizon: str   = Field(..., description="Candle timeframe: 1h | 15m | 5m")


# ===========================================================================
# List item  – shown in the strategies table  (GET /strategies)
# ===========================================================================

class StrategyListItem(StrategyBase):
    """
    Condensed view.  `indicators` and `patterns` are the *names* of the
    active flags derived from the 127 boolean columns in the DB row.
    """
    indicators: list[str] = Field(
        default_factory=list,
        description="Active technical indicator names"
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="Active candlestick pattern names"
    )
    pnl_sum: Optional[float] = Field(None, description="Back-test cumulative PnL")

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Detail view  – full record  (GET /strategies/{id})
# ===========================================================================

class StrategyDetail(StrategyListItem):
    """
    Full record.

    `indicator_details` maps each active indicator name to its period(s).
    The value is a dict because some indicators have multiple periods
    (fast / slow / signal), while most have just one.

    Example::
        {
          "macd": {"fast": 12, "slow": 26, "signal": 9},
          "rsi":  {"period": 14},
          "bop":  {}          # no period for this indicator
        }
    """
    tp: Optional[str] = None
    sl: Optional[str] = None

    indicator_details: dict[str, dict[str, Optional[int]]] = Field(
        default_factory=dict,
        description=(
            "Active indicators mapped to their period parameters. "
            "Keys are indicator names; values are dicts of period param → value."
        )
    )

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Paginated envelope  (used by GET /strategies)
# ===========================================================================

class PaginatedStrategies(BaseModel):
    total:     int  = Field(..., description="Total matching rows")
    page:      int  = Field(..., description="Current page (1-based)")
    page_size: int  = Field(..., description="Items per page")
    pages:     int  = Field(..., description="Total pages")
    results:   list[StrategyListItem]


# ===========================================================================
# Filter options  (used by GET /strategies/filters)
# ===========================================================================

class StrategyFilterOptions(BaseModel):
    symbols:      list[str] = Field(..., description="All distinct symbol values")
    time_horizons: list[str] = Field(..., description="All distinct timeframe values")