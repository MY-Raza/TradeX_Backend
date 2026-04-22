"""
TradeX – Model Result Pydantic Schemas

Schema hierarchy
----------------
ModelResultBase          common identity + core fields
  └─ ModelResultListItem     lightweight card for the table / grid view
       └─ ModelResultDetail      full record with all metrics

PaginatedModelResults    envelope used by GET /models/{model_type}
ModelTypeOptions         returned by GET /models/types (dropdown data)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# Shared base
# ===========================================================================

class ModelResultBase(BaseModel):
    model_name: str = Field(
        ...,
        description="Unique model run identifier, e.g. random_forest_clf_20260316_120233",
    )
    pnl: Optional[float] = Field(None, description="Net PnL of the back-test run")
    total_trades: Optional[int] = None
    long_trades: Optional[int] = None
    short_trades: Optional[int] = None
    win_trades: Optional[int] = None
    loss_trades: Optional[int] = None
    win_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_consecutive_wins: Optional[int] = None
    max_consecutive_losses: Optional[int] = None


# ===========================================================================
# List item  – shown in the models grid  (GET /models/{model_type})
# ===========================================================================

class ModelResultListItem(ModelResultBase):
    """Condensed view for card/table rendering."""

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Detail view  – full record  (GET /models/{model_type}/{model_name})
# ===========================================================================

class ModelResultDetail(ModelResultBase):
    """Full record including all financial and risk metrics."""

    breakeven_trades: Optional[int] = None
    gross_profit: Optional[float] = None
    gross_loss: Optional[float] = None
    net_profit: Optional[float] = None
    avg_trade_pnl: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Paginated envelope  (used by GET /models/{model_type})
# ===========================================================================

class PaginatedModelResults(BaseModel):
    total: int = Field(..., description="Total matching rows")
    page: int = Field(..., description="Current page (1-based)")
    page_size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total pages")
    model_type: Literal["ml", "dl"] = Field(
        ..., description="Which result table this page came from"
    )
    results: list[ModelResultListItem]


# ===========================================================================
# Model type options  (used by GET /models/types)
# ===========================================================================

class ModelTypeOptions(BaseModel):
    types: list[Literal["ml", "dl"]] = Field(
        default=["ml", "dl"],
        description="Available model type identifiers",
    )