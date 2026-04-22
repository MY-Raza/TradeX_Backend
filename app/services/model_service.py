from __future__ import annotations

import math
from typing import Optional, Union

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_model import DLResult, MLResult
from app.schemas.model_schema import (
    ModelResultDetail,
    ModelResultListItem,
    ModelTypeOptions,
    PaginatedModelResults,
)

# ---------------------------------------------------------------------------
# Type alias – either ORM model class
# ---------------------------------------------------------------------------

_ModelClass = Union[type[MLResult], type[DLResult]]


# ===========================================================================
# Private helpers
# ===========================================================================

def _resolve_model_class(model_type: str) -> _ModelClass:
    """Return the ORM class that corresponds to *model_type* ('ml' | 'dl')."""
    if model_type.lower() == "dl":
        return DLResult
    return MLResult


def _to_list_item(row: Union[MLResult, DLResult]) -> ModelResultListItem:
    return ModelResultListItem(
        model_name=row.model_name,
        pnl=row.pnl,
        total_trades=row.total_trades,
        long_trades=row.long_trades,
        short_trades=row.short_trades,
        win_trades=row.win_trades,
        loss_trades=row.loss_trades,
        win_rate=row.win_rate,
        loss_rate=row.loss_rate,
        max_drawdown=row.max_drawdown,
        max_drawdown_pct=row.max_drawdown_pct,
        max_consecutive_wins=row.max_consecutive_wins,
        max_consecutive_losses=row.max_consecutive_losses,
    )


def _to_detail(row: Union[MLResult, DLResult]) -> ModelResultDetail:
    return ModelResultDetail(
        model_name=row.model_name,
        pnl=row.pnl,
        total_trades=row.total_trades,
        long_trades=row.long_trades,
        short_trades=row.short_trades,
        win_trades=row.win_trades,
        loss_trades=row.loss_trades,
        breakeven_trades=row.breakeven_trades,
        win_rate=row.win_rate,
        loss_rate=row.loss_rate,
        gross_profit=row.gross_profit,
        gross_loss=row.gross_loss,
        net_profit=row.net_profit,
        avg_trade_pnl=row.avg_trade_pnl,
        avg_win=row.avg_win,
        avg_loss=row.avg_loss,
        risk_reward_ratio=row.risk_reward_ratio,
        profit_factor=row.profit_factor,
        max_drawdown=row.max_drawdown,
        max_drawdown_pct=row.max_drawdown_pct,
        sharpe_ratio=row.sharpe_ratio,
        sortino_ratio=row.sortino_ratio,
        max_consecutive_wins=row.max_consecutive_wins,
        max_consecutive_losses=row.max_consecutive_losses,
    )


# ===========================================================================
# Public service API
# ===========================================================================

async def get_model_results(
    db: AsyncSession,
    model_type: str,
    *,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedModelResults:
    """
    Return a paginated list of model results for the given *model_type*.

    Parameters
    ----------
    model_type  : 'ml' or 'dl'  (case-insensitive)
    search      : partial case-insensitive match on model_name
    page        : 1-based page number
    page_size   : rows per page (max 100)
    """
    Model = _resolve_model_class(model_type)
    stmt = select(Model)

    if search:
        stmt = stmt.where(Model.model_name.ilike(f"%{search.strip()}%"))

    # ── Total count ───────────────────────────────────────────────────────
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    # ── Pagination ────────────────────────────────────────────────────────
    pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, pages))
    offset = (page - 1) * page_size

    stmt = (
        stmt
        .order_by(Model.model_name)
        .limit(page_size)
        .offset(offset)
    )

    rows = (await db.execute(stmt)).scalars().all()

    return PaginatedModelResults(
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
        model_type=model_type.lower(),   # type: ignore[arg-type]
        results=[_to_list_item(r) for r in rows],
    )


async def get_model_result_by_name(
    db: AsyncSession,
    model_type: str,
    model_name: str,
) -> Optional[ModelResultDetail]:
    """
    Return the full detail record for a single model run.
    Returns None if not found (the route raises 404).
    """
    Model = _resolve_model_class(model_type)
    stmt = select(Model).where(Model.model_name == model_name)
    row = (await db.execute(stmt)).scalars().first()
    return _to_detail(row) if row else None


async def get_model_type_options() -> ModelTypeOptions:
    """
    Return the available model type identifiers.
    Static – no DB round-trip needed.
    """
    return ModelTypeOptions(types=["ml", "dl"])