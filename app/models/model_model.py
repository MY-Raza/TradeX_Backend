from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


# ---------------------------------------------------------------------------
# Shared mixin – all columns are identical in both tables
# ---------------------------------------------------------------------------

class _ModelResultMixin:
    """
    Columns shared by ml_results and dl_results.
    Both tables live in the `model_stats` schema.
    """

    # ── Primary key ────────────────────────────────────────────────────────
    model_name: Mapped[str] = mapped_column(
        String(150), primary_key=True, index=True,
        comment="Unique model run identifier, e.g. random_forest_clf_20260316_120233",
    )

    # ── P&L / trade counts ────────────────────────────────────────────────
    pnl: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Net PnL of the back-test run",
    )
    total_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    long_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    short_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    win_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    loss_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )
    breakeven_trades: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )

    # ── Rates ─────────────────────────────────────────────────────────────
    win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    loss_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Profit breakdown ──────────────────────────────────────────────────
    gross_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Per-trade averages ────────────────────────────────────────────────
    avg_trade_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_win: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Risk metrics ──────────────────────────────────────────────────────
    risk_reward_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Ratio metrics ─────────────────────────────────────────────────────
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Streak metrics ────────────────────────────────────────────────────
    max_consecutive_wins: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_consecutive_losses: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


# ---------------------------------------------------------------------------
# Concrete ORM models
# ---------------------------------------------------------------------------

class MLResult(_ModelResultMixin, Base):
    """Maps to `model_stats.ml_results`."""

    __tablename__ = "ml_results"
    __table_args__ = {"schema": "model_stats"}

    def __repr__(self) -> str:
        return f"<MLResult model_name={self.model_name!r} pnl={self.pnl}>"


class DLResult(_ModelResultMixin, Base):
    """Maps to `model_stats.dl_results`."""

    __tablename__ = "dl_results"
    __table_args__ = {"schema": "model_stats"}

    def __repr__(self) -> str:
        return f"<DLResult model_name={self.model_name!r} pnl={self.pnl}>"