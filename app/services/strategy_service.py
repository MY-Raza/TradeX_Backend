from __future__ import annotations

import math
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy_model import StrategyRegistry
from app.schemas.strategy_schema import (
    PaginatedStrategies,
    StrategyDetail,
    StrategyFilterOptions,
    StrategyListItem,
)

# ===========================================================================
# Column-category constants
# Computed once from the ORM model – never hard-code column names in queries.
# ===========================================================================

#: All 66 indicator flag column names
INDICATOR_COLUMNS: list[str] = [
    "bbands", "dema", "ema", "ht_trendline", "kama", "ma", "mama",
    "midpoint", "midprice", "sar", "sarext", "sma", "t3", "tema", "trima", "wma",
    "adx", "adxr", "apo", "aroon", "aroonosc", "bop", "cci", "cmo", "dx",
    "macd", "macdext", "mfi", "minus_di", "minus_dm", "mom",
    "plus_di", "plus_dm", "ppo", "roc", "rocp", "rocr", "rocr100",
    "rsi", "stoch", "stochf", "stochrsi", "trix", "willr",
    "ad", "adosc", "obv",
    "atr", "natr", "trange",
    "avgprice", "medprice", "typprice", "wclprice",
    "ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_sine", "ht_trendmode",
    "linearreg", "linearreg_angle", "linearreg_intercept", "linearreg_slope",
    "stddev", "tsf", "var",
    "ultosc",
]

#: All 61 candlestick pattern flag column names
PATTERN_COLUMNS: list[str] = [
    "cdl2crows", "cdl3blackcrows", "cdl3inside", "cdl3linestrike", "cdl3outside",
    "cdl3starsinsouth", "cdl3whitesoldiers", "cdlabandonedbaby", "cdladvanceblock",
    "cdlbelthold", "cdlbreakaway", "cdlclosingmarubozu", "cdlconcealbabyswall",
    "cdlcounterattack", "cdldarkcloudcover", "cdldoji", "cdldojistar",
    "cdldragonflydoji", "cdlengulfing", "cdleveningdojistar", "cdleveningstar",
    "cdlgapsidesidewhite", "cdlgravestonedoji", "cdlhammer", "cdlhangingman",
    "cdlharami", "cdlharamicross", "cdlhighwave", "cdlhikkake", "cdlhikkakemod",
    "cdlhomingpigeon", "cdlidentical3crows", "cdlinneck", "cdlinvertedhammer",
    "cdlkicking", "cdlkickingbylength", "cdlladderbottom", "cdllongleggeddoji",
    "cdllongline", "cdlmarubozu", "cdlmatchinglow", "cdlmathold",
    "cdlmorningdojistar", "cdlmorningstar", "cdlonneck", "cdlpiercing",
    "cdlrickshawman", "cdlrisefall3methods", "cdlseparatinglines",
    "cdlshootingstar", "cdlshortline", "cdlspinningtop", "cdlstalledpattern",
    "cdlsticksandwich", "cdltakuri", "cdltasukigap", "cdlthrusting",
    "cdltristar", "cdlunique3river", "cdlupsidegap2crows", "cdlxsidegap3methods",
]

#: Maps each indicator name → list of (label, column_name) period tuples.
#: Indicators with no lookback (bop, avgprice, etc.) map to an empty list.
INDICATOR_PERIOD_MAP: dict[str, list[tuple[str, str]]] = {
    # overlap / MA
    "bbands":        [("period", "bbands_period")],
    "dema":          [("period", "dema_period")],
    "ema":           [("period", "ema_period")],
    "ht_trendline":  [],
    "kama":          [("period", "kama_period")],
    "ma":            [("period", "ma_period")],
    "mama":          [],
    "midpoint":      [("period", "midpoint_period")],
    "midprice":      [("period", "midprice_period")],
    "sar":           [],
    "sarext":        [],
    "sma":           [("period", "sma_period")],
    "t3":            [("period", "t3_period")],
    "tema":          [("period", "tema_period")],
    "trima":         [("period", "trima_period")],
    "wma":           [("period", "wma_period")],
    # momentum
    "adx":           [("period", "adx_period")],
    "adxr":          [("period", "adxr_period")],
    "apo":           [("period", "apo_period")],
    "aroon":         [("period", "aroon_period")],
    "aroonosc":      [("period", "aroonosc_period")],
    "bop":           [],
    "cci":           [("period", "cci_period")],
    "cmo":           [("period", "cmo_period")],
    "dx":            [("period", "dx_period")],
    "macd":          [("fast",   "macd_fastperiod"),
                      ("slow",   "macd_slowperiod"),
                      ("signal", "macd_signalperiod")],
    "macdext":       [("fast",   "macdext_fastperiod"),
                      ("slow",   "macdext_slowperiod"),
                      ("signal", "macdext_signalperiod")],
    "mfi":           [("period", "mfi_period")],
    "minus_di":      [("period", "minus_di_period")],
    "minus_dm":      [("period", "minus_dm_period")],
    "mom":           [("period", "mom_period")],
    "plus_di":       [("period", "plus_di_period")],
    "plus_dm":       [("period", "plus_dm_period")],
    "ppo":           [("fast",   "ppo_fastperiod"),
                      ("slow",   "ppo_slowperiod")],
    "roc":           [("period", "roc_period")],
    "rocp":          [("period", "rocp_period")],
    "rocr":          [("period", "rocr_period")],
    "rocr100":       [("period", "rocr100_period")],
    "rsi":           [("period", "rsi_period")],
    "stoch":         [("fastk",  "stoch_fastk_period"),
                      ("slowk",  "stoch_slowk_period"),
                      ("slowd",  "stoch_slowd_period")],
    "stochf":        [("fast",   "stochf_fastperiod"),
                      ("slow",   "stochf_slowperiod")],
    "stochrsi":      [("period", "stochrsi_period")],
    "trix":          [("period", "trix_period")],
    "willr":         [("period", "willr_period")],
    # volume
    "ad":            [],
    "adosc":         [("fast",   "adosc_fastperiod"),
                      ("slow",   "adosc_slowperiod")],
    "obv":           [],
    # volatility
    "atr":           [("period", "atr_period")],
    "natr":          [("period", "natr_period")],
    "trange":        [],
    # price transform
    "avgprice":      [],
    "medprice":      [],
    "typprice":      [],
    "wclprice":      [],
    # Hilbert
    "ht_dcperiod":   [],
    "ht_dcphase":    [],
    "ht_phasor":     [],
    "ht_sine":       [],
    "ht_trendmode":  [],
    # stat / regression
    "linearreg":           [("period", "linearreg_period")],
    "linearreg_angle":     [("period", "linearreg_angle_period")],
    "linearreg_intercept": [("period", "linearreg_intercept_period")],
    "linearreg_slope":     [("period", "linearreg_slope_period")],
    "stddev":              [("period", "stddev_period")],
    "tsf":                 [("period", "tsf_period")],
    "var":                 [("period", "var_period")],
    # special
    "ultosc":        [],
}


# ===========================================================================
# Private helpers
# ===========================================================================

def _active_indicators(row: StrategyRegistry) -> list[str]:
    """Return names of all active (True) indicator flags."""
    return [col for col in INDICATOR_COLUMNS if getattr(row, col, False)]


def _active_patterns(row: StrategyRegistry) -> list[str]:
    """Return names of all active (True) pattern flags."""
    return [col for col in PATTERN_COLUMNS if getattr(row, col, False)]


def _build_indicator_details(
    row: StrategyRegistry,
    active: list[str],
) -> dict[str, dict[str, Optional[int]]]:
    """
    Build the `indicator_details` dict for the detail response.

    For each active indicator, look up its period columns from
    INDICATOR_PERIOD_MAP and read the actual values from the ORM row.
    Returns a mapping of: indicator_name → {param_label: value | None}.
    """
    details: dict[str, dict[str, Optional[int]]] = {}
    for ind in active:
        period_specs = INDICATOR_PERIOD_MAP.get(ind, [])
        params: dict[str, Optional[int]] = {}
        for label, col in period_specs:
            raw = getattr(row, col, None)
            params[label] = int(raw) if raw is not None else None
        details[ind] = params
    return details


def _to_list_item(row: StrategyRegistry) -> StrategyListItem:
    """Convert an ORM row to StrategyListItem (table card)."""
    return StrategyListItem(
        name=row.strategy,
        symbol=row.symbol,
        time_horizon=row.timehorizon,
        indicators=_active_indicators(row),
        patterns=_active_patterns(row),
        pnl_sum=row.pnl_sum,
    )


def _to_detail(row: StrategyRegistry) -> StrategyDetail:
    """Convert an ORM row to StrategyDetail (full record)."""
    active = _active_indicators(row)
    return StrategyDetail(
        name=row.strategy,
        symbol=row.symbol,
        time_horizon=row.timehorizon,
        indicators=active,
        patterns=_active_patterns(row),
        pnl_sum=row.pnl_sum,
        tp=row.tp,
        sl=row.sl,
        indicator_details=_build_indicator_details(row, active),
    )


# ===========================================================================
# Public service API
# ===========================================================================

async def get_strategies(
    db: AsyncSession,
    *,
    symbol: Optional[str] = None,
    time_horizon: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedStrategies:
    """
    Return a paginated, filtered list of strategies.

    Filters (all optional, all case-insensitive):
      symbol        – exact match on `symbol` column
      time_horizon  – exact match on `timehorizon` column
      search        – partial match on `strategy` column (ILIKE)
    """
    stmt = select(StrategyRegistry)

    if symbol:
        stmt = stmt.where(
            func.lower(StrategyRegistry.symbol) == symbol.lower().strip()
        )
    if time_horizon:
        stmt = stmt.where(
            func.lower(StrategyRegistry.timehorizon) == time_horizon.lower().strip()
        )
    if search:
        stmt = stmt.where(
            StrategyRegistry.strategy.ilike(f"%{search.strip()}%")
        )

    # ── Total count (reuse the same filter, no LIMIT/OFFSET) ──────────────
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    # ── Pagination ────────────────────────────────────────────────────────
    pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, pages))           # clamp to valid range
    offset = (page - 1) * page_size

    stmt = (
        stmt
        .order_by(StrategyRegistry.strategy)
        .limit(page_size)
        .offset(offset)
    )

    rows = (await db.execute(stmt)).scalars().all()

    return PaginatedStrategies(
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
        results=[_to_list_item(r) for r in rows],
    )


async def get_strategy_by_name(
    db: AsyncSession,
    strategy_name: str,
) -> Optional[StrategyDetail]:
    """
    Return the full detail record for a single strategy.
    Returns None if not found (the route raises 404).
    """
    stmt = select(StrategyRegistry).where(StrategyRegistry.strategy == strategy_name)
    row: Optional[StrategyRegistry] = (await db.execute(stmt)).scalars().first()
    return _to_detail(row) if row else None


async def get_filter_options(db: AsyncSession) -> StrategyFilterOptions:
    """
    Return distinct values for filterable fields.
    Used to populate frontend dropdowns without hard-coding values.
    """
    symbol_stmt = (
        select(StrategyRegistry.symbol)
        .distinct()
        .order_by(StrategyRegistry.symbol)
    )
    th_stmt = (
        select(StrategyRegistry.timehorizon)
        .distinct()
        .order_by(StrategyRegistry.timehorizon)
    )

    symbols = list((await db.execute(symbol_stmt)).scalars().all())
    time_horizons = list((await db.execute(th_stmt)).scalars().all())

    return StrategyFilterOptions(symbols=symbols, time_horizons=time_horizons)