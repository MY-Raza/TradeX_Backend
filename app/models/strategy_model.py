from __future__ import annotations
from typing import Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class StrategyRegistry(Base):
    """Maps to the `strategy_registry` PostgreSQL table."""

    __tablename__ = "strategy_registry"
    __table_args__ = {"schema": "strategies"} 

    # ── Primary key ────────────────────────────────────────────────────────
    strategy: Mapped[str] = mapped_column(
        String(100), primary_key=True, index=True,
        comment="Strategy identifier, e.g. sig_1h_btc_1"
    )

    # =========================================================================
    # GROUP 1 – Meta / Identity
    # =========================================================================
    symbol: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Ticker symbol: btc, eth, bnb, ada, xrp, doge, sol, dot, matic"
    )
    timehorizon: Mapped[str] = mapped_column(
        String(10), nullable=False, index=True,
        comment="Candle timeframe: 1h, 15m, 5m"
    )
    tp: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, comment="Take-profit multiplier"
    )
    sl: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, comment="Stop-loss multiplier"
    )
    pnl_sum: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="Cumulative back-test PnL"
    )

    # =========================================================================
    # GROUP 2 – Technical Indicator Flags  (66 BOOLEAN columns)
    # True = this indicator is active in the strategy
    # =========================================================================

    # -- Overlap / Moving average indicators --
    bbands:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    dema:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ema:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ht_trendline:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    kama:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ma:              Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    mama:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    midpoint:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    midprice:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    sar:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    sarext:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    sma:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    t3:              Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    tema:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    trima:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    wma:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Momentum indicators --
    adx:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    adxr:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    apo:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    aroon:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    aroonosc:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    bop:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cci:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cmo:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    dx:              Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    macd:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    macdext:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    mfi:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    minus_di:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    minus_dm:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    mom:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    plus_di:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    plus_dm:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ppo:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    roc:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    rocp:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    rocr:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    rocr100:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    rsi:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    stoch:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    stochf:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    stochrsi:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    trix:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    willr:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Volume indicators --
    ad:              Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    adosc:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    obv:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Volatility indicators --
    atr:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    natr:            Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    trange:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Price transform --
    avgprice:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    medprice:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    typprice:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    wclprice:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Hilbert transform --
    ht_dcperiod:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ht_dcphase:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ht_phasor:       Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ht_sine:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    ht_trendmode:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Statistical / regression --
    linearreg:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    linearreg_angle:       Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    linearreg_intercept:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    linearreg_slope:       Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    stddev:                Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    tsf:                   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    var:                   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # -- Ultimate Oscillator (stored as object/bool in CSV) --
    ultosc:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # =========================================================================
    # GROUP 3 – Candlestick Pattern Flags  (61 BOOLEAN columns)
    # True = this candlestick pattern is used in the strategy
    # =========================================================================

    cdl2crows:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3blackcrows:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3inside:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3linestrike:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3outside:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3starsinsouth:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdl3whitesoldiers:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlabandonedbaby:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdladvanceblock:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlbelthold:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlbreakaway:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlclosingmarubozu:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlconcealbabyswall: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlcounterattack:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdldarkcloudcover:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdldoji:             Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdldojistar:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdldragonflydoji:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlengulfing:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdleveningdojistar:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdleveningstar:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlgapsidesidewhite: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlgravestonedoji:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhammer:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhangingman:       Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlharami:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlharamicross:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhighwave:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhikkake:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhikkakemod:       Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlhomingpigeon:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlidentical3crows:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlinneck:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlinvertedhammer:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlkicking:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlkickingbylength:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlladderbottom:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdllongleggeddoji:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdllongline:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlmarubozu:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlmatchinglow:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlmathold:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlmorningdojistar:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlmorningstar:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlonneck:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlpiercing:         Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlrickshawman:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlrisefall3methods: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlseparatinglines:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlshootingstar:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlshortline:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlspinningtop:      Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlstalledpattern:   Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlsticksandwich:    Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdltakuri:           Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdltasukigap:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlthrusting:        Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdltristar:          Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlunique3river:     Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlupsidegap2crows:  Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    cdlxsidegap3methods: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)

    # =========================================================================
    # GROUP 4 – Single-period Parameters  (46 SMALLINT nullable columns)
    # Lookback window for the corresponding indicator flag
    # =========================================================================

    ema_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    midpoint_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    midprice_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    trima_period:               Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    wma_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    aroonosc_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    cci_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    minus_dm_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    plus_di_period:             Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    plus_dm_period:             Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    roc_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    trix_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    willr_period:               Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    linearreg_slope_period:     Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stddev_period:              Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    tsf_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    var_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    bbands_period:              Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    dema_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    kama_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    sma_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    tema_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    adxr_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    apo_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    aroon_period:               Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    mfi_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    mom_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    rocr_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    rsi_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    atr_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    natr_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    linearreg_period:           Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    linearreg_angle_period:     Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    ma_period:                  Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    t3_period:                  Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    cmo_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    minus_di_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    rocp_period:                Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stoch_fastk_period:         Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stoch_slowk_period:         Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stoch_slowd_period:         Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stochrsi_period:            Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    adx_period:                 Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    dx_period:                  Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    rocr100_period:             Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    linearreg_intercept_period: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # =========================================================================
    # GROUP 5 – Multi-period Parameters  (12 SMALLINT nullable columns)
    # Fast / slow / signal windows for oscillators with split periods
    # =========================================================================

    ppo_fastperiod:       Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    ppo_slowperiod:       Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stochf_fastperiod:    Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    stochf_slowperiod:    Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    adosc_fastperiod:     Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    adosc_slowperiod:     Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macdext_fastperiod:   Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macdext_slowperiod:   Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macdext_signalperiod: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macd_fastperiod:      Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macd_slowperiod:      Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    macd_signalperiod:    Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    # ── Repr ───────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"<StrategyRegistry id={self.id} strategy={self.strategy!r} "
            f"symbol={self.symbol!r} timehorizon={self.timehorizon!r}>"
        )