"""pricing/option_pricer.py — swaption pricers (OptionTrade and InterestRateSwaption)."""

from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a European swaption using Black / Bachelier flat vol engine.
    ol.vol_type: "LOGNORMAL" -> BlackSwaptionEngine  |  "NORMAL" -> BachelierSwaptionEngine
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()
    ol = trade.option_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(ol.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc_fixed = ql_dc.get(ol.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    dc_float = ql.Actual360()
    bdc      = ql_bdc.get(ol.bdc, ql.ModifiedFollowing)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    expiry_dt = to_ql(ol.start_date)
    mat_dt    = to_ql(ol.end_date)

    fixed_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Semiannual),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Quarterly),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                             ql.USDCurrency(), calendar, bdc, False, dc_float, sofr)

    is_payer  = (ol.option_type == "PAYER_SWAPTION")
    stype     = ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver
    underlying = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                                float_sch, sofr_idx, 0.0, dc_float)
    underlying.setPricingEngine(ql.DiscountingSwapEngine(sofr))
    try:
        atm_rate = underlying.fairRate()
    except Exception:
        atm_rate = _NAN

    exercise = ql.EuropeanExercise(expiry_dt)
    swaption = ql.Swaption(underlying, exercise)
    flat_vol = ql.QuoteHandle(ql.SimpleQuote(ol.vol))
    shift    = ol.vol_shift if ol.vol_shift else 0.0

    def _make_engine(crv, vol_qh):
        if ol.vol_type == "NORMAL":
            return ql.BachelierSwaptionEngine(crv, vol_qh)
        return ql.BlackSwaptionEngine(crv, vol_qh, ql.Actual365Fixed(), shift)

    swaption.setPricingEngine(_make_engine(sofr, flat_vol))
    premium = swaption.NPV()
    sign    = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SELL" else 1.0
    net_pnl = sign * premium

    # DV01: +1bp parallel SOFR shift
    bumped  = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b  = build_sofr_curve(ql_val, bumped)
    sofr_idx_b = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                              ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_b)
    under_b = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                             float_sch, sofr_idx_b, 0.0, dc_float)
    swptn_b = ql.Swaption(under_b, exercise)
    swptn_b.setPricingEngine(_make_engine(sofr_b, flat_vol))
    dv01     = sign * (swptn_b.NPV() - premium)
    duration = dv01 / (ol.notional * 1e-4)
    pv01     = dv01 / ol.notional * 1_000_000

    # Vega: +1% flat vol shift
    vol_up  = ql.QuoteHandle(ql.SimpleQuote(ol.vol + 0.01))
    under_v = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                             float_sch, sofr_idx, 0.0, dc_float)
    swptn_v = ql.Swaption(under_v, exercise)
    swptn_v.setPricingEngine(_make_engine(sofr, vol_up))
    vega = sign * (swptn_v.NPV() - premium)

    # Theta: reprice with evaluationDate advanced by 1 calendar day
    theta = _NAN
    try:
        ql_val_1d = ql_val + 1
        if expiry_dt.serialNumber() > ql_val_1d.serialNumber():
            ql.Settings.instance().evaluationDate = ql_val_1d
            sofr_1d     = build_sofr_curve(ql_val_1d, curve_df)
            sofr_idx_1d = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                                       ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_1d)
            under_1d = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike, dc_fixed,
                                      float_sch, sofr_idx_1d, 0.0, dc_float)
            swptn_1d = ql.Swaption(under_1d, exercise)
            swptn_1d.setPricingEngine(_make_engine(sofr_1d, flat_vol))
            premium_1d = swptn_1d.NPV()
            theta = sign * (premium_1d - premium)
    except Exception:
        pass
    finally:
        ql.Settings.instance().evaluationDate = ql_val  # always restore

    # Delta + Gamma: central finite difference on strike (∂V/∂K, ∂²V/∂K²)
    h = 1e-4
    delta = _NAN
    gamma = _NAN
    try:
        under_lo = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike - h, dc_fixed,
                                  float_sch, sofr_idx, 0.0, dc_float)
        swptn_lo = ql.Swaption(under_lo, exercise)
        swptn_lo.setPricingEngine(_make_engine(sofr, flat_vol))
        V_lo = swptn_lo.NPV()

        under_hi = ql.VanillaSwap(stype, ol.notional, fixed_sch, ol.strike + h, dc_fixed,
                                  float_sch, sofr_idx, 0.0, dc_float)
        swptn_hi = ql.Swaption(under_hi, exercise)
        swptn_hi.setPricingEngine(_make_engine(sofr, flat_vol))
        V_hi = swptn_hi.NPV()

        delta = sign * (V_lo - premium) / h           # ∂V/∂K (lower K → higher payer premium)
        gamma = sign * (V_hi - 2.0 * premium + V_lo) / (h ** 2)   # ∂²V/∂K²
    except Exception:
        pass

    # Rho: parallel SOFR sensitivity scaled to 1% (= DV01 × 100)
    rho = dv01 * 100.0

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=net_pnl,
                par_rate=atm_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=vega, theta=theta, delta=delta,
                gamma=gamma, rho=rho,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_irs_swaption(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price an InterestRateSwaption using the QuantLib Black/Bachelier swaption
    engine.  The underlying swap is constructed from legs[0] (FixedLeg) and
    legs[1] (FloatLeg); the option terms come from legs[2] (OptionLeg).

    Greeks computed:
      swap_npv  — option premium (signed by direction)
      fixed_npv — PV of underlying fixed leg (at expiry strike)
      float_npv — PV of underlying float leg (at current fwd rates)
      dv01      — parallel +1bp SOFR shift
      vega      — +1% flat vol shift
      theta     — 1-day time decay
      delta     — d(premium)/d(swap_rate) via strike -1bp
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    fl  = trade.fixed_leg     # legs[0]
    fll = trade.float_leg     # legs[1]
    ol  = trade.option_leg    # legs[2]

    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(fl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc_fixed = ql_dc.get(fl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    dc_float = ql.Actual360()
    bdc      = ql_bdc.get(fl.bdc, ql.ModifiedFollowing)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    expiry_dt = to_ql(ol.start_date)   # option expiry = underlying effective date
    mat_dt    = to_ql(fl.end_date)     # underlying swap maturity

    fixed_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Semiannual),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(expiry_dt, mat_dt, ql.Period(ql.Quarterly),
                            calendar, bdc, bdc, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                             ql.USDCurrency(), calendar, bdc, False, dc_float, sofr)

    is_payer  = (ol.option_type == "PAYER_SWAPTION")
    stype     = ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver
    notl      = fl.notional
    strike    = ol.strike

    underlying = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                                float_sch, sofr_idx, 0.0, dc_float)
    underlying.setPricingEngine(ql.DiscountingSwapEngine(sofr))

    try:
        atm_rate   = underlying.fairRate()
        fixed_npv_ = underlying.fixedLegNPV()
        float_npv_ = underlying.floatingLegNPV()
    except Exception:
        atm_rate = fixed_npv_ = float_npv_ = _NAN

    exercise  = ql.EuropeanExercise(expiry_dt)
    swaption  = ql.Swaption(underlying, exercise)
    flat_vol  = ql.QuoteHandle(ql.SimpleQuote(ol.vol))
    shift     = ol.vol_shift if ol.vol_shift else 0.0

    def _engine(crv, vqh):
        if ol.vol_type == "NORMAL":
            return ql.BachelierSwaptionEngine(crv, vqh)
        return ql.BlackSwaptionEngine(crv, vqh, ql.Actual365Fixed(), shift)

    swaption.setPricingEngine(_engine(sofr, flat_vol))
    premium  = swaption.NPV()
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SELL" else 1.0
    net_pnl  = sign * premium

    # DV01: +1bp SOFR curve shift
    bumped  = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b  = build_sofr_curve(ql_val, bumped)
    sofr_ib = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                           ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_b)
    under_b = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                             float_sch, sofr_ib, 0.0, dc_float)
    swptn_b = ql.Swaption(under_b, exercise)
    swptn_b.setPricingEngine(_engine(sofr_b, flat_vol))
    dv01     = sign * (swptn_b.NPV() - premium)
    duration = dv01 / (notl * 1e-4)
    pv01     = dv01 / notl * 1_000_000

    # Vega: +1% vol shift
    vol_up  = ql.QuoteHandle(ql.SimpleQuote(ol.vol + 0.01))
    under_v = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                             float_sch, sofr_idx, 0.0, dc_float)
    swptn_v = ql.Swaption(under_v, exercise)
    swptn_v.setPricingEngine(_engine(sofr, vol_up))
    vega = sign * (swptn_v.NPV() - premium)

    # Theta: 1-day time decay
    theta = _NAN
    try:
        ql_val_1d = ql_val + 1
        if expiry_dt.serialNumber() > ql_val_1d.serialNumber():
            ql.Settings.instance().evaluationDate = ql_val_1d
            sofr_1d = build_sofr_curve(ql_val_1d, curve_df)
            sofr_i1 = ql.IborIndex("SOFR", ql.Period(3, ql.Months), 2,
                                   ql.USDCurrency(), calendar, bdc, False, dc_float, sofr_1d)
            under_1d = ql.VanillaSwap(stype, notl, fixed_sch, strike, dc_fixed,
                                      float_sch, sofr_i1, 0.0, dc_float)
            swptn_1d = ql.Swaption(under_1d, exercise)
            swptn_1d.setPricingEngine(_engine(sofr_1d, flat_vol))
            theta = sign * (swptn_1d.NPV() - premium)
    except Exception:
        pass
    finally:
        ql.Settings.instance().evaluationDate = ql_val

    # Delta + Gamma: central finite difference on strike (∂V/∂K, ∂²V/∂K²)
    h = 1e-4
    delta = _NAN
    gamma = _NAN
    try:
        under_lo = ql.VanillaSwap(stype, notl, fixed_sch, strike - h, dc_fixed,
                                  float_sch, sofr_idx, 0.0, dc_float)
        swptn_lo = ql.Swaption(under_lo, exercise)
        swptn_lo.setPricingEngine(_engine(sofr, flat_vol))
        V_lo = swptn_lo.NPV()

        under_hi = ql.VanillaSwap(stype, notl, fixed_sch, strike + h, dc_fixed,
                                  float_sch, sofr_idx, 0.0, dc_float)
        swptn_hi = ql.Swaption(under_hi, exercise)
        swptn_hi.setPricingEngine(_engine(sofr, flat_vol))
        V_hi = swptn_hi.NPV()

        delta = sign * (V_lo - premium) / h           # ∂V/∂K
        gamma = sign * (V_hi - 2.0 * premium + V_lo) / (h ** 2)   # ∂²V/∂K²
    except Exception:
        pass

    # Rho: parallel SOFR sensitivity scaled to 1% (= DV01 × 100)
    rho = dv01 * 100.0

    return dict(
        fixed_npv=fixed_npv_, float_npv=float_npv_, swap_npv=net_pnl,
        par_rate=atm_rate, clean_price=_NAN, accrued=_NAN,
        dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
        vega=vega, theta=theta, delta=delta,
        gamma=gamma, rho=rho,
        cr01=_NAN, jump_to_default=_NAN, error="",
    )


def price_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an OptionTrade (swaption)."""
    return _price_option(trade, curve_df)


def price_irs_swaption(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an InterestRateSwaption."""
    return _price_irs_swaption(trade, curve_df)
