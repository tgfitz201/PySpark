"""pricing/swap_pricer.py — IRS, XCCY and basis-swap pricers."""

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_fixed_float(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a standard FIXED_FLOAT or FLOAT_FIXED IRS using QuantLib VanillaSwap."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    fl  = trade.fixed_leg
    flt = trade.float_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal[flt.calendar]
    dc_fixed = ql_dc[fl.day_count]
    dc_float = ql_dc[flt.day_count]
    bdc_f    = ql_bdc[fl.bdc]
    bdc_flt  = ql_bdc[flt.bdc]

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    eff = to_ql(fl.start_date); mat = to_ql(fl.end_date)

    fixed_sch = ql.Schedule(eff, mat, ql.Period(ql_freq[fl.frequency]),
                            calendar, bdc_f, bdc_f, ql.DateGeneration.Forward, False)
    float_sch = ql.Schedule(eff, mat, ql.Period(ql_freq[flt.frequency]),
                            calendar, bdc_flt, bdc_flt, ql.DateGeneration.Forward, False)
    sofr_idx  = ql.IborIndex(flt.index_name, ql.Period(flt.index_tenor_m, ql.Months),
                             flt.fixing_lag, ql.USDCurrency(), calendar, bdc_flt,
                             False, dc_float, sofr)
    short_rate  = float(curve_df["zero_rate"].iloc[0])
    eff_for_fix = eff
    for _ in range(36):
        fix_dt = calendar.advance(eff_for_fix, -flt.fixing_lag, ql.Days, ql.Preceding)
        if fix_dt > ql_val: break
        sofr_idx.addFixing(fix_dt, short_rate, True)
        eff_for_fix = calendar.advance(eff_for_fix, ql.Period(ql_freq[flt.frequency]), bdc_flt)

    stype    = ql.VanillaSwap.Payer if getattr(trade.direction, 'value', trade.direction) == "PAYER" else ql.VanillaSwap.Receiver
    notional = fl.notional; fxd_rate = fl.coupon_rate
    swap = ql.VanillaSwap(stype, notional, fixed_sch, fxd_rate, dc_fixed,
                          float_sch, sofr_idx, flt.spread, dc_float)
    swap.setPricingEngine(ql.DiscountingSwapEngine(sofr))
    swap_npv  = swap.NPV(); fixed_npv = swap.fixedLegNPV()
    float_npv = swap.floatingLegNPV(); par_rate = swap.fairRate()

    # DV01: +1bp parallel shift
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    sofr_idx_b = ql.IborIndex(flt.index_name, ql.Period(flt.index_tenor_m, ql.Months),
                              flt.fixing_lag, ql.USDCurrency(), calendar, bdc_flt,
                              False, dc_float, sofr_b)
    eff_for_fix = eff
    for _ in range(36):
        fix_dt = calendar.advance(eff_for_fix, -flt.fixing_lag, ql.Days, ql.Preceding)
        if fix_dt > ql_val: break
        sofr_idx_b.addFixing(fix_dt, short_rate, True)
        eff_for_fix = calendar.advance(eff_for_fix, ql.Period(ql_freq[flt.frequency]), bdc_flt)
    swap_b = ql.VanillaSwap(stype, notional, fixed_sch, fxd_rate, dc_fixed,
                            float_sch, sofr_idx_b, flt.spread, dc_float)
    swap_b.setPricingEngine(ql.DiscountingSwapEngine(sofr_b))
    dv01     = swap_b.NPV() - swap_npv
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    return dict(fixed_npv=fixed_npv, float_npv=float_npv, swap_npv=swap_npv,
                par_rate=par_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_fixed_fixed(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FIXED_FIXED swap: PV(receive leg[1]) - PV(pay leg[0])."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val
    sofr = build_sofr_curve(ql_val, curve_df)

    def leg_pv(leg, crv):
        dc   = ql_dc.get(leg.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
        freq = ql_freq.get(leg.frequency, ql.Semiannual)
        bdc  = ql_bdc.get(leg.bdc, ql.ModifiedFollowing)
        cal  = ql_cal.get(leg.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
        eff  = ql.Date(leg.start_date.day, leg.start_date.month, leg.start_date.year)
        mat  = ql.Date(leg.end_date.day, leg.end_date.month, leg.end_date.year)
        sch  = ql.Schedule(eff, mat, ql.Period(freq), cal, bdc, bdc,
                           ql.DateGeneration.Forward, False)
        fl   = ql.FixedRateLeg(sch, dc, [leg.notional], [leg.coupon_rate])
        return ql.CashFlows.npv(fl, crv, True)

    leg0, leg1 = trade.legs[0], trade.legs[1]
    pv0 = leg_pv(leg0, sofr)   # pay leg
    pv1 = leg_pv(leg1, sofr)   # receive leg
    swap_npv = pv1 - pv0

    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    dv01 = (leg_pv(leg1, sofr_b) - leg_pv(leg0, sofr_b)) - swap_npv
    notional = leg0.notional
    duration = dv01 / (notional * 1e-4)
    pv01 = dv01 / notional * 1_000_000

    return dict(fixed_npv=pv0, float_npv=pv1, swap_npv=swap_npv,
                par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_float_float(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FLOAT_FLOAT basis swap: NPV is PV of spread on leg[0]."""
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val
    sofr = build_sofr_curve(ql_val, curve_df)

    leg0 = trade.legs[0]   # SOFR1M + spread (pay)
    leg1 = trade.legs[1]   # SOFR3M (receive)
    notional = leg0.notional

    # For a basis swap both float legs have near-equal PV at par.
    # The NPV is essentially PV of the spread payments on leg0.
    # Use annuity factor from SOFR discount curve.
    ql_mat = ql.Date(leg0.end_date.day, leg0.end_date.month, leg0.end_date.year)
    tenor_frac = (leg0.end_date - leg0.start_date).days / 365.0
    r_avg = sofr.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    if r_avg > 0 and tenor_frac > 0:
        annuity = notional * (1.0 - math.exp(-r_avg * tenor_frac)) / r_avg
    else:
        annuity = notional * tenor_frac

    swap_npv = leg0.spread * annuity  # PV of spread payments (positive = receiver benefits)

    # DV01: bump SOFR
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    r_b    = sofr_b.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    if r_b > 0 and tenor_frac > 0:
        annuity_b = notional * (1.0 - math.exp(-r_b * tenor_frac)) / r_b
    else:
        annuity_b = notional * tenor_frac
    dv01 = leg0.spread * annuity_b - swap_npv
    duration = dv01 / (notional * 1e-4)
    pv01 = dv01 / notional * 1_000_000

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=swap_npv,
                par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Dispatch IRS pricing based on swap_subtype."""
    subtype = getattr(trade, 'swap_subtype', 'FIXED_FLOAT')
    if subtype == "FIXED_FIXED":
        return _price_fixed_fixed(trade, curve_df)
    if subtype == "FLOAT_FLOAT":
        return _price_float_float(trade, curve_df)
    return _price_fixed_float(trade, curve_df)


def _price_xccy(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a cross-currency swap by discounting each leg independently in its
    own currency then converting to USD.

    pay_leg  (legs[0]): cash flows you pay out → negative contribution to NPV
    recv_leg (legs[1]): cash flows you receive  → positive contribution to NPV
    NPV = recv_npv_usd - pay_npv_usd
    DV01 = NPV(r+1bp) - NPV(r)  → sign falls out naturally from bump.
    """
    try:
        T   = max(float(trade.tenor_y or 5), 0.25)
        r   = 0.05  # flat discount rate (market proxy)
        n   = max(1, int(round(T)))

        def _annuity(rate):
            return sum(1.0 / (1.0 + rate) ** t for t in range(1, n + 1))

        def _disc(rate):
            return 1.0 / (1.0 + rate) ** T

        pay_ccy  = str(getattr(trade, "pay_currency",     "USD") or "USD")
        recv_ccy = str(getattr(trade, "receive_currency", "EUR") or "EUR")
        fx = max(float(getattr(trade, "fx_rate", 1.0) or 1.0), 1e-9)

        def _leg_pv(leg, rate, ann, disc):
            notl = float(getattr(leg, "notional", 0) or 0)
            lt   = str(getattr(leg, "leg_type", "FIXED"))
            if lt == "FIXED":
                cpn = float(getattr(leg, "coupon_rate", 0.05) or 0.05)
                pv  = notl * (cpn * ann + disc - 1.0)
            elif lt == "FLOAT":
                sprd = float(getattr(leg, "spread", 0.0) or 0.0)
                pv   = notl * sprd * ann
            else:
                pv = 0.0
            ccy = str(getattr(leg, "currency", pay_ccy) or pay_ccy)
            if ccy == recv_ccy and ccy != pay_ccy:
                pv /= fx
            return pv

        def _npv(rate):
            ann  = _annuity(rate)
            disc = _disc(rate)
            pay  = sum(_leg_pv(l, rate, ann, disc) for l in trade.legs[:1])
            recv = sum(_leg_pv(l, rate, ann, disc) for l in trade.legs[1:])
            return recv - pay

        ann  = _annuity(r)
        disc = _disc(r)
        npv      = _npv(r)
        npv_bump = _npv(r + 0.0001)
        total_dv01 = npv_bump - npv   # correct sign: +ve for PAYER, -ve for RECEIVER

        return {
            "trade_id":        trade.trade_id,
            "swap_npv":        npv,
            "fixed_npv":       sum(_leg_pv(l, r, ann, disc) for l in trade.legs[1:]),
            "float_npv":      -sum(_leg_pv(l, r, ann, disc) for l in trade.legs[:1]),
            "dv01":            total_dv01,
            "duration":        round(T / 2.0, 4),
            "par_rate":        r,
            "pv01":            total_dv01,
            "cr01":            0.0,
            "jump_to_default": 0.0,
            "delta":           0.0,
            "gamma":           0.0,
            "vega":            0.0,
            "theta":           0.0,
            "rho":             total_dv01,
            "convexity":       0.0,
            "clean_price":     0.0,
            "accrued":         0.0,
        }
    except Exception as e:
        return {"trade_id": trade.trade_id, "error": str(e), "swap_npv": None}


def price_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an InterestRateSwap — dispatches by swap_subtype."""
    return _price_swap(trade, curve_df)


def price_xccy(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a CrossCurrencySwap."""
    return _price_xccy(trade, curve_df)


def price_irs(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an InterestRateSwap — checks for cross-currency legs first."""
    ccys = {l.currency for l in trade.legs if l.currency}
    if len(ccys) > 1:
        return _price_xccy(trade, curve_df)
    return _price_swap(trade, curve_df)
