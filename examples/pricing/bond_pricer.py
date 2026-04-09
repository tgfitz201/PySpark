"""pricing/bond_pricer.py — FixedRateBond pricer."""

from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a FixedRateBond using conventions stored on bond_leg."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl = trade.bond_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal[bl.calendar]
    dc       = ql_dc[bl.day_count]
    bdc      = ql_bdc[bl.bdc]
    freq     = ql_freq[bl.frequency]

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    issue_dt = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt = to_ql(bl.start_date); mat_dt = to_ql(bl.end_date)

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond.setPricingEngine(ql.DiscountingBondEngine(sofr))

    dirty_pct = bond.dirtyPrice()
    clean_pct = bond.cleanPrice()
    npv_usd   = bond.NPV()
    accrued   = (dirty_pct - clean_pct) / 100.0 * bl.notional
    dc_yield  = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        ytm = _NAN

    # DV01: +1bp parallel shift
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    bond_p1 = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond_p1.setPricingEngine(ql.DiscountingBondEngine(sofr_b))
    npv_p1   = bond_p1.NPV()
    raw_dv01 = npv_p1 - npv_usd
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01     = sign * raw_dv01
    duration = dv01 / (bl.notional * 1e-4)
    pv01     = dv01 / bl.notional * 1_000_000

    # Convexity: central difference (P_up + P_down - 2P) / (P * Δr²)
    bumped_m1 = curve_df.copy(); bumped_m1["zero_rate"] -= 1e-4
    sofr_m1   = build_sofr_curve(ql_val, bumped_m1)
    bond_m1   = ql.FixedRateBond(
        bl.settlement_days, bl.notional, schedule,
        [bl.coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond_m1.setPricingEngine(ql.DiscountingBondEngine(sofr_m1))
    npv_m1    = bond_m1.NPV()
    convexity = (npv_m1 + npv_p1 - 2.0 * npv_usd) / (npv_usd * 1e-8) if npv_usd != 0.0 else _NAN

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv_usd,
                par_rate=ytm, clean_price=clean_pct, accrued=accrued,
                dv01=dv01, duration=duration, pv01=pv01, convexity=convexity,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def price_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a Bond trade."""
    return _price_bond(trade, curve_df)
