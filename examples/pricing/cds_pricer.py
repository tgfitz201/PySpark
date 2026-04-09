"""pricing/cds_pricer.py — CreditDefaultSwap pricer."""

from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_cds(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    QuantLib CDS pricer: FlatHazardRate + MidPointCdsEngine.

    direction BUY  = protection buyer  (pays running spread)
    direction SELL = protection seller (receives running spread)
    """
    cl     = trade.protection_leg   # CDSProtectionLeg: reference entity, spread, recovery
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql.TARGET()
    dc       = ql.Actual360()

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    start_ql = to_ql(cl.start_date)
    mat_ql   = to_ql(cl.end_date)

    schedule = ql.Schedule(start_ql, mat_ql, ql.Period(ql.Quarterly),
                           calendar, ql.Following, ql.Following,
                           ql.DateGeneration.Forward, False)

    side     = ql.Protection.Buyer if getattr(trade.direction, 'value', trade.direction) == "BUY" else ql.Protection.Seller
    spread   = cl.credit_spread
    notional = cl.notional
    recovery = cl.recovery_rate
    hazard   = cl.hazard_rate if cl.hazard_rate > 0 else spread / (1.0 - recovery)

    dp_handle = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(ql_val, ql.QuoteHandle(ql.SimpleQuote(hazard)), ql.Actual365Fixed())
    )

    cds = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds.setPricingEngine(ql.MidPointCdsEngine(dp_handle, recovery, sofr))
    npv = cds.NPV()
    try:
        fair_spread = cds.fairSpread()
    except Exception:
        fair_spread = _NAN

    # Premium leg NPV (spread payments) and protection leg NPV (contingent)
    try:
        premium_npv    = cds.premiumNPV()
        protection_npv = cds.protectionNPV()
    except Exception:
        premium_npv = protection_npv = _NAN

    # DV01: bump SOFR by +1bp
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    cds_b  = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds_b.setPricingEngine(ql.MidPointCdsEngine(dp_handle, recovery, sofr_b))
    dv01     = cds_b.NPV() - npv
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    # CR01: bump credit spread by +1bp (implies new hazard rate)
    spread_b  = spread + 1e-4
    haz_b     = spread_b / (1.0 - recovery)
    dp_hand_b = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(ql_val, ql.QuoteHandle(ql.SimpleQuote(haz_b)), ql.Actual365Fixed())
    )
    cds_c = ql.CreditDefaultSwap(side, notional, spread, schedule, ql.Following, dc)
    cds_c.setPricingEngine(ql.MidPointCdsEngine(dp_hand_b, recovery, sofr))
    cr01 = cds_c.NPV() - npv

    # Jump-to-default: immediate default P&L
    lgd = (1.0 - recovery) * notional
    jump_to_default = lgd if getattr(trade.direction, 'value', trade.direction) == "BUY" else -lgd

    return dict(fixed_npv=premium_npv, float_npv=protection_npv, swap_npv=npv,
                par_rate=fair_spread, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=_NAN, delta=_NAN,
                gamma=_NAN, rho=_NAN,
                cr01=cr01, jump_to_default=jump_to_default, error="")


def price_cds(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a CreditDefaultSwap trade."""
    return _price_cds(trade, curve_df)
