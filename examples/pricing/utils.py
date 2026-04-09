"""pricing/utils.py — shared constants and curve builder used by all pricers."""

from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

# ── NaN sentinels ─────────────────────────────────────────────────────────────

_NAN = float("nan")
_NAN_F = float("nan")
_NAN_ROW: Dict[str, Any] = dict(
    fixed_npv=_NAN, float_npv=_NAN, swap_npv=_NAN, par_rate=_NAN,
    clean_price=_NAN, accrued=_NAN,
    dv01=_NAN, duration=_NAN, pv01=_NAN, convexity=_NAN,
    vega=_NAN, theta=_NAN, delta=_NAN,
    gamma=_NAN, rho=_NAN,
    cr01=_NAN, jump_to_default=_NAN, error="",
)


def build_sofr_curve(val_date: ql.Date, curve_df: pd.DataFrame) -> ql.YieldTermStructureHandle:
    """Build QuantLib yield curve from broadcast curve_df."""
    dc = ql.Actual365Fixed()
    dates = [val_date]
    rates = [float(curve_df["zero_rate"].iloc[0])]
    for _, row in curve_df.iterrows():
        months = max(1, int(round(float(row["tenor_y"]) * 12)))
        dates.append(val_date + ql.Period(months, ql.Months))
        rates.append(float(row["zero_rate"]))
    curve = ql.ZeroCurve(dates, rates, dc, ql.NullCalendar(),
                         ql.Linear(), ql.Continuous, ql.Annual)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def _ql_maps():
    """Return local QL convention dicts (no module-level QL objects — not picklable)."""
    freq = {
        "SEMIANNUAL": ql.Semiannual, "QUARTERLY": ql.Quarterly,
        "ANNUAL": ql.Annual, "MONTHLY": ql.Monthly,
    }
    dc = {
        "30/360":   ql.Thirty360(ql.Thirty360.BondBasis),
        "ACT/360":  ql.Actual360(),
        "ACT/365F": ql.Actual365Fixed(),
        "ACT/ACT":  ql.ActualActual(ql.ActualActual.ISDA),
    }
    bdc = {
        "MOD_FOLLOWING": ql.ModifiedFollowing,
        "FOLLOWING":     ql.Following,
        "PRECEDING":     ql.Preceding,
    }
    cal = {
        "US_GOVT":       ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        "UK_SETTLEMENT": ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
        "TARGET":        ql.TARGET(),
    }
    return freq, dc, bdc, cal
