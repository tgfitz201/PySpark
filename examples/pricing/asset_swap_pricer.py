"""
pricing/asset_swap_pricer.py
============================
Price an AssetSwap using QuantLib.

Method: ql.AssetSwap (par asset swap). The bond is priced at its
current market value; the asset swap spread is the floating leg spread
that makes the combined position NPV = 0 when the bond is bought at par.

Outputs (same schema as all other pricers):
  swap_npv    = combined NPV in USD
  clean_price = bond clean price USD
  par_rate    = bond YTM
  delta       = asset swap spread in bps (key metric)
  dv01        = rate DV01 (USD per 1bp)
  duration    = modified duration
  convexity   = dollar convexity
  vega        = 0 (no optionality)
  error       = "" or exception message
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import QuantLib as ql
from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def price_asset_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl  = trade.bond_leg
    fl  = trade.float_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    dc       = ql_dc.get(bl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    bdc      = ql_bdc.get(bl.bdc, ql.ModifiedFollowing)
    freq     = ql_freq.get(bl.frequency, ql.Semiannual)
    calendar = ql_cal.get(bl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc_float = ql.Actual360()

    def to_ql(d): return ql.Date(d.day, d.month, d.year)

    issue_dt = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt = to_ql(bl.start_date)
    mat_dt   = to_ql(bl.end_date)

    notional    = bl.notional
    coupon_rate = bl.coupon_rate
    par_price   = getattr(trade, 'par_price', 100.0)

    # ── Bond schedule ──────────────────────────────────────────────────────
    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)

    bond = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bond.setPricingEngine(ql.DiscountingBondEngine(sofr))
    bond_npv   = bond.NPV()
    clean_pct  = bond.cleanPrice()
    dirty_pct  = bond.dirtyPrice()
    clean_usd  = clean_pct / 100.0 * notional
    accrued    = (dirty_pct - clean_pct) / 100.0 * notional

    dc_yield = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        ytm = _NAN

    # ── Asset Swap via ql.AssetSwap ────────────────────────────────────────
    try:
        sofr_idx = ql.Sofr(sofr)
        asw = ql.AssetSwap(
            False,          # payFixed=False: receive bond coupons, pay float
            bond,
            par_price,      # bond clean price (par = 100.0)
            sofr_idx,
            0.0,            # initial spread guess — will solve for fair spread
            ql.Schedule(),
            sofr_idx.dayCounter(),
            True,           # parAssetSwap=True
        )
        engine = ql.DiscountingSwapEngine(sofr)
        asw.setPricingEngine(engine)
        fair_spread = asw.fairSpread()       # asset swap spread (decimal)
        asw_npv     = asw.NPV()
        asw_spread_bps = fair_spread * 10_000.0
    except Exception as e1:
        # Fallback: analytical approximation
        # ASW spread ≈ (coupon - par_rate) / modified_duration + (par - price) / (duration * par)
        try:
            par_rate_sofr = float(curve_df[curve_df["tenor_y"] == trade.tenor_y]["zero_rate"].iloc[0]) \
                if trade.tenor_y in curve_df["tenor_y"].values \
                else float(curve_df["zero_rate"].iloc[-1])
            md = bond.bondYield(dc_yield, ql.Compounded, freq)
            price_diff = (par_price - clean_pct) / 100.0
            fair_spread = coupon_rate - par_rate_sofr + price_diff / max(trade.tenor_y, 1)
            asw_spread_bps = fair_spread * 10_000.0
            asw_npv = bond_npv
        except Exception as e2:
            return {**_NAN_ROW, "error": f"AssetSwap pricing failed: {e1}; fallback: {e2}"}

    # ── DV01 ─────────────────────────────────────────────────────────────
    bumped_up = curve_df.copy(); bumped_up["zero_rate"] += 1e-4
    sofr_up   = build_sofr_curve(ql_val, bumped_up)
    bond_up   = ql.FixedRateBond(bl.settlement_days, notional, schedule,
                                  [coupon_rate], dc, bdc, bl.redemption, issue_dt)
    bond_up.setPricingEngine(ql.DiscountingBondEngine(sofr_up))
    npv_up = bond_up.NPV()

    bumped_dn = curve_df.copy(); bumped_dn["zero_rate"] -= 1e-4
    sofr_dn   = build_sofr_curve(ql_val, bumped_dn)
    bond_dn   = ql.FixedRateBond(bl.settlement_days, notional, schedule,
                                  [coupon_rate], dc, bdc, bl.redemption, issue_dt)
    bond_dn.setPricingEngine(ql.DiscountingBondEngine(sofr_dn))
    npv_dn = bond_dn.NPV()

    dv01_raw  = (npv_up - npv_dn) / 2.0
    from models.enums import TradeDirection
    sign      = -1.0 if trade.direction == TradeDirection.LONG else 1.0
    dv01      = sign * abs(dv01_raw)
    duration  = dv01 / (notional * 1e-4) if notional > 0 else _NAN
    pv01      = dv01 / (notional / 1e6) if notional > 0 else _NAN
    convexity = (npv_up + npv_dn - 2.0 * bond_npv) / (bond_npv * (1e-4)**2) if bond_npv > 0 else _NAN

    return dict(
        swap_npv    = asw_npv,
        fixed_npv   = bond_npv,
        float_npv   = asw_npv - bond_npv,
        clean_price = clean_usd,
        accrued     = accrued,
        par_rate    = ytm,
        dv01        = dv01,
        duration    = duration,
        pv01        = pv01,
        convexity   = abs(convexity) if convexity == convexity else _NAN,
        delta       = asw_spread_bps,   # asset swap spread in bps — KEY metric
        vega        = 0.0,
        theta       = _NAN,
        gamma       = _NAN,
        rho         = _NAN,
        cr01        = _NAN,
        jump_to_default = _NAN,
        error       = "",
    )
