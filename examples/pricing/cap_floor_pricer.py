"""
pricing/cap_floor_pricer.py
===========================
Price an interest-rate cap or floor using QuantLib's CapFloor engine
with a Black-76 vol from the market data cap_floor_vols surface.

Method
------
1. Build SOFR OIS discount curve from curve_df.
2. Build a flat SOFR3M forward curve (approximated with the discount curve).
3. Look up the ATM-strike vol from mkt.get_cap_floor_vol(expiry_y, strike).
4. Price with ql.BlackCapFloorEngine (Black-76 lognormal vol).
5. Greeks via bumping:
   - DV01 (+1bp parallel shift)
   - Vega (+1% absolute vol bump)
"""
from __future__ import annotations
from typing import Dict, Any
import QuantLib as ql
import pandas as pd
from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_cap_floor(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    cl     = trade.cap_floor_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    from models.market_data import MarketDataCache
    mkt    = MarketDataCache().get_or_create(val_dt)

    sofr    = build_sofr_curve(ql_val, curve_df)
    dc      = ql_dc.get(cl.day_count, ql.Actual360())
    bdc     = ql_bdc.get(cl.bdc, ql.ModifiedFollowing)
    freq    = ql_freq.get(cl.frequency, ql.Quarterly)
    cal     = ql_cal.get(cl.calendar, ql.UnitedStates(ql.UnitedStates.FederalReserve))

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    start_dt = to_ql(cl.start_date)
    mat_dt   = to_ql(cl.end_date)

    notional    = cl.notional
    strike      = cl.strike
    cf_type_str = cl.cap_floor_type.upper()  # "CAP" or "FLOOR"
    tenor_y     = trade.tenor_y or max(1, (cl.end_date - cl.start_date).days // 365)

    # Vol from cap/floor surface
    surf_vol = mkt.get_cap_floor_vol(tenor_y, strike)
    vol      = surf_vol if surf_vol > 0 else 0.50  # fallback to 50% lognormal

    # Index: SOFR3M approximated as IborIndex off the OIS discount curve
    index_tenor = ql.Period(cl.index_tenor_m, ql.Months)

    def _make_ibor(crv):
        idx = ql.IborIndex(
            "SOFR3M", index_tenor, 2, ql.USDCurrency(), cal, bdc, False,
            ql.Actual360(), crv)
        fixing_dt = idx.fixingDate(start_dt)
        if fixing_dt <= ql_val:
            spot_rate = crv.currentLink().zeroRate(0.25, ql.Continuous).rate()
            idx.addFixing(fixing_dt, spot_rate, True)
        return idx

    ibor      = _make_ibor(sofr)
    schedule  = ql.Schedule(start_dt, mat_dt, index_tenor, cal,
                             bdc, bdc, ql.DateGeneration.Forward, False)
    n_coupons = len(schedule) - 1
    if n_coupons < 1:
        return {**_NAN_ROW, "error": "schedule has no coupons"}

    def _vol_struct(v):
        return ql.ConstantOptionletVolatility(ql_val, cal, bdc, v, ql.Actual365Fixed())

    def _make_cf(ibor_idx, vs):
        coupons = [
            ql.IborCoupon(schedule[i + 1], notional, schedule[i], schedule[i + 1],
                          2, ibor_idx, 1.0, 0.0, ql.Date(), ql.Date(), ql.Actual360())
            for i in range(n_coupons)
        ]
        vh = ql.OptionletVolatilityStructureHandle(vs)
        ql.setCouponPricer(ql.Leg(coupons), ql.BlackIborCouponPricer(vh))
        strikes_list = [strike] * n_coupons
        cf = ql.Cap(coupons, strikes_list) if cf_type_str == "CAP" else ql.Floor(coupons, strikes_list)
        cf.setPricingEngine(ql.BlackCapFloorEngine(ibor_idx.forwardingTermStructure(), vh))
        return cf

    cap_floor = _make_cf(ibor, _vol_struct(vol))
    npv_usd   = cap_floor.NPV()
    sign      = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    npv_usd  *= sign

    # DV01: +1bp parallel shift
    bumped   = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b   = build_sofr_curve(ql_val, bumped)
    cf_b     = _make_cf(_make_ibor(sofr_b), _vol_struct(vol))
    raw_dv01 = (cf_b.NPV() - cap_floor.NPV()) * sign
    dv01     = raw_dv01

    # Vega: +1% absolute vol bump
    cf_vg = _make_cf(ibor, _vol_struct(vol + 0.01))
    vega  = (cf_vg.NPV() * sign) - npv_usd

    duration = dv01 / (notional * 1e-4) if notional > 0 else _NAN
    pv01     = dv01 / notional * 1_000_000 if notional > 0 else _NAN

    return dict(
        fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv_usd,
        par_rate=strike, clean_price=_NAN, accrued=_NAN,
        dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
        vega=vega, theta=_NAN, delta=_NAN,
        gamma=_NAN, rho=_NAN,
        cr01=_NAN, jump_to_default=_NAN, error="",
    )


def price_cap_floor(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    try:
        return _price_cap_floor(trade, curve_df)
    except Exception as e:
        return {**_NAN_ROW, "error": str(e)}
