"""pricing/optionable_bond_pricer.py — callable, putable, convertible, extendable, sinking-fund bond pricers."""

import math
from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW


def _price_callable_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a CallableFixedRateBond using HullWhite model + trinomial tree engine.

    legs[0] = BondLeg  (BOND)   — coupon schedule, notional, day count
    legs[1] = OptionLeg (OPTION) — option_type CALL/PUT, exercise_type, strike, vol

    Returns:
      swap_npv    = callable bond full (dirty) NPV in USD
      clean_price = clean price in USD (not % of par)
      par_rate    = YTM of the callable bond (from clean price)
      dv01        = dollar rate sensitivity to +1bp SOFR shift
      duration    = effective duration (years) = dv01 / (notional × 1e-4)
      delta       = embedded option value: bullet_npv − callable_npv for CALLABLE
                    callable_npv − bullet_npv for PUTABLE (always ≥ 0)
      vega        = d(NPV)/d(vol) via +1% HullWhite sigma bump
      convexity   = (P_up + P_down − 2P) / (P × (1bp)²)
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl = trade.bond_leg
    ol = trade.option_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(bl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc       = ql_dc.get(bl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    bdc      = ql_bdc.get(bl.bdc, ql.ModifiedFollowing)
    freq     = ql_freq.get(bl.frequency, ql.Semiannual)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)
    issue_dt     = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt     = to_ql(bl.start_date)
    mat_dt       = to_ql(bl.end_date)
    first_call_dt = to_ql(ol.start_date)

    notional    = bl.notional
    coupon_rate = bl.coupon_rate
    strike      = ol.strike      # % of par (1.00 = par)
    call_type   = trade.call_type  # "CALLABLE" or "PUTABLE"

    # HullWhite σ: derive from swaption vol surface (normal vol ≈ lognormal_vol × par_rate)
    # expiry = years to first call; tenor = remaining tenor after first call
    from models.market_data import MarketDataCache
    mkt = MarketDataCache().get_or_create(val_dt)
    expiry_y = max(0.08, (ol.start_date - val_dt).days / 365.0)
    bond_tenor_y = max(1.0, (bl.end_date - bl.start_date).days / 365.0)
    remain_y = max(1.0, bond_tenor_y - expiry_y)
    surf_vol = mkt.get_swaption_vol(expiry_y, remain_y)
    if surf_vol > 0:
        par_rate_approx = mkt._interp(mkt.par_rates, bond_tenor_y)
        hw_vol = max(0.005, surf_vol * par_rate_approx)  # lognormal→normal conversion
    else:
        hw_vol = ol.vol  # fall back to stored leg vol

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)

    # ── Bullet (non-callable) bond for comparison ──────────────────────────
    bullet = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bullet.setPricingEngine(ql.DiscountingBondEngine(sofr))
    bullet_npv   = bullet.NPV()
    bullet_clean = bullet.cleanPrice()

    # ── Build call/put schedule ────────────────────────────────────────────
    call_schedule = ql.CallabilitySchedule()
    call_price    = ql.BondPrice(strike * 100.0, ql.BondPrice.Clean)
    ql_call_type  = ql.Callability.Call if call_type == "CALLABLE" else ql.Callability.Put

    for sch_dt in schedule:
        if sch_dt >= first_call_dt and sch_dt <= mat_dt:
            call_schedule.append(ql.Callability(call_price, ql_call_type, sch_dt))

    # ── Callable bond ──────────────────────────────────────────────────────
    callable_bond = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )

    hw_model = ql.HullWhite(sofr, a=0.1, sigma=hw_vol)
    engine   = ql.TreeCallableFixedRateBondEngine(hw_model, 40)
    callable_bond.setPricingEngine(engine)

    npv_usd   = callable_bond.NPV()
    dirty_pct = callable_bond.dirtyPrice()
    clean_pct = callable_bond.cleanPrice()
    clean_usd = clean_pct / 100.0 * notional
    accrued   = (dirty_pct - clean_pct) / 100.0 * notional

    # YTM of callable bond (uses clean price from callable pricer)
    dc_yield = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = callable_bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        try:
            ytm = bullet.bondYield(dc_yield, ql.Compounded, freq)
        except Exception:
            ytm = _NAN

    # ── Embedded option value ──────────────────────────────────────────────
    # For CALLABLE: option hurts holder (reduces price) → delta > 0 for issuer
    # For PUTABLE:  option helps holder (increases price) → delta > 0 for holder
    if call_type == "CALLABLE":
        option_value = bullet_npv - npv_usd   # positive: call reduces bond value
    else:
        option_value = npv_usd - bullet_npv   # positive: put increases bond value

    # ── DV01 via +1bp parallel shift ──────────────────────────────────────
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)

    hw_b   = ql.HullWhite(sofr_b, a=0.1, sigma=hw_vol)
    eng_b  = ql.TreeCallableFixedRateBondEngine(hw_b, 40)
    cb_b   = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_b.setPricingEngine(eng_b)
    npv_p1   = cb_b.NPV()
    raw_dv01 = npv_p1 - npv_usd
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01     = sign * raw_dv01
    duration = dv01 / (notional * 1e-4)
    pv01     = dv01 / notional * 1_000_000

    # ── Convexity ─────────────────────────────────────────────────────────
    bumped_m1 = curve_df.copy(); bumped_m1["zero_rate"] -= 1e-4
    sofr_m1   = build_sofr_curve(ql_val, bumped_m1)
    hw_m1     = ql.HullWhite(sofr_m1, a=0.1, sigma=hw_vol)
    eng_m1    = ql.TreeCallableFixedRateBondEngine(hw_m1, 40)
    cb_m1     = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_m1.setPricingEngine(eng_m1)
    npv_m1    = cb_m1.NPV()
    convexity = (npv_m1 + npv_p1 - 2.0 * npv_usd) / (npv_usd * 1e-8) if npv_usd != 0.0 else _NAN

    # ── Vega: d(NPV)/d(sigma) via +1% sigma bump ──────────────────────────
    vol_bump = hw_vol + 0.01
    hw_vg    = ql.HullWhite(sofr, a=0.1, sigma=vol_bump)
    eng_vg   = ql.TreeCallableFixedRateBondEngine(hw_vg, 40)
    cb_vg    = ql.CallableFixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
        call_schedule,
    )
    cb_vg.setPricingEngine(eng_vg)
    vega = sign * (cb_vg.NPV() - npv_usd)   # signed by direction

    # ── Theta: 1-day price decay ──────────────────────────────────────────
    theta = _NAN
    try:
        ql_val_1d = ql_val + 1
        ql.Settings.instance().evaluationDate = ql_val_1d
        sofr_1d = build_sofr_curve(ql_val_1d, curve_df)
        hw_1d   = ql.HullWhite(sofr_1d, a=0.1, sigma=hw_vol)
        eng_1d  = ql.TreeCallableFixedRateBondEngine(hw_1d, 40)
        cb_1d   = ql.CallableFixedRateBond(
            bl.settlement_days, notional, schedule,
            [coupon_rate], dc, bdc, bl.redemption, issue_dt, call_schedule,
        )
        cb_1d.setPricingEngine(eng_1d)
        theta = sign * (cb_1d.NPV() - npv_usd)
    except Exception:
        pass
    finally:
        ql.Settings.instance().evaluationDate = ql_val   # always restore

    # ── Gamma and Rho reuse existing rate bumps ────────────────────────────
    # gamma = d²V/dr²  (dollar curvature of callable bond price w.r.t. rate)
    gamma = sign * (npv_m1 + npv_p1 - 2.0 * npv_usd) / (1e-4 ** 2)
    # rho   = dV/dr per 1% parallel SOFR shift  (= DV01 × 100)
    rho   = sign * raw_dv01 * 100.0

    return dict(fixed_npv=bullet_npv, float_npv=_NAN, swap_npv=npv_usd,
                par_rate=ytm, clean_price=clean_usd, accrued=accrued,
                dv01=dv01, duration=duration, pv01=pv01, convexity=convexity,
                vega=vega, theta=theta, delta=option_value,
                gamma=gamma, rho=rho,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_convertible_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a convertible bond using QuantLib BinomialConvertibleEngine (CRR 200 steps).
    Falls back to straight bond + Black-Scholes call proxy if ConvertibleFixedCouponBond fails.
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl  = trade.bond_leg
    ol  = trade.option_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    dc       = ql_dc.get(bl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    bdc      = ql_bdc.get(bl.bdc, ql.ModifiedFollowing)
    freq     = ql_freq.get(bl.frequency, ql.Semiannual)
    calendar = ql_cal.get(bl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))

    def to_ql(d): return ql.Date(d.day, d.month, d.year)

    issue_dt = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt = to_ql(bl.start_date)
    mat_dt   = to_ql(bl.end_date)

    notional      = bl.notional
    coupon_rate   = bl.coupon_rate
    stock_price   = getattr(ol, "initial_price", 100.0)
    conv_price    = getattr(ol, "strike", stock_price * 1.25)
    eq_vol        = getattr(ol, "vol", 0.25)
    div_yield     = getattr(ol, "dividend_yield", 0.013)
    conversion_ratio = notional / conv_price if conv_price > 0 else 1.0

    # Credit spread from market data (term-structure aware)
    from models.market_data import MarketDataCache, TICKER_TO_ENTITY
    mkt       = MarketDataCache().get_or_create(val_dt)
    ticker    = getattr(ol, "underlying_ticker", None) or getattr(ol, "ticker", "")
    entity    = TICKER_TO_ENTITY.get(ticker, "")
    tenor_y   = trade.tenor_y or max(1, (bl.end_date - bl.start_date).days // 365)
    if entity:
        credit_spread = mkt.get_credit_spread(entity, tenor_y)
    else:
        # Fall back to leg spread field, then a conservative IG default
        credit_spread = getattr(ol, "spread", getattr(ol, "credit_spread", 0.008))
        if credit_spread == 0.0:
            credit_spread = 0.008  # 80bps fallback for unknown issuer

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)

    # Straight bond for comparison
    bullet = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bullet.setPricingEngine(ql.DiscountingBondEngine(sofr))
    straight_npv   = bullet.NPV()
    straight_clean = bullet.cleanPrice()

    dc_yield = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = bullet.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        ytm = _NAN

    # Try QuantLib ConvertibleFixedCouponBond
    npv_usd = _NAN
    clean_usd = _NAN
    accrued = 0.0
    option_value = 0.0
    conv_bond_ok = False
    try:
        risky_spread_handle = ql.Handle(ql.SimpleQuote(credit_spread))
        spot_handle  = ql.Handle(ql.SimpleQuote(stock_price))
        div_handle   = ql.Handle(ql.FlatForward(ql_val, div_yield, ql.Actual365Fixed()))
        vol_handle   = ql.Handle(ql.BlackConstantVol(ql_val, ql.NullCalendar(),
                                                       eq_vol, ql.Actual365Fixed()))
        stock_process = ql.BlackScholesMertonProcess(spot_handle, div_handle, sofr, vol_handle)

        exercise    = ql.AmericanExercise(ql_val, mat_dt)
        dividends   = ql.DividendSchedule()
        callability = ql.CallabilitySchedule()

        conv_bond = ql.ConvertibleFixedCouponBond(
            exercise, conversion_ratio, dividends, callability,
            risky_spread_handle, issue_dt, bl.settlement_days,
            [coupon_rate], dc, schedule, bl.redemption,
        )
        engine = ql.BinomialConvertibleEngine(stock_process, "crr", 200)
        conv_bond.setPricingEngine(engine)

        npv_usd   = conv_bond.NPV()
        dirty_pct = conv_bond.dirtyPrice()
        clean_pct = conv_bond.cleanPrice()
        clean_usd = clean_pct / 100.0 * notional
        accrued   = (dirty_pct - clean_pct) / 100.0 * notional
        option_value = max(0.0, npv_usd - straight_npv)
        conv_bond_ok = True
    except Exception:
        conv_bond_ok = False
        # Fallback: straight bond + Black-Scholes call proxy
        T = max(0.01, (bl.end_date - trade.valuation_date).days / 365.25)
        r = ytm if ytm == ytm and math.isfinite(ytm) else 0.045
        if conv_price > 0 and eq_vol > 0:
            d1 = (math.log(stock_price / conv_price) + (r - div_yield + 0.5 * eq_vol**2) * T) / (eq_vol * math.sqrt(T))
            d2 = d1 - eq_vol * math.sqrt(T)
            from scipy.stats import norm as _norm
            call_per_share = (stock_price * math.exp(-div_yield * T) * _norm.cdf(d1)
                              - conv_price * math.exp(-r * T) * _norm.cdf(d2))
            option_value = max(0.0, conversion_ratio * call_per_share)
        else:
            option_value = 0.0
        npv_usd = straight_npv + option_value
        clean_usd = straight_clean / 100.0 * notional + option_value
        accrued = 0.0

    # DV01 via central difference on straight bond
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_up  = build_sofr_curve(ql_val, bumped)
    bullet_up = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bullet_up.setPricingEngine(ql.DiscountingBondEngine(sofr_up))
    npv_up = bullet_up.NPV()

    bumped_dn = curve_df.copy(); bumped_dn["zero_rate"] -= 1e-4
    sofr_dn = build_sofr_curve(ql_val, bumped_dn)
    bullet_dn = ql.FixedRateBond(
        bl.settlement_days, notional, schedule,
        [coupon_rate], dc, bdc, bl.redemption, issue_dt,
    )
    bullet_dn.setPricingEngine(ql.DiscountingBondEngine(sofr_dn))
    npv_dn = bullet_dn.NPV()

    dv01_raw = (npv_up - npv_dn) / 2.0
    sign = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01 = sign * abs(dv01_raw)
    duration = dv01 / (notional * 1e-4) if notional > 0 else _NAN
    pv01 = dv01 / (notional / 1e6) if notional > 0 else _NAN
    convexity = (npv_up + npv_dn - 2 * straight_npv) / (straight_npv * (1e-4) ** 2) if straight_npv > 0 else _NAN

    # ── Option Greeks (convertible-specific) ──────────────────────────────
    # equity_delta stored in `gamma` field: dV/dS — sensitivity to stock price
    # vega: dV/d(equity_vol)   theta: 1-day time decay   rho: rate sensitivity
    vega         = _NAN
    equity_delta = _NAN
    theta        = _NAN
    rho          = dv01 * 100.0   # per 1% parallel SOFR shift

    if conv_bond_ok:
        # Re-create lightweight handles (cheap QL constructors — used for Greek bumps)
        _rsk_h = ql.Handle(ql.SimpleQuote(credit_spread))
        _spot_h = ql.Handle(ql.SimpleQuote(stock_price))
        _div_h  = ql.Handle(ql.FlatForward(ql_val, div_yield, ql.Actual365Fixed()))
        _ex     = ql.AmericanExercise(ql_val, mat_dt)
        _divs   = ql.DividendSchedule()
        _calls  = ql.CallabilitySchedule()

        # Vega: d(NPV)/d(equity_vol) via +1% vol bump
        try:
            _vol_vg  = ql.Handle(ql.BlackConstantVol(ql_val, ql.NullCalendar(),
                                                     eq_vol + 0.01, ql.Actual365Fixed()))
            _proc_vg = ql.BlackScholesMertonProcess(_spot_h, _div_h, sofr, _vol_vg)
            _cb_vg   = ql.ConvertibleFixedCouponBond(
                _ex, conversion_ratio, _divs, _calls, _rsk_h,
                issue_dt, bl.settlement_days, [coupon_rate], dc, schedule, bl.redemption,
            )
            _cb_vg.setPricingEngine(ql.BinomialConvertibleEngine(_proc_vg, "crr", 200))
            vega = sign * (_cb_vg.NPV() - npv_usd)
        except Exception:
            pass

        # Equity delta: d(NPV)/d(S) via ±5% central difference (larger bump for binomial stability)
        try:
            h_s     = 0.05
            _vol_h  = ql.Handle(ql.BlackConstantVol(ql_val, ql.NullCalendar(),
                                                    eq_vol, ql.Actual365Fixed()))
            _sp_up  = ql.Handle(ql.SimpleQuote(stock_price * (1.0 + h_s)))
            _pr_up  = ql.BlackScholesMertonProcess(_sp_up, _div_h, sofr, _vol_h)
            _cb_up  = ql.ConvertibleFixedCouponBond(
                _ex, conversion_ratio, _divs, _calls, _rsk_h,
                issue_dt, bl.settlement_days, [coupon_rate], dc, schedule, bl.redemption,
            )
            _cb_up.setPricingEngine(ql.BinomialConvertibleEngine(_pr_up, "crr", 200))
            npv_s_up = _cb_up.NPV()

            _sp_dn  = ql.Handle(ql.SimpleQuote(stock_price * (1.0 - h_s)))
            _pr_dn  = ql.BlackScholesMertonProcess(_sp_dn, _div_h, sofr, _vol_h)
            _cb_dn  = ql.ConvertibleFixedCouponBond(
                _ex, conversion_ratio, _divs, _calls, _rsk_h,
                issue_dt, bl.settlement_days, [coupon_rate], dc, schedule, bl.redemption,
            )
            _cb_dn.setPricingEngine(ql.BinomialConvertibleEngine(_pr_dn, "crr", 200))
            npv_s_dn = _cb_dn.NPV()

            equity_delta = sign * (npv_s_up - npv_s_dn) / (2.0 * h_s * stock_price)
        except Exception:
            pass

        # Theta: 1-day time decay
        try:
            ql_val_1d = ql_val + 1
            ql.Settings.instance().evaluationDate = ql_val_1d
            sofr_1d  = build_sofr_curve(ql_val_1d, curve_df)
            _div_1d  = ql.Handle(ql.FlatForward(ql_val_1d, div_yield, ql.Actual365Fixed()))
            _vol_1d  = ql.Handle(ql.BlackConstantVol(ql_val_1d, ql.NullCalendar(),
                                                     eq_vol, ql.Actual365Fixed()))
            _proc_1d = ql.BlackScholesMertonProcess(_spot_h, _div_1d, sofr_1d, _vol_1d)
            _cb_1d   = ql.ConvertibleFixedCouponBond(
                _ex, conversion_ratio, _divs, _calls, _rsk_h,
                issue_dt, bl.settlement_days, [coupon_rate], dc, schedule, bl.redemption,
            )
            _cb_1d.setPricingEngine(ql.BinomialConvertibleEngine(_proc_1d, "crr", 200))
            theta = sign * (_cb_1d.NPV() - npv_usd)
        except Exception:
            pass
        finally:
            ql.Settings.instance().evaluationDate = ql_val   # always restore

    else:
        # Fallback: Black-Scholes analytical Greeks on the embedded call option
        T_cv = max(0.01, (bl.end_date - trade.valuation_date).days / 365.25)
        r_cv = ytm if (ytm == ytm and math.isfinite(ytm)) else 0.045
        if conv_price > 0 and eq_vol > 0 and stock_price > 0:
            import math as _m
            from scipy.stats import norm as _norm
            sqrt_T  = _m.sqrt(T_cv)
            d1 = (_m.log(stock_price / conv_price) + (r_cv - div_yield + 0.5 * eq_vol**2) * T_cv) / (eq_vol * sqrt_T)
            d2 = d1 - eq_vol * sqrt_T
            n_d1  = _norm.pdf(d1)
            exp_q = _m.exp(-div_yield * T_cv)
            exp_r = _m.exp(-r_cv * T_cv)
            # Per-share analytical Greeks, then scale by conversion_ratio
            equity_delta = sign * conversion_ratio * exp_q * _norm.cdf(d1)
            vega         = sign * conversion_ratio * stock_price * exp_q * n_d1 * sqrt_T * 0.01  # per 1% vol
            theta        = sign * conversion_ratio * (
                -stock_price * exp_q * n_d1 * eq_vol / (2.0 * sqrt_T)
                - r_cv * conv_price * exp_r * _norm.cdf(d2)
                + div_yield * stock_price * exp_q * _norm.cdf(d1)
            ) / 365.0   # per calendar day

    return dict(
        swap_npv    = npv_usd,
        clean_price = clean_usd,
        accrued     = accrued,
        par_rate    = ytm,
        dv01        = dv01,
        duration    = duration,
        pv01        = pv01,
        convexity   = convexity,
        delta       = option_value,
        vega        = vega,
        theta       = theta,
        gamma       = equity_delta,   # dV/dS: equity sensitivity of the convertible
        rho         = rho,
        fixed_npv   = straight_npv,
        float_npv   = option_value,
        cr01        = _NAN,
        jump_to_default = _NAN,
        error       = "",
    )


def _price_extendable_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an extendable bond using HullWhite putable bond framework."""
    return _price_callable_bond(trade, curve_df)


def _price_sinking_fund_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price a sinking fund bond using QuantLib AmortizingFixedRateBond."""
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    bl     = trade.bond_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr     = build_sofr_curve(ql_val, curve_df)
    calendar = ql_cal.get(bl.calendar, ql.UnitedStates(ql.UnitedStates.GovernmentBond))
    dc       = ql_dc.get(bl.day_count, ql.Thirty360(ql.Thirty360.BondBasis))
    bdc      = ql_bdc.get(bl.bdc, ql.ModifiedFollowing)
    freq     = ql_freq.get(bl.frequency, ql.Semiannual)

    def to_ql(d): return ql.Date(d.day, d.month, d.year)

    issue_dt = to_ql(bl.issue_date if bl.issue_date else bl.start_date)
    start_dt = to_ql(bl.start_date)
    mat_dt   = to_ql(bl.end_date)

    notional      = bl.notional
    coupon_rate   = bl.coupon_rate
    sinking_pct   = getattr(trade, "sinking_pct_per_period", 0.10)

    schedule = ql.Schedule(start_dt, mat_dt, ql.Period(freq), calendar,
                           bdc, bdc, ql.DateGeneration.Backward, False)

    n_periods = len(schedule) - 1
    n_per_year = 2 if bl.frequency == "SEMIANNUAL" else 1
    notionals = []
    for j in range(n_periods):
        year_num = j // n_per_year
        amortized = min(1.0, year_num * sinking_pct)
        remaining = max(0.0, 1.0 - amortized)
        notionals.append(notional * remaining)

    try:
        amort_bond = ql.AmortizingFixedRateBond(
            bl.settlement_days, notionals, schedule,
            [coupon_rate], dc, bdc, issue_dt,
        )
        amort_bond.setPricingEngine(ql.DiscountingBondEngine(sofr))

        npv_usd   = amort_bond.NPV()
        dirty_pct = amort_bond.dirtyPrice()
        clean_pct = amort_bond.cleanPrice()
        clean_usd = clean_pct / 100.0 * notional
        accrued   = (dirty_pct - clean_pct) / 100.0 * notional
    except Exception as e:
        return {**_NAN_ROW, "error": f"SinkingFund pricing failed: {e}"}

    dc_yield = ql.ActualActual(ql.ActualActual.Bond)
    try:
        ytm = amort_bond.bondYield(dc_yield, ql.Compounded, freq)
    except Exception:
        ytm = _NAN

    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_up = build_sofr_curve(ql_val, bumped)
    ab_up = ql.AmortizingFixedRateBond(
        bl.settlement_days, notionals, schedule, [coupon_rate], dc, bdc, issue_dt)
    ab_up.setPricingEngine(ql.DiscountingBondEngine(sofr_up))
    npv_up = ab_up.NPV()

    bumped_dn = curve_df.copy(); bumped_dn["zero_rate"] -= 1e-4
    sofr_dn = build_sofr_curve(ql_val, bumped_dn)
    ab_dn = ql.AmortizingFixedRateBond(
        bl.settlement_days, notionals, schedule, [coupon_rate], dc, bdc, issue_dt)
    ab_dn.setPricingEngine(ql.DiscountingBondEngine(sofr_dn))
    npv_dn = ab_dn.NPV()

    dv01_raw = (npv_up - npv_dn) / 2.0
    sign = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0
    dv01 = sign * abs(dv01_raw)
    duration = dv01 / (notional * 1e-4) if notional > 0 else _NAN
    pv01 = dv01 / (notional / 1e6) if notional > 0 else _NAN
    convexity_val = (npv_up + npv_dn - 2 * npv_usd) / (npv_usd * (1e-4) ** 2) if npv_usd > 0 else _NAN

    return dict(
        swap_npv    = npv_usd,
        clean_price = clean_usd,
        accrued     = accrued,
        par_rate    = ytm,
        dv01        = dv01,
        duration    = duration,
        pv01        = pv01,
        convexity   = abs(convexity_val) if convexity_val == convexity_val else _NAN,
        delta       = _NAN,
        vega        = _NAN,
        theta       = _NAN,
        gamma       = _NAN,
        rho         = _NAN,
        fixed_npv   = _NAN,
        float_npv   = _NAN,
        cr01        = _NAN,
        jump_to_default = _NAN,
        error       = "",
    )


def _price_optionable_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Dispatch to the correct sub-pricer based on bond_subtype."""
    subtype = getattr(trade, "bond_subtype", "CALLABLE")
    if subtype in ("CALLABLE", "PUTABLE"):
        return _price_callable_bond(trade, curve_df)
    elif subtype == "CONVERTIBLE":
        return _price_convertible_bond(trade, curve_df)
    elif subtype == "EXTENDABLE":
        return _price_extendable_bond(trade, curve_df)
    elif subtype == "SINKING_FUND":
        return _price_sinking_fund_bond(trade, curve_df)
    else:
        return _price_callable_bond(trade, curve_df)


def price_optionable_bond(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an OptionableBond — dispatches by bond_subtype."""
    return _price_optionable_bond(trade, curve_df)
