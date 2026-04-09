"""pricing/equity_pricer.py — equity swap and equity option pricers."""

import math as _math
from typing import Dict, Any

import numpy as np
import pandas as pd
import QuantLib as ql

from pricing.utils import build_sofr_curve, _ql_maps, _NAN, _NAN_F, _NAN_ROW
from models.enums import TradeDirection

_EQ_OPTION_VOLS = {
    "SPY":   {0.25: 0.14, 0.5: 0.15, 1: 0.16, 2: 0.18},
    "QQQ":   {0.25: 0.18, 0.5: 0.19, 1: 0.20, 2: 0.22},
    "AAPL":  {0.25: 0.25, 0.5: 0.26, 1: 0.27, 2: 0.28},
    "MSFT":  {0.25: 0.22, 0.5: 0.23, 1: 0.24, 2: 0.25},
    "GOOGL": {0.25: 0.24, 0.5: 0.25, 1: 0.26, 2: 0.27},
    "SX5E":  {0.25: 0.16, 0.5: 0.17, 1: 0.18, 2: 0.20},
    "NKY":   {0.25: 0.18, 0.5: 0.19, 1: 0.20, 2: 0.22},
    "IWM":   {0.25: 0.20, 0.5: 0.21, 1: 0.22, 2: 0.24},
    "FTSE":  {0.25: 0.15, 0.5: 0.16, 1: 0.17, 2: 0.19},
}


def _price_equity_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analytical single-period equity total-return swap pricer.

    NPV = sign * (Notional*exp(-q*T) - Notional*exp(-(r+s)*T))
    sign = +1 if LONG (receive equity), -1 if SHORT.
    """
    from datetime import date

    el     = trade.equity_leg
    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    sofr = build_sofr_curve(ql_val, curve_df)

    end_dt = el.end_date
    T = (date(end_dt.year, end_dt.month, end_dt.day) - val_dt).days / 365.0
    if T <= 0:
        return {**_NAN_ROW, "error": "Matured trade"}

    ql_mat = ql.Date(end_dt.day, end_dt.month, end_dt.year)
    r      = sofr.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()

    q        = el.dividend_yield
    s        = el.funding_spread
    notional = el.notional
    sign     = -1.0 if getattr(trade.direction, 'value', trade.direction) == "SHORT" else 1.0

    pv_equity  = notional * _math.exp(-q * T)
    pv_funding = notional * _math.exp(-(r + s) * T)
    npv        = sign * (pv_equity - pv_funding)

    # Par rate: dividend yield (the equity yield component)
    par_rate = q

    # DV01: bump SOFR by +1bp
    bumped = curve_df.copy(); bumped["zero_rate"] += 1e-4
    sofr_b = build_sofr_curve(ql_val, bumped)
    r_b    = sofr_b.zeroRate(ql_mat, ql.Actual365Fixed(), ql.Continuous).rate()
    pv_fund_b = notional * _math.exp(-(r_b + s) * T)
    npv_b     = sign * (pv_equity - pv_fund_b)
    dv01      = npv_b - npv
    duration  = dv01 / (notional * 1e-4)
    pv01      = dv01 / notional * 1_000_000

    # Delta: $ per 1% move in underlying (TRS has 100% delta)
    delta = sign * notional * 0.01

    # Theta: reprice at T - 1 day
    T_1d      = max(1e-4, T - 1.0 / 365.0)
    pv_eq_1d  = notional * _math.exp(-q * T_1d)
    pv_fun_1d = notional * _math.exp(-(r + s) * T_1d)
    npv_1d    = sign * (pv_eq_1d - pv_fun_1d)
    theta     = npv_1d - npv

    return dict(fixed_npv=_NAN, float_npv=_NAN, swap_npv=npv,
                par_rate=par_rate, clean_price=_NAN, accrued=_NAN,
                dv01=dv01, duration=duration, pv01=pv01, convexity=_NAN,
                vega=_NAN, theta=theta, delta=delta,
                gamma=_NAN, rho=_NAN,
                cr01=_NAN, jump_to_default=_NAN, error="")


def _price_equity_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Price a vanilla equity option using Black-Scholes (European) or
    finite-difference PDE engine (American). Greeks: delta, gamma, vega, theta, rho.
    """
    ql_freq, ql_dc, ql_bdc, ql_cal = _ql_maps()

    ol = trade.option_leg
    if ol is None:
        return {**_NAN_ROW, "error": "No EquityOptionLeg found"}

    val_dt = trade.valuation_date
    ql_val = ql.Date(val_dt.day, val_dt.month, val_dt.year)
    ql.Settings.instance().evaluationDate = ql_val

    exp_dt = ol.end_date
    T_days = (exp_dt - val_dt).days
    T      = T_days / 365.0
    if T <= 0:
        return {**_NAN_ROW, "error": f"Expired option (T={T:.3f})"}

    sofr = build_sofr_curve(ql_val, curve_df)
    ql_exp = ql.Date(exp_dt.day, exp_dt.month, exp_dt.year)
    r = sofr.zeroRate(ql_exp, ql.Actual365Fixed(), ql.Continuous, ql.Annual).rate()

    S        = ol.initial_price
    K        = ol.strike
    q        = ol.dividend_yield
    vol      = ol.vol
    notional = ol.notional

    ql_opt  = ql.Option.Call if ol.option_type == "CALL" else ql.Option.Put
    payoff  = ql.PlainVanillaPayoff(ql_opt, K)

    if ol.exercise_type == "EUROPEAN":
        exercise = ql.EuropeanExercise(ql_exp)
    else:
        ql_start = ql_val + ql.Period(1, ql.Days)
        exercise = ql.AmericanExercise(ql_start, ql_exp)

    option = ql.VanillaOption(payoff, exercise)

    spot_q  = ql.SimpleQuote(S)
    spot_h  = ql.QuoteHandle(spot_q)
    rf_ts   = ql.FlatForward(ql_val, r, ql.Actual365Fixed())
    rf_q    = ql.YieldTermStructureHandle(rf_ts)
    div_ts  = ql.FlatForward(ql_val, q, ql.Actual365Fixed())
    div_q   = ql.YieldTermStructureHandle(div_ts)
    vol_ts  = ql.BlackConstantVol(ql_val, ql.NullCalendar(), vol, ql.Actual365Fixed())
    vol_q   = ql.BlackVolTermStructureHandle(vol_ts)
    process = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q, vol_q)

    if ol.exercise_type == "EUROPEAN":
        engine = ql.AnalyticEuropeanEngine(process)
    else:
        engine = ql.FdBlackScholesVanillaEngine(process, 100, 200)

    option.setPricingEngine(engine)

    try:
        unit_premium = option.NPV()
    except Exception as e:
        return {**_NAN_ROW, "error": f"Pricing failed: {e}"}

    n_shares = notional / S
    sign = 1.0 if trade.direction == TradeDirection.BUY else -1.0
    premium = sign * unit_premium * n_shares

    def _greek(fn):
        try:
            v = fn()
            return v if v is not None and not _math.isnan(v) else _NAN
        except Exception:
            return _NAN

    unit_delta = _greek(option.delta)
    unit_gamma = _greek(option.gamma)
    unit_theta = _greek(option.theta)

    # vega / rho not always available for FD engine — fall back to bump-reprice
    unit_vega = _greek(option.vega)
    if _math.isnan(unit_vega):
        try:
            vol_bump = vol + 0.01
            vol_ts2  = ql.BlackConstantVol(ql_val, ql.NullCalendar(), vol_bump, ql.Actual365Fixed())
            vol_q2   = ql.BlackVolTermStructureHandle(vol_ts2)
            p2       = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q, vol_q2)
            opt2     = ql.VanillaOption(payoff, exercise)
            if ol.exercise_type == "EUROPEAN":
                opt2.setPricingEngine(ql.AnalyticEuropeanEngine(p2))
            else:
                opt2.setPricingEngine(ql.FdBlackScholesVanillaEngine(p2, 100, 200))
            unit_vega = opt2.NPV() - unit_premium   # per 1% vol move
        except Exception:
            unit_vega = _NAN

    unit_rho = _greek(option.rho)
    if _math.isnan(unit_rho):
        try:
            r_bump  = r + 0.0001
            rf_q2   = ql.YieldTermStructureHandle(
                          ql.FlatForward(ql_val, r_bump, ql.Actual365Fixed()))
            p2      = ql.BlackScholesMertonProcess(spot_h, div_q, rf_q2, vol_q)
            opt2    = ql.VanillaOption(payoff, exercise)
            if ol.exercise_type == "EUROPEAN":
                opt2.setPricingEngine(ql.AnalyticEuropeanEngine(p2))
            else:
                opt2.setPricingEngine(ql.FdBlackScholesVanillaEngine(p2, 100, 200))
            unit_rho = (opt2.NPV() - unit_premium) / 0.0001   # per unit rate move
        except Exception:
            unit_rho = _NAN

    # delta fallback: finite difference on spot if FD engine didn't provide it
    if _math.isnan(unit_delta):
        try:
            h = S * 0.001
            def _npv_at(s):
                sq = ql.QuoteHandle(ql.SimpleQuote(s))
                pr = ql.BlackScholesMertonProcess(sq, div_q, rf_q, vol_q)
                o  = ql.VanillaOption(payoff, exercise)
                if ol.exercise_type == "EUROPEAN":
                    o.setPricingEngine(ql.AnalyticEuropeanEngine(pr))
                else:
                    o.setPricingEngine(ql.FdBlackScholesVanillaEngine(pr, 50, 100))
                return o.NPV()
            unit_delta = (_npv_at(S + h) - _npv_at(S - h)) / (2 * h)
        except Exception:
            unit_delta = _NAN

    def _safe(v):
        return v if not _math.isnan(v) else _NAN

    delta = sign * (_safe(unit_delta) * n_shares)         if not _math.isnan(unit_delta) else _NAN
    gamma = sign * (_safe(unit_gamma) * n_shares)         if not _math.isnan(unit_gamma) else _NAN
    vega  = sign * (_safe(unit_vega)  * n_shares * 0.01)  if not _math.isnan(unit_vega)  else _NAN
    theta = sign * (_safe(unit_theta) * n_shares)         if not _math.isnan(unit_theta) else _NAN
    rho   = sign * (_safe(unit_rho)   * n_shares * 0.0001) if not _math.isnan(unit_rho) else _NAN
    dv01  = rho

    return dict(
        fixed_npv=_NAN, float_npv=_NAN, swap_npv=premium,
        par_rate=_NAN, clean_price=_NAN, accrued=_NAN,
        dv01=dv01, duration=_NAN, pv01=_NAN, convexity=_NAN,
        vega=vega, theta=theta, delta=delta,
        gamma=gamma, rho=rho,
        cr01=_NAN, jump_to_default=_NAN, error="",
    )


def price_equity_swap(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an EquitySwap trade."""
    return _price_equity_swap(trade, curve_df)


def price_equity_option(trade, curve_df: pd.DataFrame) -> Dict[str, Any]:
    """Price an EquityOptionTrade."""
    return _price_equity_option(trade, curve_df)
