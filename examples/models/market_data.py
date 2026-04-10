"""
models/market_data.py
=====================
MarketDataSnapshot — frozen dataclass holding all market observables for one valuation date.
MarketDataCache    — module-level singleton for cached access.

Market data coverage (all synthetic, calibrated to 2025-01-15):
  sofr_curve          : USD SOFR OIS zero curve (11 tenors 3M–30Y)
  par_rates           : USD par swap rates (9 tenors 1Y–30Y)
  foreign_curves      : OIS zero curves for EUR (ESTR), GBP (SONIA), JPY (TONAR)
  xccy_basis          : Cross-currency basis spreads vs USD {pair: {tenor_str: spread}}
  swaption_vol_surface: Normal vol surface {expiry_y_str: {swap_y_str: vol}} 8×8 grid
  credit_spreads      : Flat 5Y spread per entity (backward compat)
  credit_curves       : Term-structure credit spreads {entity: {tenor_str: spread}}
  equity_prices       : Spot prices for 10 tickers
  equity_divylds      : Flat continuous dividend yield per ticker
  equity_vol_surface  : Vol surface {ticker: {expiry_str: {moneyness_str: vol}}}
  cap_floor_vols      : Cap/floor normal vol cube {expiry_str: {strike_str: vol}}
  fx_rates            : Spot FX {pair: rate}
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional
import QuantLib as ql
import pandas as pd


@dataclass(frozen=True)
class MarketDataSnapshot:
    valuation_date:      date
    sofr_curve:          Dict[str, float]   # {tenor_y_str: zero_rate}
    par_rates:           Dict[str, float]   # {tenor_y_str: par_rate}
    swaption_vols:       Dict[str, float]   # legacy flat: {"expiry_y,swap_y": vol}
    credit_spreads:      Dict[str, float]   # legacy flat 5Y spread per entity
    equity_prices:       Dict[str, float]   # {ticker: spot}
    equity_divylds:      Dict[str, float]   # {ticker: div_yield}
    fx_rates:            Dict[str, float] = field(default_factory=dict)

    # ── new rich market data ──────────────────────────────────────────────────
    # Foreign OIS zero curves:  {"EUR": {"0.25": 0.035, "1": 0.032, ...}, ...}
    foreign_curves:      Dict[str, Dict[str, float]] = field(default_factory=dict)
    # XCCY basis spreads (decimal): {"EURUSD": {"1": -0.0015, "5": -0.0020, ...}}
    xccy_basis:          Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Credit term structure:   {"FORD MOTOR CO": {"1": 0.015, "3": 0.025, "5": 0.030, ...}}
    credit_curves:       Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Swaption vol surface:    {"1": {"1": 0.50, "2": 0.46, "5": 0.38, ...}, "2": {...}, ...}
    # outer key = expiry_y str, inner key = swap_y str
    swaption_vol_surface: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Equity vol surface:  {"SPY": {"0.25": {"0.90": 0.17, "1.00": 0.14, "1.10": 0.13}, ...}}
    # outer key = ticker, middle = expiry_y str, inner = moneyness (K/S) str
    equity_vol_surface:  Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    # Cap/floor normal vol cube: {"1": {"0.03": 0.55, "0.04": 0.52, ...}, "5": {...}, ...}
    # outer key = expiry_y str, inner = strike str
    cap_floor_vols:      Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── SOFR curve helpers ────────────────────────────────────────────────────

    def to_curve_df(self) -> pd.DataFrame:
        """Return sorted SOFR zero curve as DataFrame[tenor_y, zero_rate]."""
        rows = sorted((float(k), v) for k, v in self.sofr_curve.items())
        return pd.DataFrame(rows, columns=["tenor_y", "zero_rate"])

    def build_sofr_curve(self, ql_val: ql.Date) -> ql.YieldTermStructureHandle:
        """Build a QuantLib ZeroCurve from the stored SOFR data."""
        return self._build_zero_curve(self.sofr_curve, ql_val)

    def _build_zero_curve(self, curve_dict: Dict[str, float],
                          ql_val: ql.Date,
                          basis_dict: Optional[Dict[str, float]] = None
                          ) -> ql.YieldTermStructureHandle:
        """Build a QuantLib ZeroCurve from a {tenor_str: zero_rate} dict.
        Optional basis_dict adds a tenor-interpolated spread to each pillar.
        """
        dc    = ql.Actual365Fixed()
        items = sorted(curve_dict.items(), key=lambda x: float(x[0]))
        dates = [ql_val]
        rates = [float(items[0][1])]
        for k, r in items:
            months = max(1, int(round(float(k) * 12)))
            basis  = 0.0
            if basis_dict:
                basis = self._interp(basis_dict, float(k))
            dates.append(ql_val + ql.Period(months, ql.Months))
            rates.append(float(r) + basis)
        curve = ql.ZeroCurve(dates, rates, dc, ql.NullCalendar(),
                             ql.Linear(), ql.Continuous, ql.Annual)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def build_foreign_curve(self, ccy: str,
                            ql_val: ql.Date,
                            include_basis: bool = True
                            ) -> ql.YieldTermStructureHandle:
        """Build foreign OIS curve, optionally adjusted by xccy basis vs USD."""
        fc = self.foreign_curves.get(ccy, self.sofr_curve)
        basis = {}
        if include_basis:
            pair = f"{ccy}USD" if ccy != "USD" else ""
            basis = self.xccy_basis.get(pair, {})
        return self._build_zero_curve(fc, ql_val, basis_dict=basis or None)

    # ── par rate helpers ──────────────────────────────────────────────────────

    def get_par_rate(self, tenor_y: int) -> float:
        return self._interp(self.par_rates, float(tenor_y), default=0.045)

    # ── vol helpers ───────────────────────────────────────────────────────────

    def get_swaption_vol(self, expiry_y: float, swap_y: float) -> float:
        """Bilinear interpolation on the swaption vol surface."""
        surf = self.swaption_vol_surface
        if not surf:
            key = f"{int(expiry_y)},{int(swap_y)}"
            return self.swaption_vols.get(key, 0.40)
        expiries = sorted(float(k) for k in surf)
        swap_tns = sorted(float(k) for k in next(iter(surf.values())))
        exp_lo, exp_hi = self._bracket(expiries, expiry_y)
        swp_lo, swp_hi = self._bracket(swap_tns, swap_y)
        def _v(e, s):
            return surf.get(str(e), surf.get(f"{int(e)}", {})).get(
                str(s), surf.get(f"{int(e)}", {}).get(f"{int(s)}", 0.40))
        v00 = _v(exp_lo, swp_lo); v01 = _v(exp_lo, swp_hi)
        v10 = _v(exp_hi, swp_lo); v11 = _v(exp_hi, swp_hi)
        we  = (expiry_y - exp_lo) / (exp_hi - exp_lo + 1e-12) if exp_hi != exp_lo else 0.0
        ws  = (swap_y   - swp_lo) / (swp_hi - swp_lo + 1e-12) if swp_hi != swp_lo else 0.0
        return (1-we)*(1-ws)*v00 + (1-we)*ws*v01 + we*(1-ws)*v10 + we*ws*v11

    def get_equity_vol(self, ticker: str, expiry_y: float, moneyness: float = 1.0) -> float:
        """Bilinear interpolation on the equity vol surface."""
        surf = self.equity_vol_surface.get(ticker)
        if not surf:
            return 0.20
        expiries = sorted(float(k) for k in surf)
        exp_lo, exp_hi = self._bracket(expiries, expiry_y)
        def _slice(e):
            return surf.get(str(e), surf.get(f"{e:.2f}", {}))
        sl_lo = _slice(exp_lo); sl_hi = _slice(exp_hi)
        mnkeys_lo = sorted(float(k) for k in sl_lo) if sl_lo else [1.0]
        mnkeys_hi = sorted(float(k) for k in sl_hi) if sl_hi else [1.0]
        mn_lo, mn_hi = self._bracket(mnkeys_lo, moneyness)
        def _get(sl, m):
            v = sl.get(f"{m:.2f}", sl.get(str(m)))
            return v if v is not None else (sl.get(f"{1.0:.2f}") or 0.20)
        v00 = _get(sl_lo, mn_lo); v01 = _get(sl_lo, mn_hi)
        v10 = _get(sl_hi, mn_lo); v11 = _get(sl_hi, mn_hi)
        we  = (expiry_y  - exp_lo) / (exp_hi - exp_lo + 1e-12) if exp_hi != exp_lo else 0.0
        wm  = (moneyness - mn_lo)  / (mn_hi  - mn_lo  + 1e-12) if mn_hi  != mn_lo  else 0.0
        return (1-we)*(1-wm)*v00 + (1-we)*wm*v01 + we*(1-wm)*v10 + we*wm*v11

    def get_cap_floor_vol(self, expiry_y: float, strike: float) -> float:
        """Bilinear interpolation on the cap/floor vol cube."""
        cube = self.cap_floor_vols
        if not cube:
            return 0.50
        expiries = sorted(float(k) for k in cube)
        exp_lo, exp_hi = self._bracket(expiries, expiry_y)
        def _slice(e):
            return cube.get(str(int(e)), cube.get(str(e), {}))
        sl_lo = _slice(exp_lo); sl_hi = _slice(exp_hi)
        strikes_lo = sorted(float(k) for k in sl_lo) if sl_lo else [0.04]
        strikes_hi = sorted(float(k) for k in sl_hi) if sl_hi else [0.04]
        sk_lo, sk_hi = self._bracket(strikes_lo, strike)
        def _g(sl, k):
            return sl.get(f"{k:.2f}", sl.get(str(k), 0.50))
        v00 = _g(sl_lo, sk_lo); v01 = _g(sl_lo, sk_hi)
        v10 = _g(sl_hi, sk_lo); v11 = _g(sl_hi, sk_hi)
        we  = (expiry_y - exp_lo) / (exp_hi - exp_lo + 1e-12) if exp_hi != exp_lo else 0.0
        wk  = (strike   - sk_lo)  / (sk_hi  - sk_lo  + 1e-12) if sk_hi  != sk_lo  else 0.0
        return (1-we)*(1-wk)*v00 + (1-we)*wk*v01 + we*(1-wk)*v10 + we*wk*v11

    def get_credit_spread(self, entity: str, tenor_y: float = 5.0) -> float:
        """Return credit spread for entity at given tenor (interpolated from credit_curves)."""
        tc = self.credit_curves.get(entity)
        if tc:
            return self._interp(tc, tenor_y, default=self.credit_spreads.get(entity, 0.01))
        return self.credit_spreads.get(entity, 0.01)

    def build_hazard_curve(self, entity: str,
                           ql_val: ql.Date) -> ql.DefaultProbabilityTermStructureHandle:
        """Build a piecewise flat hazard rate term structure from credit_curves[entity]."""
        tc  = self.credit_curves.get(entity)
        recovery = 0.40
        if not tc:
            spread = self.credit_spreads.get(entity, 0.01)
            h = spread / (1.0 - recovery)
            return ql.DefaultProbabilityTermStructureHandle(
                ql.FlatHazardRate(ql_val, ql.QuoteHandle(ql.SimpleQuote(h)),
                                  ql.Actual365Fixed()))
        dc    = ql.Actual365Fixed()
        dates = [ql_val]
        hazards = []
        for k, spread in sorted(tc.items(), key=lambda x: float(x[0])):
            months = max(1, int(round(float(k) * 12)))
            dates.append(ql_val + ql.Period(months, ql.Months))
            hazards.append(float(spread) / (1.0 - recovery))
        # QL needs dates and hazards of same length (one hazard per interval)
        curve = ql.HazardRateCurve(dates, hazards, dc)
        curve.enableExtrapolation()
        return ql.DefaultProbabilityTermStructureHandle(curve)

    # ── generic interpolation utilities ──────────────────────────────────────

    @staticmethod
    def _interp(d: Dict[str, float], x: float, default: float = 0.0) -> float:
        """Linear interpolation / extrapolation on a {str_key: float_val} dict."""
        if not d:
            return default
        items = sorted((float(k), v) for k, v in d.items())
        xs, ys = zip(*items)
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i+1]:
                t = (x - xs[i]) / (xs[i+1] - xs[i])
                return ys[i] + t * (ys[i+1] - ys[i])
        return default

    @staticmethod
    def _bracket(vals: list, x: float):
        """Return (lo, hi) surrounding x from sorted list, clamped to edges."""
        if x <= vals[0]:  return vals[0],  vals[0]
        if x >= vals[-1]: return vals[-1], vals[-1]
        for i in range(len(vals) - 1):
            if vals[i] <= x <= vals[i+1]:
                return vals[i], vals[i+1]
        return vals[-1], vals[-1]

    # ── serialization ─────────────────────────────────────────────────────────

    def toJson(self) -> str:
        d = {
            "valuation_date":      self.valuation_date.isoformat(),
            "sofr_curve":          self.sofr_curve,
            "par_rates":           self.par_rates,
            "swaption_vols":       self.swaption_vols,
            "credit_spreads":      self.credit_spreads,
            "equity_prices":       self.equity_prices,
            "equity_divylds":      self.equity_divylds,
            "fx_rates":            self.fx_rates,
            "foreign_curves":      self.foreign_curves,
            "xccy_basis":          self.xccy_basis,
            "credit_curves":       self.credit_curves,
            "swaption_vol_surface": self.swaption_vol_surface,
            "equity_vol_surface":  self.equity_vol_surface,
            "cap_floor_vols":      self.cap_floor_vols,
        }
        return json.dumps(d)

    @classmethod
    def fromJson(cls, js: str) -> "MarketDataSnapshot":
        d = json.loads(js)
        return cls(
            valuation_date       = date.fromisoformat(d["valuation_date"]),
            sofr_curve           = d["sofr_curve"],
            par_rates            = d["par_rates"],
            swaption_vols        = d["swaption_vols"],
            credit_spreads       = d["credit_spreads"],
            equity_prices        = d["equity_prices"],
            equity_divylds       = d["equity_divylds"],
            fx_rates             = d.get("fx_rates", {}),
            foreign_curves       = d.get("foreign_curves", {}),
            xccy_basis           = d.get("xccy_basis", {}),
            credit_curves        = d.get("credit_curves", {}),
            swaption_vol_surface = d.get("swaption_vol_surface", {}),
            equity_vol_surface   = d.get("equity_vol_surface", {}),
            cap_floor_vols       = d.get("cap_floor_vols", {}),
        )


# ── default market data for 2025-01-15 ───────────────────────────────────────

def make_default_snapshot(valuation_date: date = date(2025, 1, 15)) -> MarketDataSnapshot:
    """Return the full synthetic market data snapshot for 2025-01-15."""

    # ── USD SOFR OIS zero curve ───────────────────────────────────────────────
    sofr_curve = {
        "0.25": 0.0530, "0.5": 0.0525, "1": 0.0510, "2": 0.0490,
        "3": 0.0470, "5": 0.0450, "7": 0.0440, "10": 0.0430,
        "15": 0.0425, "20": 0.0420, "30": 0.0415,
    }

    # ── Foreign OIS zero curves ───────────────────────────────────────────────
    # EUR ESTR: ECB cutting cycle, short end ~3.5%, long end ~3.0%
    eur_estr = {
        "0.25": 0.0350, "0.5": 0.0345, "1": 0.0330, "2": 0.0315,
        "3": 0.0305, "5": 0.0295, "7": 0.0290, "10": 0.0285,
        "15": 0.0282, "20": 0.0280, "30": 0.0278,
    }
    # GBP SONIA: BoE hold, short end ~4.7%, normalising to ~4.0% long end
    gbp_sonia = {
        "0.25": 0.0470, "0.5": 0.0465, "1": 0.0455, "2": 0.0440,
        "3": 0.0425, "5": 0.0410, "7": 0.0400, "10": 0.0395,
        "15": 0.0390, "20": 0.0388, "30": 0.0385,
    }
    # JPY TONAR: BoJ normalisation just starting, near-zero short end
    jpy_tonar = {
        "0.25": 0.0010, "0.5": 0.0015, "1": 0.0025, "2": 0.0060,
        "3": 0.0090, "5": 0.0130, "7": 0.0160, "10": 0.0185,
        "15": 0.0205, "20": 0.0215, "30": 0.0220,
    }

    # ── Cross-currency basis spreads (decimal, vs USD) ────────────────────────
    # Negative = foreign ccy is expensive to borrow vs USD in xccy markets
    # EURUSD basis: ~-15 to -25bps  |  GBPUSD: ~-20 to -30bps  |  USDJPY: +5 to +20bps
    xccy_basis = {
        "EURUSD": {"1": -0.0010, "2": -0.0013, "3": -0.0016,
                   "5": -0.0020, "7": -0.0022, "10": -0.0025,
                   "20": -0.0027, "30": -0.0028},
        "GBPUSD": {"1": -0.0015, "2": -0.0018, "3": -0.0021,
                   "5": -0.0025, "7": -0.0027, "10": -0.0030,
                   "20": -0.0032, "30": -0.0033},
        "USDJPY": {"1":  0.0008, "2":  0.0010, "3":  0.0013,
                   "5":  0.0016, "7":  0.0018, "10":  0.0021,
                   "20":  0.0023, "30":  0.0025},
    }

    # ── Swaption normal vol surface (8 expiries × 8 swap tenors) ─────────────
    # Normal vols in bp/yr ÷ 100 = annualised fraction
    # Pattern: vol highest at short expiry / short tenor, decreasing with tenor & expiry
    swaption_vol_surface = {
        "0.08": {"1": 0.62, "2": 0.58, "3": 0.54, "5": 0.48, "7": 0.44, "10": 0.40, "20": 0.36, "30": 0.34},
        "0.25": {"1": 0.60, "2": 0.56, "3": 0.52, "5": 0.46, "7": 0.42, "10": 0.38, "20": 0.35, "30": 0.33},
        "0.5":  {"1": 0.56, "2": 0.52, "3": 0.49, "5": 0.44, "7": 0.40, "10": 0.37, "20": 0.34, "30": 0.32},
        "1":    {"1": 0.52, "2": 0.49, "3": 0.46, "5": 0.42, "7": 0.39, "10": 0.36, "20": 0.33, "30": 0.31},
        "2":    {"1": 0.48, "2": 0.46, "3": 0.44, "5": 0.40, "7": 0.37, "10": 0.34, "20": 0.32, "30": 0.30},
        "3":    {"1": 0.46, "2": 0.44, "3": 0.42, "5": 0.38, "7": 0.35, "10": 0.33, "20": 0.31, "30": 0.29},
        "5":    {"1": 0.43, "2": 0.41, "3": 0.40, "5": 0.36, "7": 0.34, "10": 0.32, "20": 0.30, "30": 0.28},
        "10":   {"1": 0.40, "2": 0.38, "3": 0.37, "5": 0.34, "7": 0.32, "10": 0.30, "20": 0.28, "30": 0.27},
    }
    # Legacy flat-key format (backward compat)
    swaption_vols = {
        "1,5": 0.42, "1,10": 0.36, "2,5": 0.40, "2,10": 0.34,
        "3,5": 0.38, "3,10": 0.33, "5,5": 0.36, "5,10": 0.32,
    }

    # ── Credit term structure (IG names steepen gently, HY steeper) ───────────
    credit_curves = {
        "FORD MOTOR CO":      {"1": 0.015, "2": 0.020, "3": 0.025, "5": 0.030, "7": 0.033, "10": 0.037},
        "GENERAL MOTORS CO":  {"1": 0.012, "2": 0.016, "3": 0.020, "5": 0.025, "7": 0.028, "10": 0.031},
        "AT&T INC":           {"1": 0.007, "2": 0.009, "3": 0.010, "5": 0.012, "7": 0.013, "10": 0.015},
        "VERIZON COMMS":      {"1": 0.006, "2": 0.008, "3": 0.009, "5": 0.010, "7": 0.011, "10": 0.013},
        "BOEING CO":          {"1": 0.010, "2": 0.013, "3": 0.015, "5": 0.018, "7": 0.020, "10": 0.023},
        "NETFLIX INC":        {"1": 0.010, "2": 0.013, "3": 0.016, "5": 0.020, "7": 0.022, "10": 0.025},
        "TESLA INC":          {"1": 0.018, "2": 0.024, "3": 0.028, "5": 0.035, "7": 0.040, "10": 0.045},
        "IBM CORP":           {"1": 0.005, "2": 0.006, "3": 0.007, "5": 0.008, "7": 0.009, "10": 0.010},
        "COMCAST CORP":       {"1": 0.006, "2": 0.007, "3": 0.008, "5": 0.010, "7": 0.011, "10": 0.012},
        "AMAZON COM INC":     {"1": 0.003, "2": 0.004, "3": 0.005, "5": 0.006, "7": 0.007, "10": 0.008},
        "EXXON MOBIL CORP":   {"1": 0.005, "2": 0.006, "3": 0.007, "5": 0.009, "7": 0.010, "10": 0.011},
        "JPMORGAN CHASE":     {"1": 0.004, "2": 0.005, "3": 0.006, "5": 0.007, "7": 0.008, "10": 0.009},
        "WALMART INC":        {"1": 0.003, "2": 0.004, "3": 0.004, "5": 0.005, "7": 0.006, "10": 0.007},
        "BERKSHIRE HATHAWAY": {"1": 0.003, "2": 0.004, "3": 0.005, "5": 0.006, "7": 0.007, "10": 0.008},
        "META PLATFORMS":     {"1": 0.006, "2": 0.008, "3": 0.009, "5": 0.011, "7": 0.012, "10": 0.014},
        "APPLE INC":          {"1": 0.003, "2": 0.004, "3": 0.004, "5": 0.005, "7": 0.006, "10": 0.007},
        "MICROSOFT CORP":     {"1": 0.003, "2": 0.003, "3": 0.004, "5": 0.005, "7": 0.006, "10": 0.007},
        "ALPHABET INC":       {"1": 0.004, "2": 0.005, "3": 0.005, "5": 0.006, "7": 0.007, "10": 0.008},
    }
    # Flat 5Y spread for backward compat
    credit_spreads = {e: v["5"] for e, v in credit_curves.items()}

    # ── Equity vol surface (expiry × moneyness, with put skew) ───────────────
    # moneyness = K/S; puts (< 1.0) carry higher vol (skew)
    def _eq_surface(atm_by_exp: dict, skew: float = 0.08) -> dict:
        """Build a vol surface slice: atm_by_exp = {expiry: atm_vol}."""
        moneyness = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
        out = {}
        for exp, atm in atm_by_exp.items():
            out[str(exp)] = {
                f"{m:.2f}": round(atm + skew * (1.0 - m), 4)
                for m in moneyness
            }
        return out

    equity_vol_surface = {
        "SPY":   _eq_surface({0.08: 0.13, 0.25: 0.14, 0.5: 0.15, 1.0: 0.16, 2.0: 0.17}, skew=0.06),
        "QQQ":   _eq_surface({0.08: 0.17, 0.25: 0.18, 0.5: 0.19, 1.0: 0.20, 2.0: 0.21}, skew=0.07),
        "IWM":   _eq_surface({0.08: 0.19, 0.25: 0.20, 0.5: 0.21, 1.0: 0.22, 2.0: 0.23}, skew=0.08),
        "EFA":   _eq_surface({0.08: 0.15, 0.25: 0.16, 0.5: 0.17, 1.0: 0.18, 2.0: 0.19}, skew=0.06),
        "SX5E":  _eq_surface({0.08: 0.15, 0.25: 0.16, 0.5: 0.17, 1.0: 0.18, 2.0: 0.19}, skew=0.09),
        "NKY":   _eq_surface({0.08: 0.17, 0.25: 0.18, 0.5: 0.19, 1.0: 0.20, 2.0: 0.21}, skew=0.07),
        "FTSE":  _eq_surface({0.08: 0.14, 0.25: 0.15, 0.5: 0.16, 1.0: 0.17, 2.0: 0.18}, skew=0.07),
        "AAPL":  _eq_surface({0.08: 0.24, 0.25: 0.25, 0.5: 0.26, 1.0: 0.27, 2.0: 0.28}, skew=0.10),
        "MSFT":  _eq_surface({0.08: 0.21, 0.25: 0.22, 0.5: 0.23, 1.0: 0.24, 2.0: 0.25}, skew=0.09),
        "GOOGL": _eq_surface({0.08: 0.23, 0.25: 0.24, 0.5: 0.25, 1.0: 0.26, 2.0: 0.27}, skew=0.09),
    }

    # ── Cap/floor normal vol cube (expiry × strike) ───────────────────────────
    # Higher strikes → lower vol (normal vol skew); longer expiry → lower vol (term structure)
    cap_floor_vols = {
        "1":  {"0.02": 0.72, "0.03": 0.68, "0.04": 0.62, "0.05": 0.55, "0.06": 0.48, "0.07": 0.43},
        "2":  {"0.02": 0.68, "0.03": 0.64, "0.04": 0.59, "0.05": 0.52, "0.06": 0.46, "0.07": 0.41},
        "3":  {"0.02": 0.64, "0.03": 0.60, "0.04": 0.55, "0.05": 0.49, "0.06": 0.44, "0.07": 0.39},
        "5":  {"0.02": 0.58, "0.03": 0.55, "0.04": 0.51, "0.05": 0.46, "0.06": 0.41, "0.07": 0.37},
        "7":  {"0.02": 0.54, "0.03": 0.51, "0.04": 0.48, "0.05": 0.44, "0.06": 0.40, "0.07": 0.36},
        "10": {"0.02": 0.50, "0.03": 0.47, "0.04": 0.44, "0.05": 0.41, "0.06": 0.38, "0.07": 0.34},
    }

    return MarketDataSnapshot(
        valuation_date       = valuation_date,
        sofr_curve           = sofr_curve,
        par_rates            = {
            "1": 0.0510, "2": 0.0490, "3": 0.0470, "5": 0.0450,
            "7": 0.0440, "10": 0.0430, "15": 0.0425, "20": 0.0420, "30": 0.0415,
        },
        swaption_vols        = swaption_vols,
        credit_spreads       = credit_spreads,
        equity_prices        = {
            "SPY": 480.0, "QQQ": 415.0, "IWM": 210.0, "EFA": 78.0,
            "SX5E": 4800.0, "NKY": 38000.0, "FTSE": 7600.0,
            "AAPL": 233.0, "MSFT": 416.0, "GOOGL": 193.0,
        },
        equity_divylds       = {
            "SPY": 0.013, "QQQ": 0.007, "IWM": 0.015, "EFA": 0.030,
            "SX5E": 0.032, "NKY": 0.025, "FTSE": 0.038,
            "AAPL": 0.005, "MSFT": 0.007, "GOOGL": 0.000,
        },
        fx_rates             = {"EURUSD": 1.085, "GBPUSD": 1.270, "USDJPY": 155.0},
        foreign_curves       = {"EUR": eur_estr, "GBP": gbp_sonia, "JPY": jpy_tonar},
        xccy_basis           = xccy_basis,
        credit_curves        = credit_curves,
        swaption_vol_surface = swaption_vol_surface,
        equity_vol_surface   = equity_vol_surface,
        cap_floor_vols       = cap_floor_vols,
    )


# ── equity ticker → credit entity mapping ────────────────────────────────────
TICKER_TO_ENTITY: Dict[str, str] = {
    "AAPL":  "APPLE INC",
    "MSFT":  "MICROSOFT CORP",
    "GOOGL": "ALPHABET INC",
    "AMZN":  "AMAZON COM INC",
    "TSLA":  "TESLA INC",
    "SPY":   "JPMORGAN CHASE",     # S&P 500 ETF: use IG proxy
    "QQQ":   "MICROSOFT CORP",     # Tech-heavy ETF: MSFT proxy
    "GLD":   "EXXON MOBIL CORP",   # Commodity ETF: IG proxy
    "TLT":   "JPMORGAN CHASE",     # Treasury ETF: IG proxy
    "VIX":   "JPMORGAN CHASE",     # Vol index: IG proxy
    "FB":    "META PLATFORMS",
    "META":  "META PLATFORMS",
    "NFLX":  "NETFLIX INC",
    "F":     "FORD MOTOR CO",
    "GM":    "GENERAL MOTORS CO",
    "T":     "AT&T INC",
    "VZ":    "VERIZON COMMS",
    "BA":    "BOEING CO",
    "IBM":   "IBM CORP",
    "CMCSA": "COMCAST CORP",
    "XOM":   "EXXON MOBIL CORP",
    "JPM":   "JPMORGAN CHASE",
    "WMT":   "WALMART INC",
    "BRK":   "BERKSHIRE HATHAWAY",
}

# ── ISIN-based issuer assignments for IG corporate bonds ─────────────────────
_IG_ISSUERS = [
    "APPLE INC", "AMAZON COM INC", "JPMORGAN CHASE", "WALMART INC",
    "MICROSOFT CORP", "EXXON MOBIL CORP", "AT&T INC", "VERIZON COMMS",
    "IBM CORP", "COMCAST CORP",
]

def issuer_from_isin(isin: str) -> str:
    """Deterministically map an ISIN to a credit entity for IG bonds."""
    if not isin:
        return ""
    try:
        idx = int(isin.replace("US", "").lstrip("0") or "0") % len(_IG_ISSUERS)
        return _IG_ISSUERS[idx]
    except Exception:
        return ""


# ── module-level singleton cache ──────────────────────────────────────────────

class MarketDataCache:
    """Thread-safe singleton cache for MarketDataSnapshot objects."""
    _instance: Optional["MarketDataCache"] = None
    _snapshots: Dict[str, MarketDataSnapshot] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._snapshots = {}
        return cls._instance

    def get(self, valuation_date: date) -> Optional[MarketDataSnapshot]:
        return self._snapshots.get(valuation_date.isoformat())

    def put(self, snap: MarketDataSnapshot) -> None:
        self._snapshots[snap.valuation_date.isoformat()] = snap

    def get_or_create(self, valuation_date: date) -> MarketDataSnapshot:
        snap = self.get(valuation_date)
        if snap is None:
            snap = make_default_snapshot(valuation_date)
            self.put(snap)
        return snap
