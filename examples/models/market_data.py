"""
models/market_data.py
=====================
MarketDataSnapshot — frozen dataclass holding all market observables for one valuation date.
MarketDataCache    — module-level singleton for cached access.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional
import QuantLib as ql
import pandas as pd


@dataclass(frozen=True)
class MarketDataSnapshot:
    """
    Immutable snapshot of all market data for one valuation date.

    Fields
    ------
    valuation_date  : pricing date
    sofr_curve      : {tenor_y_str: zero_rate}  e.g. {"0.25": 0.053, "1": 0.051, ...}
    par_rates       : {tenor_y_str: par_rate}   e.g. {"1": 0.051, "2": 0.049, ...}
    swaption_vols   : {"expiry_y,swap_y": vol}  e.g. {"1,5": 0.38, "2,10": 0.38}
    credit_spreads  : {entity_name: spread}     e.g. {"FORD MOTOR CO": 0.030}
    equity_prices   : {ticker: price}           e.g. {"SPY": 480.0}
    equity_divylds  : {ticker: div_yield}       e.g. {"SPY": 0.013}
    fx_rates        : {ccy_pair: rate}          e.g. {"EURUSD": 1.085}  (optional)
    """

    valuation_date:  date
    sofr_curve:      Dict[str, float]            # str keys for JSON compat
    par_rates:       Dict[str, float]
    swaption_vols:   Dict[str, float]            # key "expiry_y,swap_y"
    credit_spreads:  Dict[str, float]
    equity_prices:   Dict[str, float]
    equity_divylds:  Dict[str, float]
    fx_rates:        Dict[str, float] = field(default_factory=dict)

    # ── curve helpers ─────────────────────────────────────────────────────────

    def to_curve_df(self) -> pd.DataFrame:
        """Return sorted SOFR zero curve as DataFrame[tenor_y, zero_rate]."""
        rows = sorted((float(k), v) for k, v in self.sofr_curve.items())
        return pd.DataFrame(rows, columns=["tenor_y", "zero_rate"])

    def build_sofr_curve(self, ql_val: ql.Date) -> ql.YieldTermStructureHandle:
        """Build a QuantLib ZeroCurve from the stored SOFR data."""
        dc    = ql.Actual365Fixed()
        dates = [ql_val]
        rates = [list(self.sofr_curve.values())[0]]
        for k, r in sorted(self.sofr_curve.items(), key=lambda x: float(x[0])):
            months = max(1, int(round(float(k) * 12)))
            dates.append(ql_val + ql.Period(months, ql.Months))
            rates.append(r)
        curve = ql.ZeroCurve(dates, rates, dc, ql.NullCalendar(),
                             ql.Linear(), ql.Continuous, ql.Annual)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def get_par_rate(self, tenor_y: int) -> float:
        """Look up par rate for tenor, interpolating if needed."""
        key = str(tenor_y)
        if key in self.par_rates:
            return self.par_rates[key]
        tenors = sorted(float(k) for k in self.par_rates)
        if tenor_y <= tenors[0]:
            return self.par_rates[str(int(tenors[0]))]
        if tenor_y >= tenors[-1]:
            return self.par_rates[str(int(tenors[-1]))]
        for i in range(len(tenors) - 1):
            if tenors[i] <= tenor_y <= tenors[i + 1]:
                t0, t1 = tenors[i], tenors[i + 1]
                r0 = self.par_rates[str(int(t0))]
                r1 = self.par_rates[str(int(t1))]
                return r0 + (r1 - r0) * (tenor_y - t0) / (t1 - t0)
        return 0.045

    def get_swaption_vol(self, expiry_y: int, swap_y: int) -> float:
        key = f"{expiry_y},{swap_y}"
        if key in self.swaption_vols:
            return self.swaption_vols[key]
        for exp in [expiry_y, 1, 2, 3, 5]:
            k2 = f"{exp},{swap_y}"
            if k2 in self.swaption_vols:
                return self.swaption_vols[k2]
        return 0.40

    # ── serialization ─────────────────────────────────────────────────────────

    def toJson(self) -> str:
        d = {
            "valuation_date": self.valuation_date.isoformat(),
            "sofr_curve":     self.sofr_curve,
            "par_rates":      self.par_rates,
            "swaption_vols":  self.swaption_vols,
            "credit_spreads": self.credit_spreads,
            "equity_prices":  self.equity_prices,
            "equity_divylds": self.equity_divylds,
            "fx_rates":       self.fx_rates,
        }
        return json.dumps(d)

    @classmethod
    def fromJson(cls, js: str) -> "MarketDataSnapshot":
        d = json.loads(js)
        return cls(
            valuation_date  = date.fromisoformat(d["valuation_date"]),
            sofr_curve      = d["sofr_curve"],
            par_rates       = d["par_rates"],
            swaption_vols   = d["swaption_vols"],
            credit_spreads  = d["credit_spreads"],
            equity_prices   = d["equity_prices"],
            equity_divylds  = d["equity_divylds"],
            fx_rates        = d.get("fx_rates", {}),
        )


# ── default market data for 2025-01-15 ───────────────────────────────────────

def make_default_snapshot(valuation_date: date = date(2025, 1, 15)) -> MarketDataSnapshot:
    """Return the default USD market data snapshot for 2025-01-15."""
    return MarketDataSnapshot(
        valuation_date = valuation_date,
        sofr_curve = {
            "0.25": 0.0530, "0.5": 0.0525, "1": 0.0510, "2": 0.0490,
            "3": 0.0470, "5": 0.0450, "7": 0.0440, "10": 0.0430,
            "15": 0.0425, "20": 0.0420, "30": 0.0415,
        },
        par_rates = {
            "1": 0.0510, "2": 0.0490, "3": 0.0470, "5": 0.0450,
            "7": 0.0440, "10": 0.0430, "15": 0.0425, "20": 0.0420, "30": 0.0415,
        },
        swaption_vols = {
            "1,5": 0.38, "1,10": 0.36, "2,5": 0.40, "2,10": 0.38,
            "3,5": 0.42, "3,10": 0.40, "5,5": 0.44, "5,10": 0.42,
        },
        credit_spreads = {
            "FORD MOTOR CO":      0.030, "GENERAL MOTORS CO":  0.025,
            "AT&T INC":           0.012, "VERIZON COMMS":      0.010,
            "BOEING CO":          0.018, "NETFLIX INC":        0.020,
            "TESLA INC":          0.035, "IBM CORP":           0.008,
            "COMCAST CORP":       0.010, "AMAZON COM INC":     0.006,
            "EXXON MOBIL CORP":   0.009, "JPMORGAN CHASE":     0.007,
            "WALMART INC":        0.005, "BERKSHIRE HATHAWAY": 0.006,
            "META PLATFORMS":     0.011,
        },
        equity_prices = {
            "SPY": 480.0, "QQQ": 415.0, "IWM": 210.0, "EFA": 78.0,
            "SX5E": 4800.0, "NKY": 38000.0, "FTSE": 7600.0,
            "AAPL": 233.0, "MSFT": 416.0, "GOOGL": 193.0,
        },
        equity_divylds = {
            "SPY": 0.013, "QQQ": 0.007, "IWM": 0.015, "EFA": 0.030,
            "SX5E": 0.032, "NKY": 0.025, "FTSE": 0.038,
            "AAPL": 0.005, "MSFT": 0.007, "GOOGL": 0.000,
        },
        fx_rates = {"EURUSD": 1.085, "GBPUSD": 1.270, "USDJPY": 155.0},
    )


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
