"""
models/pricing_result.py
========================
PricingResult — flat dataclass capturing identity + pricing output for one trade.
Provides write_csv and portfolio_summary class methods for CSV output.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

_NAN = float("nan")


@dataclass
class PricingResult:
    """
    Flat pricing result record — one per trade.

    Identity fields
    ---------------
    trade_id, book, counterparty, valuation_date, direction, run_timestamp

    Trade metadata
    --------------
    instrument       : "IRS" | "BOND" | "SWAPTION" | "EQ_SWAP" | "CDS" | "EQ_OPT"
    swap_subtype     : IRS variant (FIXED_FLOAT | FLOAT_FIXED | FIXED_FIXED | FLOAT_FLOAT)
    tenor_y          : trade tenor in years
    leg_count        : number of legs
    leg_types        : comma-separated leg type strings
    notional         : first leg notional
    coupon_rate      : fixed coupon / strike / credit_spread
    start_date       : first leg start date
    end_date         : first leg end date (maturity)
    currency         : ISO currency code

    Equity / option fields
    ----------------------
    underlying_ticker, initial_price, dividend_yield
    strike, option_type, exercise_type, vol

    Credit fields
    -------------
    reference_entity, credit_spread, recovery_rate, seniority, doc_clause

    IRS fields
    ----------
    spread, index_name

    Pricing outputs
    ---------------
    npv, fixed_npv, float_npv, par_rate, clean_price, accrued, premium
    dv01, duration, pv01, convexity
    vega, theta, delta, gamma, rho
    cr01, jump_to_default
    error
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    trade_id:       str = ""
    book:           str = ""
    counterparty:   str = ""
    valuation_date: str = ""
    direction:      str = ""
    run_timestamp:  str = ""

    # ── Trade metadata ────────────────────────────────────────────────────────
    instrument:    str   = ""
    swap_subtype:  str   = ""
    tenor_y:       int   = 0
    leg_count:     int   = 0
    leg_types:     str   = ""
    notional:      float = _NAN
    coupon_rate:   float = _NAN
    start_date:    str   = ""
    end_date:      str   = ""
    currency:      str   = "USD"

    # ── Equity / option ───────────────────────────────────────────────────────
    underlying_ticker: str   = ""
    initial_price:     float = _NAN
    dividend_yield:    float = _NAN
    strike:            float = _NAN
    option_type:       str   = ""
    exercise_type:     str   = ""
    vol:               float = _NAN

    # ── Credit ────────────────────────────────────────────────────────────────
    reference_entity: str   = ""
    credit_spread:    float = _NAN
    recovery_rate:    float = _NAN
    seniority:        str   = ""
    doc_clause:       str   = ""

    # ── IRS ───────────────────────────────────────────────────────────────────
    spread:     float = _NAN
    index_name: str   = ""

    # ── Pricing outputs ───────────────────────────────────────────────────────
    npv:          float = _NAN
    fixed_npv:    float = _NAN
    float_npv:    float = _NAN
    par_rate:     float = _NAN
    clean_price:  float = _NAN
    accrued:      float = _NAN
    premium:      float = _NAN

    dv01:         float = _NAN
    duration:     float = _NAN
    pv01:         float = _NAN
    convexity:    float = _NAN

    vega:         float = _NAN
    theta:        float = _NAN
    delta:        float = _NAN
    gamma:        float = _NAN
    rho:          float = _NAN

    cr01:             float = _NAN
    jump_to_default:  float = _NAN

    error: str = ""

    # ── Class methods ─────────────────────────────────────────────────────────

    def _fmt(self, v: Any) -> str:
        """Format a field value for CSV output."""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    def to_dict(self) -> Dict[str, str]:
        """Return a dict of string-formatted field values."""
        return {f.name: self._fmt(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def write_csv(cls, results: List["PricingResult"], filepath: str) -> None:
        """Write a list of PricingResult objects to a CSV file."""
        if not results:
            return
        fieldnames = [f.name for f in fields(cls)]
        with open(filepath, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())

    @classmethod
    def portfolio_summary(cls, results: List["PricingResult"]) -> List[Dict[str, str]]:
        """Build a list of {metric, value} dicts for portfolio-level statistics."""
        if not results:
            return []

        def _sum(attr: str) -> float:
            return sum(getattr(r, attr) for r in results
                       if not math.isnan(getattr(r, attr, _NAN)))

        def _mean(attr: str) -> float:
            vals = [getattr(r, attr) for r in results
                    if not math.isnan(getattr(r, attr, _NAN))]
            return sum(vals) / len(vals) if vals else _NAN

        def _count(attr: str, val: Any) -> int:
            return sum(1 for r in results if getattr(r, attr, None) == val)

        inst = results[0].instrument if results else ""
        rows: List[Dict[str, str]] = [
            {"metric": "Instrument",    "value": inst},
            {"metric": "Trade Count",   "value": str(len(results))},
            {"metric": "Error Count",   "value": str(sum(1 for r in results if r.error))},
            {"metric": "Total Notional","value": f"${_sum('notional'):,.0f}"},
            {"metric": "Total NPV",     "value": f"${_sum('npv'):,.2f}"},
            {"metric": "Total DV01",    "value": f"${_sum('dv01'):,.2f}"},
        ]

        if inst == "IRS":
            rows += [
                {"metric": "Payers",    "value": str(_count("direction", "PAYER"))},
                {"metric": "Receivers", "value": str(_count("direction", "RECEIVER"))},
                {"metric": "Avg Par Rate", "value": f"{_mean('par_rate'):.4%}"},
            ]
            for sub in ["FIXED_FLOAT", "FLOAT_FIXED", "FIXED_FIXED", "FLOAT_FLOAT"]:
                rows.append({"metric": f"  {sub}",
                             "value": str(sum(1 for r in results if r.swap_subtype == sub))})

        elif inst == "BOND":
            rows += [
                {"metric": "Long",            "value": str(_count("direction", "LONG"))},
                {"metric": "Short",           "value": str(_count("direction", "SHORT"))},
                {"metric": "Avg Clean Px",    "value": f"{_mean('clean_price'):.4f}"},
                {"metric": "Avg YTM",         "value": f"{_mean('par_rate'):.4%}"},
                {"metric": "Total Convexity", "value": f"{_sum('convexity'):,.1f}"},
            ]

        elif inst == "SWAPTION":
            rows += [
                {"metric": "Buy",           "value": str(_count("direction", "BUY"))},
                {"metric": "Sell",          "value": str(_count("direction", "SELL"))},
                {"metric": "Total Premium", "value": f"${_sum('premium'):,.2f}"},
                {"metric": "Total Vega",    "value": f"${_sum('vega'):,.2f}"},
                {"metric": "Total Theta",   "value": f"${_sum('theta'):,.2f}"},
            ]

        elif inst == "EQ_SWAP":
            rows += [
                {"metric": "Long",          "value": str(_count("direction", "LONG"))},
                {"metric": "Short",         "value": str(_count("direction", "SHORT"))},
                {"metric": "Total EQ NPV",  "value": f"${_sum('npv'):,.2f}"},
                {"metric": "Total Delta",   "value": f"${_sum('delta'):,.0f}"},
                {"metric": "Total EQ DV01", "value": f"${_sum('dv01'):,.2f}"},
            ]

        elif inst == "CDS":
            rows += [
                {"metric": "Buy",      "value": str(_count("direction", "BUY"))},
                {"metric": "Sell",     "value": str(_count("direction", "SELL"))},
                {"metric": "Total CR01","value": f"${_sum('cr01'):,.2f}"},
                {"metric": "Total JTD","value": f"${_sum('jump_to_default'):,.0f}"},
            ]

        elif inst == "EQ_OPT":
            rows += [
                {"metric": "Buy",           "value": str(_count("direction", "BUY"))},
                {"metric": "Sell",          "value": str(_count("direction", "SELL"))},
                {"metric": "Total Premium", "value": f"${_sum('premium'):,.2f}"},
                {"metric": "Total Vega",    "value": f"${_sum('vega'):,.2f}"},
                {"metric": "Total Delta",   "value": f"${_sum('delta'):,.2f}"},
                {"metric": "Total Gamma",   "value": f"${_sum('gamma'):,.4f}"},
                {"metric": "Total Theta",   "value": f"${_sum('theta'):,.2f}"},
                {"metric": "Total Rho",     "value": f"${_sum('rho'):,.2f}"},
            ]

        return rows
