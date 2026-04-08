"""
models/option_trade.py
======================
OptionTrade — European swaption, cap, floor, or rate option.

Inherits from TradeBase:
  trade_id, book, counterparty, valuation_date, direction, legs

direction values
----------------
  TradeDirection.BUY  — long option (paid premium upfront)
  TradeDirection.SELL — short option (received premium)

legs[0] must be an OptionLeg.

Swaption anatomy
----------------
  OptionLeg.leg_type          = "OPTION"
  OptionLeg.notional          = swaption notional
  OptionLeg.start_date        = option expiry / underlying effective date
  OptionLeg.end_date          = underlying swap maturity
  OptionLeg.strike            = underlying fixed rate
  OptionLeg.option_type       = "PAYER_SWAPTION" | "RECEIVER_SWAPTION"
  OptionLeg.vol               = flat implied vol  (lognormal or normal)
  OptionLeg.underlying_tenor_m= underlying swap tenor in months
  OptionLeg.day_count / frequency / calendar / bdc = underlying conventions

Properties
----------
  option_leg : OptionLeg  → legs[0]

Example
-------
>>> from datetime import date
>>> from models import TradeDirection, OptionLeg, OptionTrade
>>> expiry = date(2026, 1, 15);  mat = date(2031, 1, 15)  # 1y into 5y
>>> ot = OptionTrade(
...     trade_id="OPT-001", book="OPTIONS-NY", counterparty="CPTY-01",
...     valuation_date=date(2025, 1, 15), direction=TradeDirection.BUY,
...     legs=[OptionLeg("OPTION", 10e6, expiry, mat, strike=0.044,
...                     option_type="PAYER_SWAPTION", vol=0.40,
...                     underlying_tenor_m=60)],
...     tenor_y=1, underlying_tenor_y=5,
... )
>>> ot.option_leg.strike
0.044
"""

from __future__ import annotations

from dataclasses import dataclass

from models.trade_base import TradeBase
from models.leg import OptionLeg


@dataclass
class OptionTrade(TradeBase, trade_type="Option"):
    """European swaption / cap / floor built from an OptionLeg."""

    tenor_y:            int = 0   # option expiry in years (display)
    underlying_tenor_y: int = 0   # underlying instrument tenor in years (display)

    @property
    def option_leg(self) -> OptionLeg:
        """The OptionLeg (legs[0])."""
        ol = self.legs[0]
        if not isinstance(ol, OptionLeg):
            raise TypeError(f"legs[0] is {type(ol).__name__}, expected OptionLeg")
        return ol

    def _computed_props(self):
        ol = self.option_leg
        return {
            "option_leg_ref": {
                "_python_class":      "OptionLeg",
                "option_type":        ol.option_type,
                "exercise_type":      ol.exercise_type,
                "strike_pct":         f"{ol.strike * 100:.4f}%",
                "vol_pct":            f"{ol.vol * 100:.4f}%",
                "vol_type":           ol.vol_type,
                "vol_shift":          ol.vol_shift,
                "expiry":             str(ol.start_date),
                "underlying_tenor_m": ol.underlying_tenor_m,
                "underlying_type":    ol.underlying_type,
                "notional":           ol.notional,
            },
            "expiry_tenor_y":      self.tenor_y,
            "underlying_tenor_y":  self.underlying_tenor_y,
        }
