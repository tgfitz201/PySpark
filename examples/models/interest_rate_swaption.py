"""
models/interest_rate_swaption.py
=================================
InterestRateSwaption — European swaption with explicit 3-leg structure.

Leg layout
----------
legs[0] : FixedLeg   — underlying swap's fixed coupon stream
legs[1] : FloatLeg   — underlying swap's floating stream (SOFR + spread)
legs[2] : OptionLeg  — the optionality:
              start_date      = option expiry / underlying effective date
              end_date        = underlying swap maturity
              strike          = fixed rate (decimal, e.g. 0.045)
              option_type     = "PAYER_SWAPTION" | "RECEIVER_SWAPTION"
              vol             = flat implied vol (lognormal or normal)
              vol_type        = "LOGNORMAL" | "NORMAL"
              vol_shift       = shift for shifted-lognormal (default 0.03)
              underlying_tenor_m = underlying swap tenor in months

Direction
---------
TradeDirection.BUY   — long swaption (paid premium, positive vega)
TradeDirection.SELL  — short swaption (received premium, negative vega)

Distinction from OptionTrade
-----------------------------
OptionTrade collapses all swap detail into the OptionLeg.  InterestRateSwaption
keeps the underlying fixed/float legs explicit so cashflow greeks (fixed_npv,
float_npv, DV01 decomposition) are available alongside option greeks.
"""

from __future__ import annotations
from dataclasses import dataclass
from models.trade_base import TradeBase
from models.leg import FixedLeg, FloatLeg, OptionLeg


@dataclass
class InterestRateSwaption(TradeBase, trade_type="IRSwaption"):
    """
    European interest rate swaption with explicit fixed + float + option legs.

    Parameters
    ----------
    tenor_y             : option expiry in years (for display / bucketing)
    underlying_tenor_y  : underlying swap tenor in years
    swap_subtype        : "FIXED_FLOAT" (payer swaption) |
                          "FLOAT_FIXED" (receiver swaption)
    """
    tenor_y:            int = 0
    underlying_tenor_y: int = 0
    swap_subtype:       str = "FIXED_FLOAT"

    # ── leg accessors ─────────────────────────────────────────────────────────

    @property
    def fixed_leg(self) -> FixedLeg:
        """legs[0] — the underlying swap's fixed coupon leg."""
        leg = self.legs[0]
        if not isinstance(leg, FixedLeg):
            raise TypeError(f"legs[0] is {type(leg).__name__}, expected FixedLeg")
        return leg

    @property
    def float_leg(self) -> FloatLeg:
        """legs[1] — the underlying swap's floating rate leg."""
        leg = self.legs[1]
        if not isinstance(leg, FloatLeg):
            raise TypeError(f"legs[1] is {type(leg).__name__}, expected FloatLeg")
        return leg

    @property
    def option_leg(self) -> OptionLeg:
        """legs[2] — the swaption optionality leg."""
        leg = self.legs[2]
        if not isinstance(leg, OptionLeg):
            raise TypeError(f"legs[2] is {type(leg).__name__}, expected OptionLeg")
        return leg

    def _computed_props(self):
        ol = self.option_leg
        fl = self.fixed_leg
        return {
            "fixed_rate":         fl.coupon_rate,
            "option_expiry":      str(ol.start_date),
            "option_type":        ol.option_type,
            "strike":             ol.strike,
            "vol":                ol.vol,
            "vol_type":           ol.vol_type,
            "underlying_tenor_m": ol.underlying_tenor_m,
            "tenor_y":            self.tenor_y,
            "underlying_tenor_y": self.underlying_tenor_y,
        }
