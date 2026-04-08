"""
models/callable_bond.py
=======================
CallableBond — fixed-rate bond with embedded call or put option.

Leg structure (2 legs):
  legs[0] BondLeg (leg_type="BOND") — coupon + redemption cash flows
          Same as the existing Bond model's leg: coupon_rate, notional, dates
  legs[1] OptionLeg (leg_type="OPTION") — the embedded option terms
          option_type: "CALL" (issuer can redeem early) or "PUT" (holder can sell back)
          exercise_type: "AMERICAN" (continuously callable) or "BERMUDAN" (call schedule)
          strike: call/put price as % of par (e.g. 1.00 = par, 1.02 = 102)
          vol: interest rate volatility for the HullWhite model (e.g. 0.01 = 1%)
          vol_type: "NORMAL" (HullWhite uses normal vol)
          start_date: first call/put date
          end_date: last call/put date (= bond maturity for continuously callable)

Properties:
  bond_leg    → legs[0] (BaseLeg with leg_type BOND)
  option_leg  → legs[1] (OptionLeg)
  call_type   : str = "CALLABLE" | "PUTABLE"
  tenor_y     : int (bond tenor in years)

trade_type = "CallableBond"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.trade_base import TradeBase
from models.leg import BaseLeg, OptionLeg


@dataclass
class CallableBond(TradeBase, trade_type="CallableBond"):
    """Fixed-rate bond with embedded issuer call or holder put option."""

    tenor_y: int           = 0
    isin:    Optional[str] = None

    @property
    def bond_leg(self) -> BaseLeg:
        """The BOND cash-flow leg (legs[0])."""
        return self.legs[0]

    @property
    def option_leg(self) -> OptionLeg:
        """The embedded OPTION leg (legs[1])."""
        return self.legs[1]

    @property
    def call_type(self) -> str:
        """CALLABLE (issuer call) or PUTABLE (holder put)."""
        ot = getattr(self.option_leg, "option_type", "CALL")
        return "CALLABLE" if ot == "CALL" else "PUTABLE"

    def _computed_props(self):
        bl = self.bond_leg
        ol = self.option_leg
        return {
            "bond_leg_ref": {
                "_python_class":   "BaseLeg",
                "leg_type":        bl.leg_type,
                "notional":        bl.notional,
                "coupon_rate_pct": f"{bl.coupon_rate * 100:.4f}%",
                "maturity":        str(bl.end_date),
                "day_count":       bl.day_count,
                "frequency":       bl.frequency,
                "redemption_pct":  bl.redemption,
                "settlement_days": bl.settlement_days,
                "issue_date":      str(bl.issue_date),
            },
            "option_leg_ref": {
                "_python_class": "OptionLeg",
                "option_type":   ol.option_type,
                "exercise_type": ol.exercise_type,
                "strike_pct":    f"{ol.strike * 100:.2f}%",
                "vol_pct":       f"{ol.vol * 100:.2f}%",
                "vol_type":      ol.vol_type,
                "first_call":    str(ol.start_date),
                "last_call":     str(ol.end_date),
            },
            "call_type": self.call_type,
            "tenor_y":   self.tenor_y,
            "isin":      self.isin,
        }
