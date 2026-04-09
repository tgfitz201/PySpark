"""
models/optionable_bond.py
=========================
OptionableBond — fixed-rate bond with an embedded option (or sinking fund schedule).

bond_subtype values
-------------------
  CALLABLE     — issuer has right to redeem early at call price (HullWhite)
  PUTABLE      — holder has right to sell back at put price (HullWhite)
  CONVERTIBLE  — holder can convert bond to equity shares (Binomial)
  EXTENDABLE   — holder can extend maturity past original date (HullWhite putable)
  SINKING_FUND — mandatory annual partial redemptions (AmortizingFixedRateBond)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from models.trade_base import TradeBase
from models.leg import BaseLeg, OptionLeg, EquityOptionLeg


VALID_SUBTYPES = {"CALLABLE", "PUTABLE", "CONVERTIBLE", "EXTENDABLE", "SINKING_FUND"}


@dataclass
class OptionableBond(TradeBase, trade_type="OptionableBond"):
    """Fixed-rate bond with an embedded option or sinking fund schedule."""

    tenor_y:               int            = 0
    isin:                  Optional[str]  = None
    bond_subtype:          str            = "CALLABLE"
    sinking_pct_per_period: float         = 0.0
    conversion_premium:    float          = 0.0

    @property
    def bond_leg(self) -> BaseLeg:
        return self.legs[0]

    @property
    def option_leg(self):
        """Return legs[1] if present (None for SINKING_FUND)."""
        return self.legs[1] if len(self.legs) > 1 else None

    @property
    def call_type(self) -> str:
        """Return CALLABLE or PUTABLE for backward compat with _price_callable_bond."""
        if self.bond_subtype == "PUTABLE":
            return "PUTABLE"
        if self.bond_subtype == "CALLABLE":
            return "CALLABLE"
        # For EXTENDABLE, use PUT option logic
        if self.bond_subtype == "EXTENDABLE":
            return "PUTABLE"
        return "CALLABLE"

    def _computed_props(self):
        bl = self.bond_leg
        ol = self.option_leg
        props = {
            "bond_subtype": self.bond_subtype,
            "bond_leg_ref": {
                "_python_class": "BaseLeg",
                "leg_type": bl.leg_type,
                "notional": bl.notional,
                "coupon_rate_pct": f"{bl.coupon_rate * 100:.4f}%",
                "maturity": str(bl.end_date),
                "day_count": bl.day_count,
                "frequency": bl.frequency,
                "redemption_pct": bl.redemption,
                "settlement_days": bl.settlement_days,
                "issue_date": str(bl.issue_date),
            },
            "isin": self.isin,
            "tenor_y": self.tenor_y,
        }
        if ol is not None:
            props["option_leg_ref"] = {
                "_python_class": type(ol).__name__,
                "option_type": getattr(ol, "option_type", ""),
                "exercise_type": getattr(ol, "exercise_type", ""),
                "strike": getattr(ol, "strike", None),
                "vol": getattr(ol, "vol", None),
                "start_date": str(ol.start_date),
                "end_date": str(ol.end_date),
            }
        if self.bond_subtype == "SINKING_FUND":
            props["sinking_pct_per_period"] = self.sinking_pct_per_period
        if self.bond_subtype == "CONVERTIBLE":
            props["conversion_premium"] = self.conversion_premium
        return props
