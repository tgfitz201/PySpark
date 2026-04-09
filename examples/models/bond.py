"""
models/bond.py
==============
Bond — fixed-rate coupon bond trade with a single BOND leg.

Inherits from TradeBase:
  trade_id, book, counterparty, valuation_date, direction, legs

direction values
----------------
  TradeDirection.LONG  — asset position (bought bond)
  TradeDirection.SHORT — short position (sold bond)

The single legs[0] must be a BaseLeg with leg_type="BOND".

Property
--------
  bond_leg : BaseLeg  → legs[0]

Example
-------
>>> from datetime import date
>>> from models import TradeDirection, BaseLeg, Bond
>>> issued = date(2024, 7, 15);  mat = date(2034, 7, 15)
>>> b = Bond(
...     trade_id="BOND-001", book="GOVTBOND-NY", counterparty="CPTY-01",
...     valuation_date=date(2025, 1, 15), direction=TradeDirection.LONG,
...     legs=[BaseLeg("BOND", 1_000_000, issued, mat, coupon_rate=0.045,
...                   day_count="ACT/ACT", redemption=100.0, settlement_days=2,
...                   issue_date=issued)],
...     tenor_y=10, isin="US912810XX00",
... )
>>> b.bond_leg.coupon_rate
0.045
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.trade_base import TradeBase
from models.leg import BaseLeg


@dataclass
class Bond(TradeBase, trade_type="Bond"):
    """Fixed-rate coupon bond with a single BOND leg."""

    tenor_y: int          = 0
    isin:    Optional[str] = None

    @property
    def bond_leg(self) -> BaseLeg:
        """The single BOND leg (legs[0])."""
        return self.legs[0]

    def _computed_props(self):
        bl = self.bond_leg
        return {
            "bond_leg_ref": {
                "_python_class": "BaseLeg",
                "leg_type":       bl.leg_type,
                "notional":       bl.notional,
                "coupon_rate_pct": f"{bl.coupon_rate * 100:.4f}%",
                "maturity":       str(bl.end_date),
                "day_count":      bl.day_count,
                "frequency":      bl.frequency,
                "redemption_pct": bl.redemption,
                "settlement_days": bl.settlement_days,
                "issue_date":     str(bl.issue_date),
            },
            "isin": self.isin,
            "tenor_y": self.tenor_y,
        }

    def price(self, curve_df):
        from pricing.bond_pricer import price_bond
        return price_bond(self, curve_df)
