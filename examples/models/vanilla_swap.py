"""
models/vanilla_swap.py
======================
VanillaSwap — any combination of Fixed and/or Float legs.

Inherits from TradeBase, so it carries:
  trade_id, book, counterparty, valuation_date, direction, legs

direction values
----------------
  TradeDirection.PAYER    — pay fixed / receive float
  TradeDirection.RECEIVER — receive fixed / pay float
  (for non-standard combos the pricer interprets legs[0] as "pay" side)

Leg combinations supported
--------------------------
  FIXED + FLOAT  — standard USD IRS  (most common)
  FIXED + FIXED  — fixed-fixed basis  (two different rates)
  FLOAT + FLOAT  — floating basis     (e.g. SOFR 1M vs 3M)
  N legs         — anything; pricer iterates trade.legs

Properties (derived from legs list — no stored leg fields)
----------------------------------------------------------
  fixed_leg   : first FIXED leg  (or None)
  float_leg   : first FLOAT leg  (or None)
  fixed_legs  : all FIXED legs
  float_legs  : all FLOAT legs

Example
-------
>>> from datetime import date
>>> from models import TradeDirection, BaseLeg, VanillaSwap
>>> eff = date(2025, 1, 17);  mat = date(2030, 1, 17)
>>> t = VanillaSwap(
...     trade_id="IRS-001", book="IRD-NY", counterparty="CPTY-01",
...     valuation_date=date(2025, 1, 15), direction=TradeDirection.PAYER,
...     legs=[
...         BaseLeg("FIXED", 10e6, eff, mat, coupon_rate=0.045),
...         BaseLeg("FLOAT", 10e6, eff, mat, frequency="QUARTERLY", day_count="ACT/360"),
...     ],
...     tenor_y=5,
... )
>>> t.fixed_leg.coupon_rate
0.045
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from models.trade_base import TradeBase
from models.leg import BaseLeg


@dataclass
class VanillaSwap(TradeBase, trade_type="VanillaSwap"):
    """Flexible interest rate swap — any combination of fixed/floating legs."""

    tenor_y:      int = 0   # display / sanity-check field; must have default (legs has one)
    swap_subtype: str = "FIXED_FLOAT"  # FIXED_FLOAT|FLOAT_FIXED|FIXED_FIXED|FLOAT_FLOAT

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def fixed_legs(self) -> List[BaseLeg]:
        return [l for l in self.legs if l.leg_type == "FIXED"]

    @property
    def float_legs(self) -> List[BaseLeg]:
        return [l for l in self.legs if l.leg_type == "FLOAT"]

    @property
    def fixed_leg(self) -> Optional[BaseLeg]:
        fl = self.fixed_legs
        return fl[0] if fl else None

    @property
    def float_leg(self) -> Optional[BaseLeg]:
        fl = self.float_legs
        return fl[0] if fl else None

    def _computed_props(self):
        fl = self.fixed_leg
        vl = self.float_leg
        props = {
            "swap_description": (
                f"{self.swap_subtype}  {self.direction.value}  "
                f"{self.tenor_y}y  notional=${self.legs[0].notional:,.0f}"
            ),
            "leg_count": len(self.legs),
            "fixed_leg_ref": {
                "_python_class": "BaseLeg",
                "leg_type": fl.leg_type,
                "coupon_rate_pct": f"{fl.coupon_rate * 100:.4f}%",
                "day_count": fl.day_count,
                "frequency": fl.frequency,
                "notional": fl.notional,
            } if fl else None,
            "float_leg_ref": {
                "_python_class": "BaseLeg",
                "leg_type": vl.leg_type,
                "spread_pct": f"{vl.spread * 100:.4f}%",
                "index": vl.index_name,
                "day_count": vl.day_count,
                "frequency": vl.frequency,
                "notional": vl.notional,
            } if vl else None,
        }
        return props

    def price(self, curve_df):
        from pricing.swap_pricer import price_swap
        return price_swap(self, curve_df)
