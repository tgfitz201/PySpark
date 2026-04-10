"""
models/cap_floor.py
===================
CapFloor — interest-rate cap or floor instrument.

A cap is a strip of European call options on a floating rate (SOFR/LIBOR);
the buyer receives max(rate - strike, 0) × notional × accrual_period on
each reset date, providing protection against rates rising above the strike.
A floor provides protection against rates falling below the strike.

Structure: single CapFloorLeg carrying all schedule/strike/vol details.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from models.trade_base import TradeBase
from models.leg import CapFloorLeg


@dataclass
class CapFloor(TradeBase, trade_type="CapFloor"):
    """Interest-rate cap or floor (strip of caplets/floorlets)."""

    tenor_y:        int   = 0
    cap_floor_type: str   = "CAP"    # CAP | FLOOR — mirrors leg field for convenience

    @property
    def cap_floor_leg(self) -> CapFloorLeg:
        return self.legs[0]

    def _computed_props(self):
        cl = self.cap_floor_leg
        return {
            "cap_floor_type": self.cap_floor_type,
            "cap_floor_leg_ref": {
                "_python_class":   "CapFloorLeg",
                "cap_floor_type":  cl.cap_floor_type,
                "notional":        cl.notional,
                "start_date":      str(cl.start_date),
                "maturity":        str(cl.end_date),
                "strike_pct":      f"{cl.strike * 100:.4f}%",
                "index":           cl.index_name,
                "frequency":       cl.frequency,
                "day_count":       cl.day_count,
            },
            "tenor_y": self.tenor_y,
        }

    def price(self, curve_df):
        from pricing.cap_floor_pricer import price_cap_floor
        return price_cap_floor(self, curve_df)
