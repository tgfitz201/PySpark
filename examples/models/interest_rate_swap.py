"""
models/interest_rate_swap.py
============================
InterestRateSwap — replaces/aliases VanillaSwap. Registered under
trade_type="InterestRateSwap". VanillaSwap class still works for old data.

When legs carry different currencies this is a Cross-Currency IRS.
Use fx_rate (receive_ccy per pay_ccy) to convert legs to a common currency.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from models.trade_base import TradeBase
from models.leg import BaseLeg


@dataclass
class InterestRateSwap(TradeBase, trade_type="InterestRateSwap"):
    """Interest rate swap — any combination of fixed/floating legs.
    When legs have different currencies this is treated as a Cross-Currency IRS.
    fx_rate = receive_currency per pay_currency (e.g. 0.93 EUR per USD).
    """
    tenor_y:      int   = 0
    swap_subtype: str   = "FIXED_FLOAT"   # FIXED_FLOAT|FLOAT_FIXED|FIXED_FIXED|FLOAT_FLOAT
    fx_rate:      float = 1.0             # rcv/pay spot rate; 1.0 = single-currency

    @property
    def is_xccy(self) -> bool:
        """True when legs span more than one currency → cross-currency IRS."""
        ccys = {l.currency for l in self.legs if l.currency}
        return len(ccys) > 1

    @property
    def pay_currency(self) -> str:
        return self.legs[0].currency if self.legs else "USD"

    @property
    def receive_currency(self) -> str:
        return self.legs[1].currency if len(self.legs) > 1 else "USD"

    @property
    def fixed_legs(self) -> List[BaseLeg]:
        return [l for l in self.legs if l.leg_type == "FIXED"]

    @property
    def float_legs(self) -> List[BaseLeg]:
        return [l for l in self.legs if l.leg_type == "FLOAT"]

    @property
    def fixed_leg(self) -> Optional[BaseLeg]:
        fl = self.fixed_legs; return fl[0] if fl else None

    @property
    def float_leg(self) -> Optional[BaseLeg]:
        fl = self.float_legs; return fl[0] if fl else None
