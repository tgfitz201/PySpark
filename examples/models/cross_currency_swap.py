"""
models/cross_currency_swap.py
=============================
CrossCurrencySwap — two-currency fixed/float swap with optional
notional exchange at start and maturity.

Legs:
  legs[0] — pay leg (currency = pay_currency)
  legs[1] — receive leg (currency = receive_currency)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from models.trade_base import TradeBase
from models.leg import BaseLeg


@dataclass
class CrossCurrencySwap(TradeBase, trade_type="CrossCurrencySwap"):
    """Cross-currency interest rate swap."""
    tenor_y:                   int   = 0
    pay_currency:              str   = "USD"
    receive_currency:          str   = "EUR"
    fx_rate:                   float = 1.0    # spot receive/pay
    initial_notional_exchange: bool  = True
    final_notional_exchange:   bool  = True
    swap_subtype:              str   = "FIXED_FLOAT"

    @property
    def pay_leg(self) -> Optional[BaseLeg]:
        return self.legs[0] if self.legs else None

    @property
    def receive_leg(self) -> Optional[BaseLeg]:
        return self.legs[1] if len(self.legs) > 1 else None

    def price(self, curve_df):
        from pricing.swap_pricer import price_xccy
        return price_xccy(self, curve_df)
