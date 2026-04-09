"""
models/asset_swap.py
====================
AssetSwap — a fixed-rate bond packaged with an interest rate swap.

The bondholder receives the bond's fixed coupon and pays SOFR + spread
on the floating leg. The "asset swap spread" is the key metric: how many
basis points over SOFR the investor earns after financing the bond at par.

Leg structure
-------------
  legs[0] = BaseLeg(leg_type="BOND")   — the underlying fixed-rate bond
  legs[1] = FloatLeg(leg_type="FLOAT") — SOFR floating leg (pay side)

Extra fields
------------
  par_price   : float = 100.0   — price at which asset swap is structured (usually par)
  isin        : str   = ""      — bond ISIN
  tenor_y     : int   = 0       — bond tenor in years
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from models.trade_base import TradeBase
from models.leg import BaseLeg, FloatLeg


@dataclass
class AssetSwap(TradeBase, trade_type="AssetSwap"):
    """Fixed-rate bond packaged with an IRS — key output is asset swap spread."""

    tenor_y:   int           = 0
    isin:      Optional[str] = None
    par_price: float         = 100.0

    @property
    def bond_leg(self) -> BaseLeg:
        return self.legs[0]

    @property
    def float_leg(self) -> FloatLeg:
        return self.legs[1]

    def price(self, curve_df) -> dict:
        from pricing.asset_swap_pricer import price_asset_swap
        return price_asset_swap(self, curve_df)
