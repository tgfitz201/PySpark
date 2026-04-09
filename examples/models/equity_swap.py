"""
models/equity_swap.py
=====================
EquitySwap — equity total-return swap (receive equity / pay SOFR+spread).

Inherits from TradeBase:
  trade_id, book, counterparty, valuation_date, direction, legs

direction values
----------------
  TradeDirection.LONG  — receive equity total return, pay SOFR+spread
  TradeDirection.SHORT — pay equity total return, receive SOFR+spread

Legs
----
  legs[0] : EquityLeg  (equity total-return leg)
  legs[1] : BaseLeg("FLOAT", ...)  (funding leg — optional; may be implicit)

Pricing model (single-period analytical)
-----------------------------------------
  equity forward: F = S0 × exp((r − q) × T)
  PV(equity leg) = Notional × exp(−q × T)
  PV(funding leg)= Notional × exp(−r × T)
  NPV            = sign × (PV_equity − PV_funding)

Greeks
------
  delta   : $ per 1% move in underlying equity
  dv01    : $ per 1bp parallel SOFR shift (funding-leg sensitivity)
  theta   : daily P&L decay

Properties
----------
  equity_leg  : EquityLeg  → first EQUITY leg
  funding_leg : BaseLeg    → first FLOAT leg (may be None)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.trade_base import TradeBase
from models.leg import BaseLeg, EquityLeg


@dataclass
class EquitySwap(TradeBase, trade_type="EquitySwap"):
    """Equity total-return swap with an EquityLeg and optional floating funding leg."""

    tenor_y:           int          = 0
    underlying_ticker: str          = ""   # display field; source of truth is equity_leg

    @property
    def equity_leg(self) -> EquityLeg:
        """First EQUITY leg (legs[0])."""
        for l in self.legs:
            if l.leg_type == "EQUITY":
                if not isinstance(l, EquityLeg):
                    raise TypeError(f"leg_type=EQUITY but got {type(l).__name__}")
                return l
        raise ValueError("EquitySwap has no EQUITY leg")

    @property
    def funding_leg(self) -> Optional[BaseLeg]:
        """First FLOAT leg used as funding (may be None for unfunded swaps)."""
        for l in self.legs:
            if l.leg_type == "FLOAT":
                return l
        return None

    def _computed_props(self):
        el = self.equity_leg
        fl = self.funding_leg
        return {
            "equity_leg_ref": {
                "_python_class":      "EquityLeg",
                "underlying_ticker":  el.underlying_ticker,
                "initial_price":      el.initial_price,
                "dividend_yield_pct": f"{el.dividend_yield * 100:.4f}%",
                "equity_return_type": el.equity_return_type,
                "participation_rate": el.participation_rate,
                "reset_frequency":    el.reset_frequency,
                "funding_spread_pct": f"{el.funding_spread * 100:.4f}%",
                "notional":           el.notional,
            },
            "funding_leg_ref": {
                "_python_class": "BaseLeg",
                "leg_type":      fl.leg_type,
                "index":         fl.index_name,
                "spread_pct":    f"{fl.spread * 100:.4f}%",
                "notional":      fl.notional,
            } if fl else None,
            "tenor_y": self.tenor_y,
        }

    def price(self, curve_df):
        from pricing.equity_pricer import price_equity_swap
        return price_equity_swap(self, curve_df)
