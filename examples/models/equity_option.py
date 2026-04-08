"""
models/equity_option.py
=======================
EquityOptionTrade — vanilla equity option (call or put).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.trade_base import TradeBase
from models.leg import EquityOptionLeg


@dataclass
class EquityOptionTrade(TradeBase, trade_type="EquityOption"):
    """Vanilla equity option — call or put on a single underlying."""

    tenor_y: int = 0

    @property
    def option_leg(self) -> Optional[EquityOptionLeg]:
        for l in self.legs:
            if isinstance(l, EquityOptionLeg):
                return l
        return None

    @property
    def underlying_ticker(self) -> str:
        ol = self.option_leg
        return ol.underlying_ticker if ol else ""

    @property
    def is_call(self) -> bool:
        ol = self.option_leg
        return ol.option_type == "CALL" if ol else False

    @property
    def is_put(self) -> bool:
        ol = self.option_leg
        return ol.option_type == "PUT" if ol else False

    def _computed_props(self):
        ol = self.option_leg
        if ol is None:
            return {}
        moneyness = ol.strike / ol.initial_price if ol.initial_price else None
        return {
            "equity_option_leg_ref": {
                "_python_class":      "EquityOptionLeg",
                "underlying_ticker":  ol.underlying_ticker,
                "option_type":        ol.option_type,
                "exercise_type":      ol.exercise_type,
                "initial_price":      ol.initial_price,
                "strike":             ol.strike,
                "moneyness_pct":      f"{moneyness * 100:.2f}%" if moneyness else None,
                "vol_pct":            f"{ol.vol * 100:.2f}%",
                "dividend_yield_pct": f"{ol.dividend_yield * 100:.4f}%",
                "risk_free_rate":     ol.risk_free_rate,
                "pricing_model":      ol.pricing_model,
                "n_steps":            ol.n_steps,
                "lot_size":           ol.lot_size,
                "expiry":             str(ol.end_date),
                "notional":           ol.notional,
            },
            "is_call":   self.is_call,
            "is_put":    self.is_put,
            "tenor_y":   self.tenor_y,
        }
