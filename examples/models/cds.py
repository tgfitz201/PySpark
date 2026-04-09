"""
models/cds.py
=============
CreditDefaultSwap — vanilla CDS (single-name, index, or sovereign).

Inherits from TradeBase:
  trade_id, book, counterparty, valuation_date, direction, legs

direction values
----------------
  TradeDirection.BUY  — protection buyer  (pays spread, receives par on default)
  TradeDirection.SELL — protection seller (receives spread, pays par on default)

Leg structure (2 legs)
----------------------
  legs[0] CreditLeg — protection/contingent leg
                       reference entity, spread, recovery, hazard rate, seniority
  legs[1] FixedLeg  — premium leg (quarterly spread payments)
                       coupon_rate = credit_spread, ACT/360 quarterly

Pricing (QuantLib)
------------------
  • FlatHazardRate derived from CreditLeg.credit_spread & recovery_rate
    hazard ≈ spread / (1 − recovery)
  • MidPointCdsEngine with SOFR discount curve
  • fair_spread() = par CDS spread for zero-NPV at current default probability
  • fixed_npv  = premiumNPV()    (signed: negative for BUY protection buyer)
  • float_npv  = protectionNPV() (signed: positive for BUY protection buyer)

Greeks
------
  cr01           : $ per +1bp credit-spread shift  (analogous to DV01)
  jump_to_default: immediate default P&L
                   buyer : (1−recovery)×notional − accrued_premium
                   seller: −(1−recovery)×notional + accrued_premium
  dv01           : $ per +1bp SOFR shift

Properties
----------
  credit_leg  : CreditLeg → legs[0]
  premium_leg : FixedLeg  → legs[1]
"""

from __future__ import annotations

from dataclasses import dataclass

from models.trade_base import TradeBase
from models.leg import CDSPremiumLeg, CDSProtectionLeg


@dataclass
class CreditDefaultSwap(TradeBase, trade_type="CDS"):
    """
    Single-name or index CDS.

    Leg structure (2 legs)
    ----------------------
    legs[0]  CDSProtectionLeg  — contingent payment on credit event
                                  (1 − recovery_rate) × notional
    legs[1]  CDSPremiumLeg     — quarterly running-spread payments
                                  credit_spread × notional × ACT/360 fraction
    """

    tenor_y: int = 0

    @property
    def protection_leg(self) -> CDSProtectionLeg:
        """Contingent default payment leg (legs[0])."""
        pl = self.legs[0]
        if not isinstance(pl, CDSProtectionLeg):
            raise TypeError(f"legs[0] is {type(pl).__name__}, expected CDSProtectionLeg")
        return pl

    @property
    def premium_leg(self) -> CDSPremiumLeg:
        """Quarterly running-spread payment leg (legs[1])."""
        if len(self.legs) < 2:
            raise TypeError("CDS has no premium leg (legs[1] missing)")
        pl = self.legs[1]
        if not isinstance(pl, CDSPremiumLeg):
            raise TypeError(f"legs[1] is {type(pl).__name__}, expected CDSPremiumLeg")
        return pl

    # Keep credit_leg as an alias for backwards compatibility
    @property
    def credit_leg(self) -> CDSProtectionLeg:
        """Alias for protection_leg (backwards compatibility)."""
        return self.protection_leg

    def _computed_props(self):
        cl = self.protection_leg
        return {
            "protection_leg": {
                "_python_class":       "CDSProtectionLeg",
                "reference_entity":    cl.reference_entity,
                "credit_spread_bps":   f"{cl.credit_spread * 10000:.1f}bps",
                "recovery_rate_pct":   f"{cl.recovery_rate * 100:.1f}%",
                "hazard_rate":         round(cl.hazard_rate, 8),
                "seniority":           cl.seniority,
                "doc_clause":          cl.doc_clause,
                "upfront_fee":         cl.upfront_fee,
                "protection_notional": cl.notional,
                "maturity":            str(cl.end_date),
            },
            "tenor_y":   self.tenor_y,
            "direction": self.direction.value,
        }

    def price(self, curve_df):
        from pricing.cds_pricer import price_cds
        return price_cds(self, curve_df)
