"""
models/enums.py
===============
Shared enumerations for the trade and leg model hierarchy.

All enums inherit from ``str`` so they:
  • serialise naturally to JSON as plain strings
  • compare equal to their string values  (TradeDirection.PAYER == "PAYER")
  • can be reconstructed from strings     (TradeDirection("PAYER") works)
"""

from __future__ import annotations
from enum import Enum


class TradeDirection(str, Enum):
    """Position direction for any trade type."""
    PAYER    = "PAYER"      # IRS: pay fixed / receive float
    RECEIVER = "RECEIVER"   # IRS: receive fixed / pay float
    LONG     = "LONG"       # Bond / option: asset (long) position
    SHORT    = "SHORT"      # Bond / option: liability (short) position
    BUY      = "BUY"        # Option: bought (long premium)
    SELL     = "SELL"       # Option: sold  (short premium)


class LegType(str, Enum):
    """Discriminator for BaseLeg sub-types."""
    FIXED  = "FIXED"
    FLOAT  = "FLOAT"
    BOND   = "BOND"
    OPTION = "OPTION"


class OptionType(str, Enum):
    """Specific option flavour stored on an OptionLeg."""
    PAYER_SWAPTION    = "PAYER_SWAPTION"     # right to enter as fixed payer
    RECEIVER_SWAPTION = "RECEIVER_SWAPTION"  # right to enter as fixed receiver
    CAP               = "CAP"                # cap on floating rate
    FLOOR             = "FLOOR"              # floor on floating rate
    CALL              = "CALL"               # generic call
    PUT               = "PUT"                # generic put


class ExerciseType(str, Enum):
    """Option exercise style."""
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"
    BERMUDAN = "BERMUDAN"
