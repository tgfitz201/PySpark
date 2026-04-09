"""
models/callable_bond.py
=======================
Backward-compatibility shim — CallableBond is now OptionableBond with
bond_subtype CALLABLE or PUTABLE.
"""
from __future__ import annotations
from models.optionable_bond import OptionableBond


class CallableBond(OptionableBond, trade_type="CallableBond"):
    """Backward-compat alias for OptionableBond with CALLABLE/PUTABLE subtype."""

    def __post_init__(self):
        if not self.bond_subtype or self.bond_subtype not in ("CALLABLE", "PUTABLE"):
            ol = self.option_leg
            if ol is not None:
                ot = getattr(ol, "option_type", "CALL")
                self.bond_subtype = "CALLABLE" if ot == "CALL" else "PUTABLE"
            else:
                self.bond_subtype = "CALLABLE"
