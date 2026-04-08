"""
models/trade_reference.py
=========================
TradeReference — immutable reference / identifier object for any trade.

Separates trade identity (who, when, where) from trade economics (what).
Used as the primary key for CRUD operations in the trade repository.

Fields
------
    trade_id       : system-generated unique identifier  e.g. "IRS-0001"
    book           : trading book / desk                  e.g. "IRD-NY"
    counterparty   : counterparty legal name              e.g. "CPTY-01"
    valuation_date : pricing reference date
    portfolio      : optional portfolio name              e.g. "RATES-BOOK"
    strategy       : optional strategy tag                e.g. "CARRY"
    trader         : optional trader identifier           e.g. "T001"
    legal_entity   : optional internal legal entity       e.g. "GSNY"

Usage
-----
>>> ref = TradeReference("IRS-001", book="IRD-NY", counterparty="CPTY-01",
...                      valuation_date=date(2025, 1, 15))
>>> ref.trade_id
'IRS-001'
>>> import json; ref2 = TradeReference.fromJson(ref.toJson()); ref == ref2
True
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class TradeReference:
    """Immutable trade identity — the primary key for all trade CRUD operations."""

    trade_id:       str
    book:           str           = ""
    counterparty:   str           = ""
    valuation_date: Optional[date] = None
    portfolio:      str           = ""
    strategy:       str           = ""
    trader:         str           = ""
    legal_entity:   str           = ""

    # ── serialisation ────────────────────────────────────────────────────────

    def toJson(self) -> str:
        d = asdict(self)
        if self.valuation_date is not None:
            d["valuation_date"] = self.valuation_date.isoformat()
        return json.dumps(d)

    @classmethod
    def fromJson(cls, s: str) -> "TradeReference":
        d = json.loads(s)
        if d.get("valuation_date"):
            d["valuation_date"] = date.fromisoformat(d["valuation_date"])
        return cls(**d)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"TradeReference({self.trade_id!r}, book={self.book!r}, "
                f"counterparty={self.counterparty!r}, "
                f"valuation_date={self.valuation_date})")
