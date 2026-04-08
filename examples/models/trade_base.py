"""
models/trade_base.py
====================
Abstract base dataclass for all trade types.

Every trade subclass:
  • inherits common identity + direction + legs fields
  • registers itself via the ``trade_type`` keyword so fromJson can dispatch
  • gets toJson / fromJson / toArgList / sparkSchema for free

TradeBase fields
----------------
    trade_id       : unique trade identifier
    book           : trading book / desk
    counterparty   : counterparty name
    valuation_date : pricing reference date
    direction      : TradeDirection enum
                     PAYER / RECEIVER for swaps
                     LONG  / SHORT   for bonds
                     BUY   / SELL    for options
    legs           : List[BaseLeg] — all legs of the trade
                     subclasses fill this in; pricer reads conventions from here
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date
from typing import Any, ClassVar, Dict, List, Optional, Type, get_type_hints
import typing

from models.enums import TradeDirection
from models.leg import (BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg,
                        CreditLeg, CDSPremiumLeg, CDSProtectionLeg, EquityOptionLeg)


# ── Spark type mapping ────────────────────────────────────────────────────────

def _py_to_spark(py_type: Any, nullable: bool = True):
    """Convert a Python type annotation to a PySpark DataType."""
    from pyspark.sql.types import (
        StringType, IntegerType, DoubleType, BooleanType,
        DateType, StructType, StructField, ArrayType,
    )
    from dataclasses import fields as dc_fields

    origin = getattr(py_type, "__origin__", None)

    # Optional[X]
    if origin is typing.Union:
        args = [a for a in py_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return _py_to_spark(args[0], nullable=True)

    # List[X]
    if origin is list and getattr(py_type, "__args__", None):
        elem_spark = _py_to_spark(py_type.__args__[0], nullable=True)
        return ArrayType(elem_spark, True)

    # str enums map to StringType
    if isinstance(py_type, type) and issubclass(py_type, enum.Enum):
        return StringType()

    mapping = {str: StringType(), int: IntegerType(), float: DoubleType(),
               bool: BooleanType(), date: DateType()}
    if py_type in mapping:
        return mapping[py_type]

    # Any dataclass (TradeBase subclass or BaseLeg)
    if is_dataclass(py_type) and isinstance(py_type, type):
        hints = get_type_hints(py_type)
        sub_fields = []
        for f in dc_fields(py_type):
            ftype = hints[f.name]
            is_opt = (getattr(ftype, "__origin__", None) is typing.Union
                      and type(None) in ftype.__args__)
            sub_fields.append(StructField(f.name, _py_to_spark(ftype), is_opt))
        return StructType(sub_fields)

    raise TypeError(f"No Spark type mapping for {py_type!r}")


# ══════════════════════════════════════════════════════════════════════════════
# TradeBase
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeBase:
    """Root dataclass for all financial trade types."""

    trade_id:       str
    book:           str
    counterparty:   str
    valuation_date: date
    direction:      TradeDirection          # replaces old swap_type
    trader:         str = ""               # optional trader / desk identifier
    legs:           List[BaseLeg] = field(default_factory=list)   # all trade legs

    # ── class-level registry ───────────────────────────────────────────────────
    _registry:   ClassVar[Dict[str, Type[TradeBase]]] = {}
    _trade_type: ClassVar[str]                         = "TradeBase"

    def __init_subclass__(cls, trade_type: str = "", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if trade_type:
            cls._trade_type = trade_type
            TradeBase._registry[trade_type] = cls

    # ── serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Recursively convert field values to JSON-serialisable primitives."""
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, TradeBase):
            return value._to_dict()
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: TradeBase._serialize(getattr(value, f.name))
                    for f in fields(value)}
        if isinstance(value, (list, tuple)):
            return [TradeBase._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: TradeBase._serialize(v) for k, v in value.items()}
        return value

    def _to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"trade_type": self._trade_type}
        for f in fields(self):
            d[f.name] = self._serialize(getattr(self, f.name))
        return d

    # ── enriched dict (with _python_class breadcrumbs + computed props) ───────

    @staticmethod
    def _enrich_leg(leg: "BaseLeg") -> Dict[str, Any]:
        """Serialize a leg including its runtime Python class name."""
        d: Dict[str, Any] = {"_python_class": type(leg).__name__}
        for f in fields(leg):
            d[f.name] = TradeBase._serialize(getattr(leg, f.name))
        return d

    def _to_enriched_dict(self) -> Dict[str, Any]:
        """
        Build a fully-resolved dict for pretty-printing.
        Includes:
          _python_class  — runtime class name at trade and leg level
          _computed      — derived/referenced properties (overridden per subclass)
          legs           — each leg serialized with its own _python_class
        """
        d: Dict[str, Any] = {
            "_python_class": type(self).__name__,
            "trade_type":    self._trade_type,
        }
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name == "legs":
                d["legs"] = [self._enrich_leg(leg) for leg in val]
            else:
                d[f.name] = self._serialize(val)
        # subclasses inject computed properties here
        computed = self._computed_props()
        if computed:
            d["_computed"] = computed
        return d

    def _computed_props(self) -> Dict[str, Any]:
        """Override in subclasses to inject derived / referenced properties."""
        return {}

    # ── public API ────────────────────────────────────────────────────────────

    def toJson(self) -> str:
        return json.dumps(self._to_dict(), indent=2)

    def printJson(self, indent: int = 2) -> None:
        """Print a fully-resolved JSON tree following all references and computed properties."""
        print(json.dumps(self._to_enriched_dict(), indent=indent, default=str))

    @classmethod
    def fromJson(cls, json_str: str) -> "TradeBase":
        raw: Dict[str, Any] = json.loads(json_str)
        trade_type = raw.pop("trade_type", None)
        if trade_type is None:
            raise KeyError("JSON missing required key \'trade_type\'")
        target = cls._registry.get(trade_type)
        if target is None:
            raise KeyError(f"Unknown trade_type {trade_type!r}. "
                           f"Registered: {sorted(cls._registry)}")
        return target._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TradeBase":
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            raw_val = data.get(f.name)
            py_type = hints.get(f.name)
            kwargs[f.name] = cls._coerce(raw_val, py_type)
        return cls(**kwargs)

    @staticmethod
    def _make_leg(data: dict) -> BaseLeg:
        """Dispatch leg construction by leg_type field."""
        lt = data.get("leg_type", "FIXED")
        if lt == "FIXED":
            return TradeBase._reconstruct(FixedLeg, data)
        if lt == "FLOAT":
            return TradeBase._reconstruct(FloatLeg, data)
        if lt == "OPTION":
            return TradeBase._reconstruct(OptionLeg, data)
        if lt == "EQUITY":
            return TradeBase._reconstruct(EquityLeg, data)
        if lt == "CDS_PREMIUM":
            return TradeBase._reconstruct(CDSPremiumLeg, data)
        if lt == "CDS_PROTECTION":
            return TradeBase._reconstruct(CDSProtectionLeg, data)
        if lt == "CREDIT":
            return TradeBase._reconstruct(CreditLeg, data)
        if lt == "EQUITY_OPTION":
            return TradeBase._reconstruct(EquityOptionLeg, data)
        return TradeBase._reconstruct(BaseLeg, data)

    @staticmethod
    def _coerce(value: Any, py_type: Any) -> Any:
        """Convert a raw JSON value to the declared Python type."""
        if value is None:
            return None

        origin = getattr(py_type, "__origin__", None)

        # Unwrap Optional[X]
        if origin is typing.Union:
            inner = [a for a in py_type.__args__ if a is not type(None)]
            if len(inner) == 1:
                return TradeBase._coerce(value, inner[0])

        # List[BaseLeg] or other List[X]
        if origin is list and isinstance(value, list):
            if getattr(py_type, "__args__", None):
                elem_t = py_type.__args__[0]
                # Check if element type is BaseLeg or a subclass
                try:
                    if elem_t is BaseLeg or (isinstance(elem_t, type) and issubclass(elem_t, BaseLeg)):
                        return [TradeBase._make_leg(v) if isinstance(v, dict) else v
                                for v in value]
                except TypeError:
                    pass
                return [TradeBase._coerce(v, elem_t) for v in value]
            return value

        # Enum types
        if isinstance(py_type, type) and issubclass(py_type, enum.Enum):
            if isinstance(value, py_type):
                return value
            return py_type(value)

        # date from ISO string
        if py_type is date and isinstance(value, str):
            return date.fromisoformat(value)

        # Nested TradeBase — dispatch via trade_type
        try:
            if isinstance(value, dict) and issubclass(py_type, TradeBase):
                nested_type = value.pop("trade_type", py_type._trade_type)
                target = TradeBase._registry.get(nested_type, py_type)
                return target._from_dict(value)
        except TypeError:
            pass

        # Generic nested dataclass (BaseLeg etc.)
        if isinstance(value, dict) and is_dataclass(py_type):
            return TradeBase._reconstruct(py_type, value)

        return value

    @staticmethod
    def _reconstruct(cls: type, data: Dict[str, Any]) -> Any:
        """Reconstruct any dataclass from a plain dict, coercing field types."""
        hints = get_type_hints(cls)
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            raw = data.get(f.name)
            kwargs[f.name] = TradeBase._coerce(raw, hints.get(f.name, type(raw)))
        return cls(**kwargs)

    def toArgList(self) -> List[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def sparkSchema(cls) -> "pyspark.sql.types.StructType":  # noqa: F821
        from pyspark.sql.types import StructType, StructField
        hints = get_type_hints(cls)
        schema_fields = []
        for f in fields(cls):
            py_type = hints[f.name]
            nullable = (getattr(py_type, "__origin__", None) is typing.Union
                        and type(None) in py_type.__args__)
            spark_type = _py_to_spark(py_type, nullable=nullable)
            schema_fields.append(StructField(f.name, spark_type, nullable))
        return StructType(schema_fields)

    def __repr__(self) -> str:
        parts = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
        return f"{type(self).__name__}({parts})"

    @property
    def ref(self) -> "TradeReference":
        """Return a TradeReference aggregating this trade's identity fields."""
        from models.trade_reference import TradeReference  # avoid circular at module level
        return TradeReference(
            trade_id=self.trade_id,
            book=self.book,
            counterparty=self.counterparty,
            valuation_date=self.valuation_date,
            trader=self.trader,
        )
