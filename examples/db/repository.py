"""
db/repository.py
================
TradeRepository — SQLite-backed CRUD store for all TradeBase subclasses.

Class-Table-Inheritance (CTI) schema: every table name maps to a Python class name.
Old tables (trades, trade_legs, pricing_results) are preserved for backward compat.

New tables:
  TradeBase, VanillaSwap, Bond, OptionTrade, EquitySwap,
  CreditDefaultSwap, EquityOptionTrade
  BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg, CreditLeg, EquityOptionLeg
  PricingResult

API
---
  upsert(trade)              → None
  upsert_many(trades)        → int
  get(trade_id)              → Optional[TradeBase]
  list_all(trade_type=None)  → List[TradeBase]
  save_results(results, run_id)  → int   [accepts List[PricingResult]]
  clear_results()            → None
  get_results_df(run_id=None) → pd.DataFrame
  count(trade_type=None)     → int
  delete(trade_id)           → bool
  delete_all()               → int
  close()                    → None
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_LEGACY = """
-- ─── Legacy tables (preserved for backward compat) ───────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT    PRIMARY KEY,
    trade_type      TEXT    NOT NULL,
    book            TEXT    DEFAULT '',
    counterparty    TEXT    DEFAULT '',
    valuation_date  TEXT,
    direction       TEXT    DEFAULT '',
    trade_json      TEXT    NOT NULL,
    created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    updated_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);
CREATE TABLE IF NOT EXISTS trade_legs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    TEXT    NOT NULL REFERENCES trades(trade_id) ON DELETE CASCADE,
    leg_index   INTEGER NOT NULL,
    leg_type    TEXT    NOT NULL,
    leg_json    TEXT    NOT NULL,
    UNIQUE(trade_id, leg_index)
);
CREATE TABLE IF NOT EXISTS pricing_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT    NOT NULL,
    run_id          TEXT    NOT NULL DEFAULT 'DEFAULT',
    run_timestamp   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    instrument      TEXT,
    book            TEXT,
    counterparty    TEXT,
    direction       TEXT,
    tenor_y         REAL,
    notional        REAL,
    coupon_rate     REAL,
    swap_npv        REAL,
    par_rate        REAL,
    clean_price     REAL,
    accrued         REAL,
    fixed_npv       REAL,
    float_npv       REAL,
    dv01            REAL,
    duration        REAL,
    pv01            REAL,
    convexity       REAL,
    vega            REAL,
    theta           REAL,
    delta           REAL,
    cr01            REAL,
    jump_to_default REAL,
    error           TEXT    DEFAULT '',
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);
CREATE INDEX IF NOT EXISTS idx_results_trade ON pricing_results(trade_id);
CREATE INDEX IF NOT EXISTS idx_results_run   ON pricing_results(run_id);
CREATE INDEX IF NOT EXISTS idx_legs_trade    ON trade_legs(trade_id);
"""

_DDL_CTI = """
-- ─── Class-Table-Inheritance schema ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS TradeBase (
    trade_id       TEXT PRIMARY KEY,
    trade_type     TEXT NOT NULL,
    book           TEXT DEFAULT '',
    counterparty   TEXT DEFAULT '',
    valuation_date TEXT NOT NULL,
    direction      TEXT NOT NULL,
    created_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    updated_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
);

CREATE TABLE IF NOT EXISTS VanillaSwap (
    trade_id     TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y      INTEGER DEFAULT 0,
    swap_subtype TEXT    DEFAULT 'FIXED_FLOAT'
);
CREATE TABLE IF NOT EXISTS InterestRateSwap (
    trade_id    TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y     INTEGER DEFAULT 0,
    swap_subtype TEXT DEFAULT 'FIXED_FLOAT',
    fx_rate     REAL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS CrossCurrencySwap (
    trade_id                   TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y                    INTEGER DEFAULT 0,
    pay_currency               TEXT DEFAULT 'USD',
    receive_currency           TEXT DEFAULT 'EUR',
    fx_rate                    REAL DEFAULT 1.0,
    initial_notional_exchange  INTEGER DEFAULT 1,
    final_notional_exchange    INTEGER DEFAULT 1,
    swap_subtype               TEXT DEFAULT 'FIXED_FLOAT'
);
CREATE TABLE IF NOT EXISTS Bond (
    trade_id TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y  INTEGER DEFAULT 0,
    isin     TEXT
);
CREATE TABLE IF NOT EXISTS OptionTrade (
    trade_id           TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y            INTEGER DEFAULT 0,
    underlying_tenor_y INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS EquitySwap (
    trade_id          TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y           INTEGER DEFAULT 0,
    underlying_ticker TEXT    DEFAULT ''
);
CREATE TABLE IF NOT EXISTS CreditDefaultSwap (
    trade_id TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y  INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS EquityOptionTrade (
    trade_id TEXT PRIMARY KEY REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    tenor_y  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS BaseLeg (
    leg_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT    NOT NULL REFERENCES TradeBase(trade_id) ON DELETE CASCADE,
    leg_index       INTEGER NOT NULL,
    leg_type        TEXT    NOT NULL,
    notional        REAL    NOT NULL,
    start_date      TEXT    NOT NULL,
    end_date        TEXT    NOT NULL,
    currency        TEXT    DEFAULT 'USD',
    calendar        TEXT    DEFAULT 'US_GOVT',
    day_count       TEXT    DEFAULT '30/360',
    bdc             TEXT    DEFAULT 'MOD_FOLLOWING',
    frequency       TEXT    DEFAULT 'SEMIANNUAL',
    coupon_rate     REAL    DEFAULT 0.0,
    spread          REAL    DEFAULT 0.0,
    index_name      TEXT    DEFAULT 'SOFR3M',
    index_tenor_m   INTEGER DEFAULT 3,
    fixing_lag      INTEGER DEFAULT 2,
    redemption      REAL    DEFAULT 100.0,
    settlement_days INTEGER DEFAULT 2,
    issue_date      TEXT,
    UNIQUE(trade_id, leg_index)
);
CREATE TABLE IF NOT EXISTS FixedLeg (
    leg_id      INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    coupon_rate REAL    DEFAULT 0.0,
    day_count   TEXT    DEFAULT '30/360',
    frequency   TEXT    DEFAULT 'SEMIANNUAL'
);
CREATE TABLE IF NOT EXISTS FloatLeg (
    leg_id        INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    spread        REAL    DEFAULT 0.0,
    index_name    TEXT    DEFAULT 'SOFR3M',
    index_tenor_m INTEGER DEFAULT 3,
    fixing_lag    INTEGER DEFAULT 2,
    day_count     TEXT    DEFAULT 'ACT/360',
    frequency     TEXT    DEFAULT 'QUARTERLY'
);
CREATE TABLE IF NOT EXISTS OptionLeg (
    leg_id              INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    strike              REAL    DEFAULT 0.05,
    option_type         TEXT    DEFAULT 'PAYER_SWAPTION',
    exercise_type       TEXT    DEFAULT 'EUROPEAN',
    vol                 REAL    DEFAULT 0.40,
    vol_type            TEXT    DEFAULT 'LOGNORMAL',
    vol_shift           REAL    DEFAULT 0.03,
    underlying_tenor_m  INTEGER DEFAULT 60,
    underlying_type     TEXT    DEFAULT 'SWAP'
);
CREATE TABLE IF NOT EXISTS EquityLeg (
    leg_id             INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    underlying_ticker  TEXT    DEFAULT 'SPY',
    initial_price      REAL    DEFAULT 100.0,
    dividend_yield     REAL    DEFAULT 0.015,
    equity_return_type TEXT    DEFAULT 'TOTAL',
    reset_frequency    TEXT    DEFAULT 'ANNUAL',
    participation_rate REAL    DEFAULT 1.0,
    funding_spread     REAL    DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS CreditLeg (
    leg_id           INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    reference_entity TEXT    DEFAULT 'CORP',
    credit_spread    REAL    DEFAULT 0.015,
    recovery_rate    REAL    DEFAULT 0.40,
    hazard_rate      REAL    DEFAULT 0.0,
    seniority        TEXT    DEFAULT 'SENIOR_UNSECURED',
    doc_clause       TEXT    DEFAULT 'CR',
    upfront_fee      REAL    DEFAULT 0.0,
    step_up_spread   REAL    DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS EquityOptionLeg (
    leg_id            INTEGER PRIMARY KEY REFERENCES BaseLeg(leg_id) ON DELETE CASCADE,
    underlying_ticker TEXT    DEFAULT 'SPY',
    initial_price     REAL    DEFAULT 100.0,
    strike            REAL    DEFAULT 100.0,
    option_type       TEXT    DEFAULT 'CALL',
    exercise_type     TEXT    DEFAULT 'EUROPEAN',
    vol               REAL    DEFAULT 0.25,
    dividend_yield    REAL    DEFAULT 0.013,
    risk_free_rate    REAL    DEFAULT 0.0,
    pricing_model     TEXT    DEFAULT 'BLACK_SCHOLES',
    n_steps           INTEGER DEFAULT 200,
    lot_size          INTEGER DEFAULT 100
);

CREATE TABLE IF NOT EXISTS PricingResult (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT    NOT NULL,
    run_id          TEXT    NOT NULL DEFAULT 'DEFAULT',
    run_timestamp   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now')),
    instrument      TEXT,
    book            TEXT,
    counterparty    TEXT,
    direction       TEXT,
    swap_subtype    TEXT,
    tenor_y         REAL,
    notional        REAL,
    coupon_rate     REAL,
    npv             REAL,
    fixed_npv       REAL,
    float_npv       REAL,
    par_rate        REAL,
    clean_price     REAL,
    accrued         REAL,
    premium         REAL,
    dv01            REAL,
    duration        REAL,
    pv01            REAL,
    convexity       REAL,
    vega            REAL,
    theta           REAL,
    delta           REAL,
    gamma           REAL,
    rho             REAL,
    cr01            REAL,
    jump_to_default REAL,
    option_type       TEXT,
    exercise_type     TEXT,
    strike            REAL,
    vol               REAL,
    underlying_ticker TEXT,
    valuation_date    TEXT,
    error           TEXT    DEFAULT '',
    UNIQUE(trade_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_tradebase_type ON TradeBase(trade_type);
CREATE INDEX IF NOT EXISTS idx_baseleg_trade  ON BaseLeg(trade_id);
CREATE INDEX IF NOT EXISTS idx_pr_run         ON PricingResult(run_id);
CREATE INDEX IF NOT EXISTS idx_pr_trade_run   ON PricingResult(trade_id, run_id);
"""

# Maps _trade_type value → subclass table name
_TRADE_TYPE_TO_SUBTABLE: Dict[str, str] = {
    "VanillaSwap":       "VanillaSwap",
    "InterestRateSwap":  "InterestRateSwap",
    "CrossCurrencySwap": "CrossCurrencySwap",
    "Bond":              "Bond",
    "CallableBond":      "Bond",    # shares Bond subtable (tenor_y, isin)
    "Option":            "OptionTrade",
    "IRSwaption":        "OptionTrade",   # shares OptionTrade subtable
    "EquitySwap":        "EquitySwap",
    "CDS":               "CreditDefaultSwap",
    "EquityOption":      "EquityOptionTrade",
}

# Leg type → subclass table name (only types with extra tables)
_LEG_TYPE_TO_SUBTABLE: Dict[str, str] = {
    "FIXED":         "FixedLeg",
    "FLOAT":         "FloatLeg",
    "OPTION":        "OptionLeg",
    "EQUITY":        "EquityLeg",
    "CREDIT":        "CreditLeg",
    "EQUITY_OPTION": "EquityOptionLeg",
}


def _parse_date(s) -> Optional[date]:
    if s is None or s == "":
        return None
    if isinstance(s, date):
        return s
    try:
        return date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def _row_to_leg(row: dict):
    """Reconstruct a leg object from a joined row dict."""
    from models.leg import (BaseLeg, FixedLeg, FloatLeg, OptionLeg, EquityLeg,
                            CreditLeg, CDSPremiumLeg, CDSProtectionLeg, EquityOptionLeg)

    leg_type = row["leg_type"]

    common = dict(
        leg_type=leg_type,
        notional=float(row["notional"]),
        start_date=_parse_date(row["start_date"]),
        end_date=_parse_date(row["end_date"]),
        currency=row.get("currency") or "USD",
        calendar=row.get("calendar") or "US_GOVT",
        day_count=row.get("day_count") or "30/360",
        bdc=row.get("bdc") or "MOD_FOLLOWING",
        frequency=row.get("frequency") or "SEMIANNUAL",
        coupon_rate=float(row.get("coupon_rate") or 0.0),
        spread=float(row.get("spread") or 0.0),
        index_name=row.get("index_name") or "SOFR3M",
        index_tenor_m=int(row.get("index_tenor_m") or 3),
        fixing_lag=int(row.get("fixing_lag") or 2),
        redemption=float(row.get("redemption") or 100.0),
        settlement_days=int(row.get("settlement_days") or 2),
        issue_date=_parse_date(row.get("issue_date")),
    )

    if leg_type == "FIXED":
        return FixedLeg(**common)
    if leg_type == "FLOAT":
        return FloatLeg(**common)
    if leg_type == "OPTION":
        return OptionLeg(
            **common,
            strike=float(row.get("strike") or 0.05),
            option_type=row.get("option_type") or "PAYER_SWAPTION",
            exercise_type=row.get("exercise_type") or "EUROPEAN",
            vol=float(row.get("vol") or 0.40),
            vol_type=row.get("vol_type") or "LOGNORMAL",
            vol_shift=float(row.get("vol_shift") or 0.03),
            underlying_tenor_m=int(row.get("underlying_tenor_m") or 60),
            underlying_type=row.get("underlying_type") or "SWAP",
        )
    if leg_type == "EQUITY":
        return EquityLeg(
            **common,
            underlying_ticker=row.get("el_underlying_ticker") or "SPY",
            initial_price=float(row.get("el_initial_price") or 100.0),
            dividend_yield=float(row.get("el_dividend_yield") or 0.015),
            equity_return_type=row.get("equity_return_type") or "TOTAL",
            reset_frequency=row.get("reset_frequency") or "ANNUAL",
            participation_rate=float(row.get("participation_rate") or 1.0),
            funding_spread=float(row.get("funding_spread") or 0.0),
        )
    if leg_type in ("CREDIT", "CDS_PROTECTION"):
        # CDS_PROTECTION (CDSProtectionLeg) and legacy CREDIT (CreditLeg) share
        # the same DB columns — reconstruct as the typed subclass.
        cls = CDSProtectionLeg if leg_type == "CDS_PROTECTION" else CreditLeg
        return cls(
            **common,
            reference_entity=row.get("reference_entity") or "CORP",
            credit_spread=float(row.get("credit_spread") or 0.015),
            recovery_rate=float(row.get("recovery_rate") or 0.40),
            hazard_rate=float(row.get("hazard_rate") or 0.0),
            seniority=row.get("seniority") or "SENIOR_UNSECURED",
            doc_clause=row.get("doc_clause") or "CR",
            upfront_fee=float(row.get("upfront_fee") or 0.0),
            step_up_spread=float(row.get("step_up_spread") or 0.0),
        )
    if leg_type == "CDS_PREMIUM":
        # CDSPremiumLeg — a FixedLeg subclass with ACT/360 quarterly defaults.
        # accrued_on_default is not persisted in a separate column; use default True.
        return CDSPremiumLeg(**common)
    if leg_type == "EQUITY_OPTION":
        return EquityOptionLeg(
            **common,
            underlying_ticker=row.get("eol_underlying_ticker") or "SPY",
            initial_price=float(row.get("eol_initial_price") or 100.0),
            strike=float(row.get("eol_strike") or 100.0),
            option_type=row.get("eol_option_type") or "CALL",
            exercise_type=row.get("eol_exercise_type") or "EUROPEAN",
            vol=float(row.get("eol_vol") or 0.25),
            dividend_yield=float(row.get("eol_dividend_yield") or 0.013),
            risk_free_rate=float(row.get("risk_free_rate") or 0.0),
            pricing_model=row.get("pricing_model") or "BLACK_SCHOLES",
            n_steps=int(row.get("n_steps") or 200),
            lot_size=int(row.get("lot_size") or 100),
        )
    # BOND and any unrecognised types
    return BaseLeg(**common)


_LEG_JOIN_SQL = """
SELECT
    bl.leg_id, bl.trade_id, bl.leg_index, bl.leg_type,
    bl.notional, bl.start_date, bl.end_date,
    bl.currency, bl.calendar, bl.day_count, bl.bdc, bl.frequency,
    bl.coupon_rate, bl.spread, bl.index_name, bl.index_tenor_m, bl.fixing_lag,
    bl.redemption, bl.settlement_days, bl.issue_date,
    fl.coupon_rate AS fl_coupon_rate, fl.day_count AS fl_day_count, fl.frequency AS fl_frequency,
    fll.spread AS fll_spread, fll.index_name AS fll_index_name,
    fll.index_tenor_m AS fll_index_tenor_m, fll.fixing_lag AS fll_fixing_lag,
    fll.day_count AS fll_day_count, fll.frequency AS fll_frequency,
    ol.strike, ol.option_type, ol.exercise_type, ol.vol, ol.vol_type,
    ol.vol_shift, ol.underlying_tenor_m, ol.underlying_type,
    el.underlying_ticker AS el_underlying_ticker, el.initial_price AS el_initial_price,
    el.dividend_yield AS el_dividend_yield, el.equity_return_type,
    el.reset_frequency, el.participation_rate, el.funding_spread,
    cl.reference_entity, cl.credit_spread, cl.recovery_rate, cl.hazard_rate,
    cl.seniority, cl.doc_clause, cl.upfront_fee, cl.step_up_spread,
    eol.underlying_ticker AS eol_underlying_ticker, eol.initial_price AS eol_initial_price,
    eol.strike AS eol_strike, eol.option_type AS eol_option_type,
    eol.exercise_type AS eol_exercise_type, eol.vol AS eol_vol,
    eol.dividend_yield AS eol_dividend_yield, eol.risk_free_rate,
    eol.pricing_model, eol.n_steps, eol.lot_size
FROM BaseLeg bl
LEFT JOIN FixedLeg fl  ON bl.leg_id = fl.leg_id
LEFT JOIN FloatLeg fll ON bl.leg_id = fll.leg_id
LEFT JOIN OptionLeg ol ON bl.leg_id = ol.leg_id
LEFT JOIN EquityLeg el ON bl.leg_id = el.leg_id
LEFT JOIN CreditLeg cl ON bl.leg_id = cl.leg_id
LEFT JOIN EquityOptionLeg eol ON bl.leg_id = eol.leg_id
WHERE bl.trade_id = ?
ORDER BY bl.leg_index
"""


# ── Repository ────────────────────────────────────────────────────────────────

class TradeRepository:
    """
    SQLite-backed CRUD repository for all TradeBase subclasses.

    Uses class-table-inheritance (CTI): every table name maps to a Python class.

    Parameters
    ----------
    db_path : str
        Path to SQLite database file.  Use ":memory:" for in-process testing.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._ensure_schema()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _tx(self):
        """Context manager that commits on success or rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _ensure_schema(self) -> None:
        self._conn.executescript(_DDL_LEGACY)
        self._conn.executescript(_DDL_CTI)
        self._conn.commit()
        # Migrate PricingResult: add columns that may not exist in older DBs
        for col, defn in [
            ("option_type",       "TEXT"),
            ("exercise_type",     "TEXT"),
            ("strike",            "REAL"),
            ("vol",               "REAL"),
            ("underlying_ticker", "TEXT"),
            ("valuation_date",    "TEXT"),
        ]:
            try:
                self._conn.execute(f"ALTER TABLE PricingResult ADD COLUMN {col} {defn}")
            except Exception:
                pass  # column already exists
        # Migrate TradeBase: add trader column if missing
        try:
            self._conn.execute("ALTER TABLE TradeBase ADD COLUMN trader TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists
        # Migrate InterestRateSwap: add fx_rate column for XCCY support
        try:
            self._conn.execute("ALTER TABLE InterestRateSwap ADD COLUMN fx_rate REAL DEFAULT 1.0")
        except Exception:
            pass  # column already exists
        self._conn.commit()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _direction_str(trade) -> str:
        d = trade.direction
        return d.value if hasattr(d, "value") else str(d)

    @staticmethod
    def _val_date_str(trade) -> str:
        vd = trade.valuation_date
        return vd.isoformat() if isinstance(vd, date) else str(vd)

    # ── CREATE / UPDATE ───────────────────────────────────────────────────────

    def upsert(self, trade) -> None:
        """Insert or replace a trade (and its legs) using CTI tables."""
        from dataclasses import fields as dc_fields

        trade_type  = trade._trade_type
        subtable    = _TRADE_TYPE_TO_SUBTABLE.get(trade_type)
        val_date    = self._val_date_str(trade)
        direction   = self._direction_str(trade)
        trade_id    = trade.trade_id

        with self._tx():
            # 1. TradeBase
            self._conn.execute(
                """
                INSERT INTO TradeBase
                    (trade_id, trade_type, book, counterparty, trader,
                     valuation_date, direction, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%S','now'))
                ON CONFLICT(trade_id) DO UPDATE SET
                    trade_type     = excluded.trade_type,
                    book           = excluded.book,
                    counterparty   = excluded.counterparty,
                    trader         = excluded.trader,
                    valuation_date = excluded.valuation_date,
                    direction      = excluded.direction,
                    updated_at     = excluded.updated_at
                """,
                (trade_id, trade_type, trade.book, trade.counterparty,
                 getattr(trade, "trader", ""), val_date, direction),
            )

            # 2. Subclass table
            if subtable == "VanillaSwap":
                self._conn.execute(
                    "INSERT OR REPLACE INTO VanillaSwap (trade_id, tenor_y, swap_subtype) VALUES (?, ?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0), getattr(trade, "swap_subtype", "FIXED_FLOAT")),
                )
            elif subtable == "InterestRateSwap":
                self._conn.execute(
                    "INSERT OR REPLACE INTO InterestRateSwap (trade_id, tenor_y, swap_subtype, fx_rate) VALUES (?, ?, ?, ?)",
                    (trade.trade_id, getattr(trade, "tenor_y", 0),
                     getattr(trade, "swap_subtype", "FIXED_FLOAT"),
                     float(getattr(trade, "fx_rate", 1.0))),
                )
            elif subtable == "CrossCurrencySwap":
                self._conn.execute(
                    "INSERT OR REPLACE INTO CrossCurrencySwap "
                    "(trade_id, tenor_y, pay_currency, receive_currency, fx_rate, "
                    " initial_notional_exchange, final_notional_exchange, swap_subtype) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        trade.trade_id,
                        getattr(trade, "tenor_y", 0),
                        getattr(trade, "pay_currency", "USD"),
                        getattr(trade, "receive_currency", "EUR"),
                        float(getattr(trade, "fx_rate", 1.0)),
                        int(getattr(trade, "initial_notional_exchange", True)),
                        int(getattr(trade, "final_notional_exchange", True)),
                        getattr(trade, "swap_subtype", "FIXED_FLOAT"),
                    ),
                )
            elif subtable == "Bond":
                self._conn.execute(
                    "INSERT OR REPLACE INTO Bond (trade_id, tenor_y, isin) VALUES (?, ?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0), getattr(trade, "isin", None)),
                )
            elif subtable == "OptionTrade":
                self._conn.execute(
                    "INSERT OR REPLACE INTO OptionTrade (trade_id, tenor_y, underlying_tenor_y) VALUES (?, ?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0), getattr(trade, "underlying_tenor_y", 0)),
                )
            elif subtable == "EquitySwap":
                self._conn.execute(
                    "INSERT OR REPLACE INTO EquitySwap (trade_id, tenor_y, underlying_ticker) VALUES (?, ?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0), getattr(trade, "underlying_ticker", "")),
                )
            elif subtable == "CreditDefaultSwap":
                self._conn.execute(
                    "INSERT OR REPLACE INTO CreditDefaultSwap (trade_id, tenor_y) VALUES (?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0)),
                )
            elif subtable == "EquityOptionTrade":
                self._conn.execute(
                    "INSERT OR REPLACE INTO EquityOptionTrade (trade_id, tenor_y) VALUES (?, ?)",
                    (trade_id, getattr(trade, "tenor_y", 0)),
                )

            # 3. Refresh legs
            self._conn.execute("DELETE FROM BaseLeg WHERE trade_id = ?", (trade_id,))

            # 4+5. Insert each leg
            for idx, leg in enumerate(trade.legs):
                sd = leg.start_date.isoformat() if isinstance(leg.start_date, date) else str(leg.start_date)
                ed = leg.end_date.isoformat()   if isinstance(leg.end_date, date)   else str(leg.end_date)
                id_str = (leg.issue_date.isoformat() if isinstance(leg.issue_date, date)
                          else (str(leg.issue_date) if leg.issue_date else None))

                cur = self._conn.execute(
                    """
                    INSERT INTO BaseLeg (
                        trade_id, leg_index, leg_type,
                        notional, start_date, end_date,
                        currency, calendar, day_count, bdc, frequency,
                        coupon_rate, spread, index_name, index_tenor_m, fixing_lag,
                        redemption, settlement_days, issue_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trade_id, idx, leg.leg_type,
                     leg.notional, sd, ed,
                     getattr(leg, "currency", "USD"),
                     getattr(leg, "calendar", "US_GOVT"),
                     getattr(leg, "day_count", "30/360"),
                     getattr(leg, "bdc", "MOD_FOLLOWING"),
                     getattr(leg, "frequency", "SEMIANNUAL"),
                     getattr(leg, "coupon_rate", 0.0),
                     getattr(leg, "spread", 0.0),
                     getattr(leg, "index_name", "SOFR3M"),
                     getattr(leg, "index_tenor_m", 3),
                     getattr(leg, "fixing_lag", 2),
                     getattr(leg, "redemption", 100.0),
                     getattr(leg, "settlement_days", 2),
                     id_str),
                )
                leg_id = cur.lastrowid

                # Subclass leg tables
                lt = leg.leg_type
                if lt == "FIXED":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO FixedLeg
                           (leg_id, coupon_rate, day_count, frequency)
                           VALUES (?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "coupon_rate", 0.0),
                         getattr(leg, "day_count", "30/360"),
                         getattr(leg, "frequency", "SEMIANNUAL")),
                    )
                elif lt == "FLOAT":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO FloatLeg
                           (leg_id, spread, index_name, index_tenor_m, fixing_lag, day_count, frequency)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "spread", 0.0),
                         getattr(leg, "index_name", "SOFR3M"),
                         getattr(leg, "index_tenor_m", 3),
                         getattr(leg, "fixing_lag", 2),
                         getattr(leg, "day_count", "ACT/360"),
                         getattr(leg, "frequency", "QUARTERLY")),
                    )
                elif lt == "OPTION":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO OptionLeg
                           (leg_id, strike, option_type, exercise_type,
                            vol, vol_type, vol_shift, underlying_tenor_m, underlying_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "strike", 0.05),
                         getattr(leg, "option_type", "PAYER_SWAPTION"),
                         getattr(leg, "exercise_type", "EUROPEAN"),
                         getattr(leg, "vol", 0.40),
                         getattr(leg, "vol_type", "LOGNORMAL"),
                         getattr(leg, "vol_shift", 0.03),
                         getattr(leg, "underlying_tenor_m", 60),
                         getattr(leg, "underlying_type", "SWAP")),
                    )
                elif lt == "EQUITY":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO EquityLeg
                           (leg_id, underlying_ticker, initial_price, dividend_yield,
                            equity_return_type, reset_frequency, participation_rate, funding_spread)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "underlying_ticker", "SPY"),
                         getattr(leg, "initial_price", 100.0),
                         getattr(leg, "dividend_yield", 0.015),
                         getattr(leg, "equity_return_type", "TOTAL"),
                         getattr(leg, "reset_frequency", "ANNUAL"),
                         getattr(leg, "participation_rate", 1.0),
                         getattr(leg, "funding_spread", 0.0)),
                    )
                elif lt == "CREDIT":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO CreditLeg
                           (leg_id, reference_entity, credit_spread, recovery_rate,
                            hazard_rate, seniority, doc_clause, upfront_fee, step_up_spread)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "reference_entity", "CORP"),
                         getattr(leg, "credit_spread", 0.015),
                         getattr(leg, "recovery_rate", 0.40),
                         getattr(leg, "hazard_rate", 0.0),
                         getattr(leg, "seniority", "SENIOR_UNSECURED"),
                         getattr(leg, "doc_clause", "CR"),
                         getattr(leg, "upfront_fee", 0.0),
                         getattr(leg, "step_up_spread", 0.0)),
                    )
                elif lt == "EQUITY_OPTION":
                    self._conn.execute(
                        """INSERT OR REPLACE INTO EquityOptionLeg
                           (leg_id, underlying_ticker, initial_price, strike,
                            option_type, exercise_type, vol, dividend_yield,
                            risk_free_rate, pricing_model, n_steps, lot_size)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (leg_id,
                         getattr(leg, "underlying_ticker", "SPY"),
                         getattr(leg, "initial_price", 100.0),
                         getattr(leg, "strike", 100.0),
                         getattr(leg, "option_type", "CALL"),
                         getattr(leg, "exercise_type", "EUROPEAN"),
                         getattr(leg, "vol", 0.25),
                         getattr(leg, "dividend_yield", 0.013),
                         getattr(leg, "risk_free_rate", 0.0),
                         getattr(leg, "pricing_model", "BLACK_SCHOLES"),
                         getattr(leg, "n_steps", 200),
                         getattr(leg, "lot_size", 100)),
                    )

    def upsert_many(self, trades: list) -> int:
        """Bulk upsert; returns count."""
        for t in trades:
            self.upsert(t)
        return len(trades)

    # ── READ ──────────────────────────────────────────────────────────────────

    def get(self, trade_id: str):
        """Return a single trade by trade_id, or None if not found."""
        # 1. Fetch TradeBase row
        base_row = self._conn.execute(
            "SELECT trade_id, trade_type, book, counterparty, trader, valuation_date, direction "
            "FROM TradeBase WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        if base_row is None:
            return None

        base = dict(base_row)
        trade_type = base["trade_type"]
        subtable   = _TRADE_TYPE_TO_SUBTABLE.get(trade_type)

        # 2. Fetch subclass row
        sub = {}
        if subtable:
            sub_row = self._conn.execute(
                f"SELECT * FROM {subtable} WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if sub_row:
                sub = dict(sub_row)

        # 3. Fetch legs via big JOIN
        leg_rows = self._conn.execute(_LEG_JOIN_SQL, (trade_id,)).fetchall()
        legs = [_row_to_leg(dict(r)) for r in leg_rows]

        # 4. Reconstruct Python object
        from models.enums import TradeDirection
        val_date  = _parse_date(base["valuation_date"]) or date.today()
        direction = TradeDirection(base["direction"])

        common = dict(
            trade_id=trade_id,
            book=base.get("book", ""),
            counterparty=base.get("counterparty", ""),
            trader=base.get("trader", ""),
            valuation_date=val_date,
            direction=direction,
            legs=legs,
        )

        if trade_type == "VanillaSwap":
            from models.interest_rate_swap import InterestRateSwap
            # Migrate legacy VanillaSwap records to InterestRateSwap on read
            return InterestRateSwap(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                swap_subtype=sub.get("swap_subtype") or "FIXED_FLOAT",
            )
        if trade_type == "InterestRateSwap":
            from models.interest_rate_swap import InterestRateSwap
            return InterestRateSwap(
                **common,
                tenor_y=int(sub.get("tenor_y", 0) or 0),
                swap_subtype=sub.get("swap_subtype", "FIXED_FLOAT") or "FIXED_FLOAT",
                fx_rate=float(sub.get("fx_rate", 1.0) or 1.0),
            )
        if trade_type == "CrossCurrencySwap":
            from models.cross_currency_swap import CrossCurrencySwap
            return CrossCurrencySwap(
                **common,
                tenor_y=int(sub.get("tenor_y", 0) or 0),
                pay_currency=sub.get("pay_currency", "USD") or "USD",
                receive_currency=sub.get("receive_currency", "EUR") or "EUR",
                fx_rate=float(sub.get("fx_rate", 1.0) or 1.0),
                initial_notional_exchange=bool(sub.get("initial_notional_exchange", 1)),
                final_notional_exchange=bool(sub.get("final_notional_exchange", 1)),
                swap_subtype=sub.get("swap_subtype", "FIXED_FLOAT") or "FIXED_FLOAT",
            )
        if trade_type == "Bond":
            from models.bond import Bond
            return Bond(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                isin=sub.get("isin"),
            )
        if trade_type == "CallableBond":
            from models.callable_bond import CallableBond
            return CallableBond(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                isin=sub.get("isin"),
            )
        if trade_type == "Option":
            from models.option_trade import OptionTrade
            return OptionTrade(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                underlying_tenor_y=int(sub.get("underlying_tenor_y") or 0),
            )
        if trade_type == "IRSwaption":
            from models.interest_rate_swaption import InterestRateSwaption
            return InterestRateSwaption(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                underlying_tenor_y=int(sub.get("underlying_tenor_y") or 0),
            )
        if trade_type == "EquitySwap":
            from models.equity_swap import EquitySwap
            return EquitySwap(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
                underlying_ticker=sub.get("underlying_ticker") or "",
            )
        if trade_type == "CDS":
            from models.cds import CreditDefaultSwap
            return CreditDefaultSwap(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
            )
        if trade_type == "EquityOption":
            from models.equity_option import EquityOptionTrade
            return EquityOptionTrade(
                **common,
                tenor_y=int(sub.get("tenor_y") or 0),
            )

        # Fallback: try generic fromJson path via old trades table
        old = self._conn.execute(
            "SELECT trade_json FROM trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()
        if old:
            from models.trade_base import TradeBase
            return TradeBase.fromJson(old[0])
        return None

    def list_all(self, trade_type: Optional[str] = None,
                 book: Optional[str] = None) -> list:
        """Return all trades, optionally filtered by trade_type and/or book."""
        sql = "SELECT trade_id FROM TradeBase WHERE 1=1"
        params: list = []
        if trade_type:
            sql += " AND trade_type = ?"
            params.append(trade_type)
        if book:
            sql += " AND book = ?"
            params.append(book)
        sql += " ORDER BY trade_id"
        rows = self._conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            t = self.get(row[0])
            if t is not None:
                result.append(t)
        return result

    def count(self, trade_type: Optional[str] = None) -> int:
        """Count stored trades."""
        if trade_type:
            return self._conn.execute(
                "SELECT COUNT(*) FROM TradeBase WHERE trade_type = ?", (trade_type,)
            ).fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM TradeBase").fetchone()[0]

    def exists(self, trade_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM TradeBase WHERE trade_id = ?", (trade_id,)
        ).fetchone() is not None

    # ── DELETE ────────────────────────────────────────────────────────────────

    def delete(self, trade_id: str) -> bool:
        """Delete a trade and cascade to subclass + leg tables."""
        with self._tx():
            cur = self._conn.execute(
                "DELETE FROM TradeBase WHERE trade_id = ?", (trade_id,)
            )
        return cur.rowcount > 0

    def delete_all(self) -> int:
        """Delete all trades. Returns count deleted."""
        with self._tx():
            n = self._conn.execute("SELECT COUNT(*) FROM TradeBase").fetchone()[0]
            self._conn.execute("DELETE FROM TradeBase")
        return n

    # ── PRICING RESULTS (new PricingResult table) ─────────────────────────────

    def save_results(self, results, run_id: str = "DEFAULT") -> int:
        """
        Persist a list of PricingResult objects to the PricingResult table.

        Parameters
        ----------
        results : List[PricingResult]
        run_id  : identifier for this pricing run  e.g. "RUN-1"

        Returns
        -------
        Number of rows inserted.
        """
        if not results:
            return 0

        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        def _f(v):
            import math
            if v is None:
                return None
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        rows = []
        for pr in results:
            rows.append((
                pr.trade_id, run_id, ts,
                getattr(pr, "instrument", None),
                getattr(pr, "book", None),
                getattr(pr, "counterparty", None),
                getattr(pr, "direction", None),
                getattr(pr, "swap_subtype", None),
                _f(getattr(pr, "tenor_y", None)),
                _f(getattr(pr, "notional", None)),
                _f(getattr(pr, "coupon_rate", None)),
                _f(getattr(pr, "npv", None)),
                _f(getattr(pr, "fixed_npv", None)),
                _f(getattr(pr, "float_npv", None)),
                _f(getattr(pr, "par_rate", None)),
                _f(getattr(pr, "clean_price", None)),
                _f(getattr(pr, "accrued", None)),
                _f(getattr(pr, "premium", None)),
                _f(getattr(pr, "dv01", None)),
                _f(getattr(pr, "duration", None)),
                _f(getattr(pr, "pv01", None)),
                _f(getattr(pr, "convexity", None)),
                _f(getattr(pr, "vega", None)),
                _f(getattr(pr, "theta", None)),
                _f(getattr(pr, "delta", None)),
                _f(getattr(pr, "gamma", None)),
                _f(getattr(pr, "rho", None)),
                _f(getattr(pr, "cr01", None)),
                _f(getattr(pr, "jump_to_default", None)),
                getattr(pr, "option_type", None) or None,
                getattr(pr, "exercise_type", None) or None,
                _f(getattr(pr, "strike", None)),
                _f(getattr(pr, "vol", None)),
                getattr(pr, "underlying_ticker", None) or None,
                getattr(pr, "valuation_date", None) or None,
                getattr(pr, "error", "") or "",
            ))

        sql = """
            INSERT INTO PricingResult (
                trade_id, run_id, run_timestamp,
                instrument, book, counterparty, direction, swap_subtype,
                tenor_y, notional, coupon_rate,
                npv, fixed_npv, float_npv,
                par_rate, clean_price, accrued, premium,
                dv01, duration, pv01, convexity,
                vega, theta, delta, gamma, rho,
                cr01, jump_to_default,
                option_type, exercise_type, strike, vol, underlying_ticker, valuation_date,
                error
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(trade_id, run_id) DO UPDATE SET
                run_timestamp   = excluded.run_timestamp,
                instrument      = excluded.instrument,
                book            = excluded.book,
                counterparty    = excluded.counterparty,
                direction       = excluded.direction,
                swap_subtype    = excluded.swap_subtype,
                tenor_y         = excluded.tenor_y,
                notional        = excluded.notional,
                coupon_rate     = excluded.coupon_rate,
                npv             = excluded.npv,
                fixed_npv       = excluded.fixed_npv,
                float_npv       = excluded.float_npv,
                par_rate        = excluded.par_rate,
                clean_price     = excluded.clean_price,
                accrued         = excluded.accrued,
                premium         = excluded.premium,
                dv01            = excluded.dv01,
                duration        = excluded.duration,
                pv01            = excluded.pv01,
                convexity       = excluded.convexity,
                vega            = excluded.vega,
                theta           = excluded.theta,
                delta           = excluded.delta,
                gamma           = excluded.gamma,
                rho             = excluded.rho,
                cr01            = excluded.cr01,
                jump_to_default = excluded.jump_to_default,
                option_type       = excluded.option_type,
                exercise_type     = excluded.exercise_type,
                strike            = excluded.strike,
                vol               = excluded.vol,
                underlying_ticker = excluded.underlying_ticker,
                valuation_date    = excluded.valuation_date,
                error           = excluded.error
        """
        with self._tx():
            self._conn.executemany(sql, rows)
        return len(rows)

    def clear_results(self) -> None:
        """Delete all rows from the PricingResult table."""
        with self._tx():
            self._conn.execute("DELETE FROM PricingResult")

    def get_results_df(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """
        SELECT * FROM PricingResult, optionally filtered by run_id.
        Returns a pandas DataFrame ordered by trade_id, run_id.
        """
        sql = "SELECT * FROM PricingResult"
        params: list = []
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params.append(run_id)
        sql += " ORDER BY trade_id, run_id"
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)

    # ── LEGACY pricing results (old pricing_results table) ────────────────────

    _RESULT_COLS = [
        "trade_id", "instrument", "book", "counterparty", "direction",
        "tenor_y", "notional", "coupon_rate",
        "swap_npv", "par_rate", "clean_price", "accrued",
        "fixed_npv", "float_npv",
        "dv01", "duration", "pv01", "convexity",
        "vega", "theta", "delta", "cr01", "jump_to_default",
        "error",
    ]

    def get_results(self, trade_id: Optional[str] = None,
                    run_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve legacy pricing results as a DataFrame."""
        sql = "SELECT * FROM pricing_results WHERE 1=1"
        params: list = []
        if trade_id:
            sql += " AND trade_id = ?"
            params.append(trade_id)
        if run_id:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY trade_id, run_timestamp"
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def get_latest_results(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """Return the most recent legacy pricing run's results."""
        if run_id:
            return self.get_results(run_id=run_id)
        latest = self._conn.execute(
            "SELECT run_id FROM pricing_results ORDER BY run_timestamp DESC LIMIT 1"
        ).fetchone()
        if latest is None:
            return pd.DataFrame()
        return self.get_results(run_id=latest[0])

    # ── PORTFOLIO QUERIES ─────────────────────────────────────────────────────

    def get_legs(self, trade_id: str) -> list:
        """Return leg dicts for a trade."""
        rows = self._conn.execute(_LEG_JOIN_SQL, (trade_id,)).fetchall()
        return [dict(r) for r in rows]

    def legs_summary(self) -> pd.DataFrame:
        """Return a summary of all stored legs."""
        cur = self._conn.execute(
            "SELECT trade_id, leg_index, leg_type FROM BaseLeg ORDER BY trade_id, leg_index"
        )
        return pd.DataFrame(cur.fetchall(), columns=["trade_id", "leg_index", "leg_type"])

    def trade_summary(self) -> pd.DataFrame:
        """Return a high-level summary of all stored trades."""
        cur = self._conn.execute(
            """
            SELECT trade_type, book, direction,
                   COUNT(*) as count,
                   GROUP_CONCAT(trade_id, ', ') as trade_ids
            FROM TradeBase
            GROUP BY trade_type, book, direction
            ORDER BY trade_type, book
            """
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    def __repr__(self) -> str:
        n = self.count()
        return f"TradeRepository(db={self.db_path!r}, trades={n})"
