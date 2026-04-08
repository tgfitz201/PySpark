"""
db/market_data_repository.py
============================
SQLite CRUD for MarketDataSnapshot objects.

Schema
------
market_data_snapshots(
    valuation_date TEXT PRIMARY KEY,
    snapshot_json  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
)

Usage
-----
>>> from db import MarketDataRepository
>>> from models.market_data import make_default_snapshot
>>> repo = MarketDataRepository("trades.db")
>>> repo.upsert(make_default_snapshot())
>>> snap = repo.get(date(2025, 1, 15))
>>> repo.close()
"""

import sqlite3
from datetime import date, datetime
from typing import Optional, List
from models.market_data import MarketDataSnapshot


class MarketDataRepository:
    """SQLite repository for MarketDataSnapshot objects."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS market_data_snapshots (
                valuation_date TEXT PRIMARY KEY,
                snapshot_json  TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )
        """)
        self._con.commit()

    def upsert(self, snap: MarketDataSnapshot) -> None:
        now = datetime.utcnow().isoformat()
        self._con.execute("""
            INSERT INTO market_data_snapshots(valuation_date, snapshot_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(valuation_date) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_at    = excluded.updated_at
        """, (snap.valuation_date.isoformat(), snap.toJson(), now, now))
        self._con.commit()

    def get(self, valuation_date: date) -> Optional[MarketDataSnapshot]:
        row = self._con.execute(
            "SELECT snapshot_json FROM market_data_snapshots WHERE valuation_date = ?",
            (valuation_date.isoformat(),)
        ).fetchone()
        return MarketDataSnapshot.fromJson(row[0]) if row else None

    def list_dates(self) -> List[str]:
        rows = self._con.execute(
            "SELECT valuation_date FROM market_data_snapshots ORDER BY valuation_date DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def delete(self, valuation_date: date) -> None:
        self._con.execute(
            "DELETE FROM market_data_snapshots WHERE valuation_date = ?",
            (valuation_date.isoformat(),)
        )
        self._con.commit()

    def close(self):
        self._con.close()
