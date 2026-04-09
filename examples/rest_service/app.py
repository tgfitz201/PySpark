"""
rest_service/app.py
===================
FastAPI Trade CRUD REST Service.

Endpoints:
  GET    /health                     → {"status": "ok", "trade_count": N}
  GET    /trades                     → list all trades (JSON array)
  GET    /trades/{trade_id}          → single trade JSON
  POST   /trades                     → create trade from JSON body
  PUT    /trades/{trade_id}          → full replace from JSON body
  DELETE /trades/{trade_id}          → delete, 404 if not found
  GET    /trades/{trade_id}/price    → price the trade, return PricingResult fields
  GET    /results                    → list all PricingResult rows (run_id optional query param)
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure examples/ directory is on path for relative imports
_THIS_DIR   = Path(__file__).resolve().parent          # examples/rest_service/
_EXAMPLES   = _THIS_DIR.parent                          # examples/
_PROJECT    = _EXAMPLES.parent                          # PySpark/
for _p in [str(_EXAMPLES), str(_PROJECT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from db.repository import TradeRepository
from models.trade_base import TradeBase

# DB path
_DB_PATH = str(_PROJECT / "trades.db")

app = FastAPI(title="Trade CRUD REST Service", version="1.0.0")

# ── Caches ────────────────────────────────────────────────────────────────────
# Curve: rebuilt once per process, shared across all concurrent pricing requests.
# TTL: 5 minutes — refreshes automatically if market data changes.
_CURVE_TTL = 300   # seconds
_curve_lock = threading.Lock()
_curve_cache: Dict[str, Any] = {"df": None, "ts": 0.0}


def _get_curve_df():
    """Return cached curve DataFrame, rebuilding if older than TTL."""
    now = time.monotonic()
    if _curve_cache["df"] is None or (now - _curve_cache["ts"]) > _CURVE_TTL:
        with _curve_lock:
            if _curve_cache["df"] is None or (now - _curve_cache["ts"]) > _CURVE_TTL:
                from manage_trades import make_curve_df
                _curve_cache["df"] = make_curve_df()
                _curve_cache["ts"] = time.monotonic()
    return _curve_cache["df"]


def _invalidate_curve():
    """Force curve rebuild on next request (call after market data update)."""
    with _curve_lock:
        _curve_cache["ts"] = 0.0


# Trade cache: keyed by trade_id, max 5000 entries (full portfolio).
# Invalidated individually on POST/PUT/DELETE.
@functools.lru_cache(maxsize=5000)
def _cached_trade(trade_id: str):
    repo = _get_repo()
    try:
        return repo.get(trade_id)
    finally:
        repo.close()


def _invalidate_trade(trade_id: str):
    """Drop a single trade from the LRU cache after a write."""
    # lru_cache has no per-key eviction; clear the whole cache — it's cheap to
    # rebuild (SQLite reads are fast) and correctness beats micro-optimisation.
    _cached_trade.cache_clear()


def _get_repo() -> TradeRepository:
    return TradeRepository(_DB_PATH)


def _nan_safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


def _trade_to_dict(trade: TradeBase) -> Dict[str, Any]:
    return _nan_safe(trade._to_enriched_dict())


# ── Cache management ──────────────────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    ci = _cached_trade.cache_info()
    age = time.monotonic() - _curve_cache["ts"] if _curve_cache["df"] is not None else None
    return {
        "trade_cache": {"hits": ci.hits, "misses": ci.misses,
                        "size": ci.currsize, "maxsize": ci.maxsize},
        "curve_cache": {"loaded": _curve_cache["df"] is not None,
                        "age_seconds": round(age, 1) if age is not None else None,
                        "ttl_seconds": _CURVE_TTL},
    }


@app.post("/cache/clear")
def cache_clear() -> Dict[str, Any]:
    _cached_trade.cache_clear()
    _invalidate_curve()
    return {"cleared": True}


# ── Startup: warm curve + trade cache ─────────────────────────────────────────

@app.on_event("startup")
def _warm_cache():
    """Pre-build the yield curve on startup so the first pricing request is fast."""
    _get_curve_df()   # builds and caches the curve; trade cache warms on first access


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    repo = _get_repo()
    try:
        count = repo.count()
    finally:
        repo.close()
    return {"status": "ok", "trade_count": count}


# ── Trades CRUD ───────────────────────────────────────────────────────────────

@app.get("/trades")
def list_trades() -> List[Dict[str, Any]]:
    repo = _get_repo()
    try:
        trades = repo.list_all()
    finally:
        repo.close()
    return [_trade_to_dict(t) for t in trades]


@app.get("/trades/{trade_id}")
def get_trade(trade_id: str) -> Dict[str, Any]:
    trade = _cached_trade(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    return _trade_to_dict(trade)


@app.post("/trades", status_code=201)
def create_trade(body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        trade = TradeBase.fromJson(json.dumps(body))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid trade data: {e}")
    repo = _get_repo()
    try:
        repo.upsert(trade)
    finally:
        repo.close()
    _invalidate_trade(trade.trade_id)
    return _trade_to_dict(trade)


@app.put("/trades/{trade_id}")
def replace_trade(trade_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body["trade_id"] = trade_id
    try:
        trade = TradeBase.fromJson(json.dumps(body))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid trade data: {e}")
    repo = _get_repo()
    try:
        repo.upsert(trade)
    finally:
        repo.close()
    _invalidate_trade(trade_id)
    return _trade_to_dict(trade)


@app.delete("/trades/{trade_id}", status_code=200)
def delete_trade(trade_id: str) -> Dict[str, Any]:
    repo = _get_repo()
    try:
        deleted = repo.delete(trade_id)
    finally:
        repo.close()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    _invalidate_trade(trade_id)
    return {"deleted": trade_id}


# ── Pricing ───────────────────────────────────────────────────────────────────

@app.get("/trades/{trade_id}/price")
def price_trade(trade_id: str) -> Dict[str, Any]:
    trade = _cached_trade(trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")
    try:
        from manage_trades import _price_one
        result = _price_one(trade, _get_curve_df())
        return _nan_safe(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Portfolio Tree ────────────────────────────────────────────────────────────

@app.get("/tree")
def trade_tree() -> Dict[str, Any]:
    """Return full portfolio tree: Root -> Trader -> Book -> Trade -> Legs."""
    repo = _get_repo()
    try:
        trades = repo.list_all()
    finally:
        repo.close()

    from collections import defaultdict
    tree = defaultdict(lambda: defaultdict(list))
    for t in trades:
        trader = getattr(t, "trader", "") or "(no trader)"
        book   = getattr(t, "book",   "") or "(no book)"
        tree[trader][book].append(t)

    def leg_node(leg) -> Dict[str, Any]:
        lt   = getattr(leg, "leg_type", type(leg).__name__)
        notl = getattr(leg, "notional", 0)
        return {
            "type": "leg",
            "name": f"{lt}  ${notl:,.0f}",
            "leg_type": lt,
            "notional": notl,
        }

    def trade_node(t) -> Dict[str, Any]:
        legs = getattr(t, "legs", [])
        return {
            "type": "trade",
            "name": t.trade_id,
            "trade_id": t.trade_id,
            "instrument": type(t).__name__,
            "direction": str(t.direction.value),
            "notional": getattr(legs[0], "notional", 0) if legs else 0,
            "children": [leg_node(l) for l in legs],
        }

    def book_node(book_name: str, book_trades: list) -> Dict[str, Any]:
        return {
            "type": "book",
            "name": book_name,
            "trade_count": len(book_trades),
            "children": [trade_node(t) for t in sorted(book_trades, key=lambda x: x.trade_id)],
        }

    def trader_node(trader_name: str, books: dict) -> Dict[str, Any]:
        book_nodes = [book_node(b, books[b]) for b in sorted(books)]
        total = sum(len(v) for v in books.values())
        return {
            "type": "trader",
            "name": trader_name,
            "trade_count": total,
            "children": book_nodes,
        }

    trader_nodes = [trader_node(tr, tree[tr]) for tr in sorted(tree)]
    total_trades = sum(n["trade_count"] for n in trader_nodes)

    return {
        "type": "root",
        "name": "Portfolio",
        "trade_count": total_trades,
        "children": trader_nodes,
    }


# ── Results ───────────────────────────────────────────────────────────────────

@app.get("/results")
def list_results(run_id: Optional[str] = Query(default=None)) -> List[Dict[str, Any]]:
    repo = _get_repo()
    try:
        df = repo.get_results_df(run_id=run_id)
    finally:
        repo.close()
    records = df.where(df.notna(), other=None).to_dict("records")
    return _nan_safe(records)
