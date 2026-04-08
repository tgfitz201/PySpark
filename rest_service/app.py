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

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure examples/ is on path so all relative imports work
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from db.repository import TradeRepository
from models.trade_base import TradeBase

# DB path: resolve relative to this file's parent directory
_DB_PATH = str(Path(__file__).resolve().parent.parent / "trades.db")

app = FastAPI(title="Trade CRUD REST Service", version="1.0.0")


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
    repo = _get_repo()
    try:
        trade = repo.get(trade_id)
    finally:
        repo.close()
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
    return _trade_to_dict(trade)


@app.put("/trades/{trade_id}")
def replace_trade(trade_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body["trade_id"] = trade_id  # enforce trade_id from URL
    try:
        trade = TradeBase.fromJson(json.dumps(body))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid trade data: {e}")
    repo = _get_repo()
    try:
        repo.upsert(trade)
    finally:
        repo.close()
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
    return {"deleted": trade_id}


# ── Pricing ───────────────────────────────────────────────────────────────────

@app.get("/trades/{trade_id}/price")
def price_trade(trade_id: str) -> Dict[str, Any]:
    repo = _get_repo()
    try:
        trade = repo.get(trade_id)
    finally:
        repo.close()
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' not found")

    try:
        # Import pricing function from manage_swaps
        from manage_swaps import _price_one, make_curve_df
        curve_df = make_curve_df()
        result = _price_one(trade, curve_df)
        return _nan_safe(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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
