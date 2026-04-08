"""
REST service endpoint tests using FastAPI TestClient (no live server needed).
"""
import pytest, json

@pytest.fixture(scope="module")
def client():
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rest_service.app import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"REST service not available: {e}")

class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "trade_count" in data

class TestTradesCRUD:
    def test_list_trades(self, client):
        r = client.get("/trades")
        assert r.status_code == 200
        trades = r.json()
        assert isinstance(trades, list)
        assert len(trades) >= 1

    def test_get_trade(self, client):
        # get first trade id from list
        r = client.get("/trades")
        trades = r.json()
        tid = trades[0]["trade_id"]
        r2 = client.get(f"/trades/{tid}")
        assert r2.status_code == 200
        assert r2.json()["trade_id"] == tid

    def test_get_nonexistent_trade(self, client):
        r = client.get("/trades/NONEXISTENT-9999")
        assert r.status_code == 404

    def test_delete_nonexistent_trade(self, client):
        r = client.delete("/trades/NONEXISTENT-9999")
        assert r.status_code == 404
