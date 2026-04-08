# Trade CRUD REST Service

FastAPI-based REST service for trade management and pricing.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with trade count |
| GET | `/trades` | List all trades |
| GET | `/trades/{trade_id}` | Get single trade |
| POST | `/trades` | Create trade from JSON body |
| PUT | `/trades/{trade_id}` | Replace trade from JSON body |
| DELETE | `/trades/{trade_id}` | Delete trade (404 if not found) |
| GET | `/trades/{trade_id}/price` | Price a trade, return PricingResult fields |
| GET | `/results` | List all PricingResult rows (optional `?run_id=RUN-1`) |

## Running

```bash
cd /Users/Fitzgerald/PycharmProjects/PySpark/examples
uvicorn rest_service.app:app --reload
```

Or from project root:
```bash
cd /Users/Fitzgerald/PycharmProjects/PySpark
uvicorn rest_service.app:app --reload --app-dir examples
```

## Notes

- DB path resolves to `../trades.db` relative to `rest_service/`
- Trade I/O uses `_to_enriched_dict()` for output, `TradeBase.fromJson()` for input
- Pricing delegates to `run_pricing()` from `manage_swaps.py`
