"""FastAPI forecast service: /health, /predict (q50 + q90), /metrics (Prometheus text)."""
import os, time, lightgbm as lgb, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURES = ["lag_7", "lag_14", "lag_28", "roll7", "roll28", "dow", "month", "price", "promo", "is_bf"]
MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/model_v1")
app = FastAPI(title="ecomm-demand-forecast")
_m = {q: lgb.Booster(model_file=f"{MODEL_DIR}/q{q}.txt") for q in ("50", "90")}
_stats = {"requests": 0, "errors": 0, "latency_sum": 0.0}

class Row(BaseModel):
    lag_7: float = Field(ge=0); lag_14: float = Field(ge=0); lag_28: float = Field(ge=0)
    roll7: float = Field(ge=0); roll28: float = Field(ge=0)
    dow: int = Field(ge=0, le=6); month: int = Field(ge=1, le=12)
    price: float = Field(gt=0); promo: int = Field(ge=0, le=1); is_bf: int = Field(ge=0, le=1)

class Batch(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=1000)

@app.get("/health")
def health(): return {"status": "ok", "model_dir": MODEL_DIR}

@app.post("/predict")
def predict(b: Batch):
    t = time.perf_counter()
    try:
        X = np.array([[getattr(r, f) for f in FEATURES] for r in b.rows])
        p50, p90 = _m["50"].predict(X), np.maximum(_m["90"].predict(X), _m["50"].predict(X))
    except Exception as e:
        _stats["errors"] += 1; raise HTTPException(500, str(e))
    _stats["requests"] += 1; _stats["latency_sum"] += time.perf_counter() - t
    return {"q50": np.maximum(p50, 0).round(2).tolist(), "q90": np.maximum(p90, 0).round(2).tolist()}

@app.get("/metrics")
def metrics():
    s = _stats
    return (f"predict_requests_total {s['requests']}\npredict_errors_total {s['errors']}\n"
            f"predict_latency_seconds_sum {s['latency_sum']:.6f}\n")
