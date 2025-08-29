from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
import os, json, time

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.middleware.wsgi import WSGIMiddleware

# -------------------------------
# robust model path resolution
# -------------------------------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "models",                # repo/models (often empty or placeholders)
    HERE.parent / "models",         # ../models  ← typical training output
    Path("/home/ubuntu/ccfraud/models"),
]

# allow override via env
env_override = os.getenv("MODELS_DIR")
if env_override:
    CANDIDATES.insert(0, Path(env_override))

MODELS_DIR = None
for d in CANDIDATES:
    pre = d / "preprocessor.joblib"
    mdl = d / "xgb_model.json"
    if pre.exists() and mdl.exists():
        MODELS_DIR = d
        break

if MODELS_DIR is None:
    raise FileNotFoundError(
        f"could not find model artifacts (preprocessor.joblib + xgb_model.json) "
        f"in any of: {', '.join(str(p) for p in CANDIDATES)}. "
        f"set MODELS_DIR env var if needed."
    )

PREPROC_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH   = MODELS_DIR / "xgb_model.json"
META_PATH    = MODELS_DIR / "metadata.json"

print(f"[startup] MODELS_DIR={MODELS_DIR}  (cwd={os.getcwd()})")
print(f"[startup] using PREPROC_PATH={PREPROC_PATH}")
print(f"[startup] using MODEL_PATH={MODEL_PATH}")
print(f"[startup] using META_PATH={META_PATH}")

# -------------------------------
# load artifacts
# -------------------------------
preproc = joblib.load(PREPROC_PATH)

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

# -------------------------------
# prometheus metrics
# -------------------------------
PREDICTIONS = Counter("predictions_total", "Total predictions", ["outcome"])
REQ_LATENCY = Histogram("request_latency_seconds", "Prediction request latency (s)")
ERRORS      = Counter("prediction_errors_total", "Prediction errors")
AUPRC       = Gauge("model_auprc", "Model AUPRC (from last training)")
ROC_AUC     = Gauge("model_roc_auc", "Model ROC-AUC (from last training)")

try:
    if META_PATH.exists():
        meta = json.load(open(META_PATH))
        if "validation" in meta:
            AUPRC.set(float(meta["validation"].get("auprc", 0)))
            ROC_AUC.set(float(meta["validation"].get("roc_auc", 0)))
except Exception as e:
    print(f"[startup] metadata load warning: {e}")

# configurable probability threshold
PROBA_THRESHOLD = float(os.getenv("PROBA_THRESHOLD", "0.5"))

# -------------------------------
# fastapi app
# -------------------------------
app = FastAPI(title="ccfraud-api", version="1.0.0")

class Record(BaseModel):
    # ensure names match your training features
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

class Batch(BaseModel):
    records: List[Record]

@app.get("/healthz")
def healthz():
    return {"status": "ok", "models_dir": str(MODELS_DIR)}

@app.get("/readyz")
def readyz():
    return {"status": "ready"}

@app.post("/predict")
def predict(batch: Batch) -> Dict[str, Any]:
    start = time.time()
    try:
        df = pd.DataFrame([r.dict() for r in batch.records])

        # use preprocessor if it provides transform; else assume numeric DF already aligned
        X = preproc.transform(df) if hasattr(preproc, "transform") else df.values

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= PROBA_THRESHOLD).astype(int)

        pos = int(preds.sum())
        neg = int(len(preds) - pos)
        if pos: PREDICTIONS.labels(outcome="fraud").inc(pos)
        if neg: PREDICTIONS.labels(outcome="legit").inc(neg)

        return {
            "n": len(preds),
            "threshold": PROBA_THRESHOLD,
            "probabilities": proba.tolist(),
            "predictions": preds.tolist()
        }
    except Exception as e:
        ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        REQ_LATENCY.observe(time.time() - start)

# expose /metrics (prometheus wsgi mounted under fastapi)
metrics_app = make_asgi_app()
app.mount("/metrics", WSGIMiddleware(metrics_app))

# optional: run directly with `python app.py` (you can still use `uvicorn app:app` too)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
