
import json
import logging
import os
import secrets
import time
from app.drift import assess_drift
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "churn_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

model = None
metadata = None
drift_reference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata, drift_reference

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model artifacts are missing. Run python src/train.py first."
        )

    reference_path = ROOT / "artifacts" / "drift_reference.json"

    if not reference_path.exists():
        raise RuntimeError(
            "Drift reference missing. Run python src/build_drift_reference.py first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    drift_reference = json.loads(reference_path.read_text())

    logger.info("Model loaded: %s", metadata["model_version"])
    yield


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0.0",
    description="API for predicting telecom customer churn.",
    lifespan=lifespan,
)


async def require_api_key(
    x_api_key: str | None = Header(default=None),
):
    expected = os.getenv("CHURN_API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API key authentication is not configured",
        )

    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        logger.info(json.dumps({
            "method": request.method,
            "path": request.url.path,
            "status_code": status,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }))


def predict_customer(customer: CustomerInput) -> PredictionResponse:
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    # Column names match the training data; customerID was excluded.
    features = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(metadata["decision_threshold"])

    return PredictionResponse(
        churn_probability=round(probability, 6),
        predicted_churn=probability >= threshold,
        threshold=threshold,
        model_version=metadata["model_version"],
    )


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"] if metadata else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerInput,
    _: None = Depends(require_api_key),
):
    return predict_customer(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    _: None = Depends(require_api_key),
):
    return BatchPredictionResponse(
        predictions=[predict_customer(c) for c in request.customers]
    )
@app.post("/drift")
def check_drift(
    request: BatchPredictionRequest,
    _: None = Depends(require_api_key),
):
    recent = pd.DataFrame([
        customer.model_dump()
        for customer in request.customers
    ])

    return assess_drift(drift_reference, recent)