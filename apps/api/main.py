from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


APP = FastAPI(title="Incident Ticket Triage API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3)
    system: Optional[str] = None
    source: Optional[str] = None
    error_code: Optional[str] = None


class PredictResponse(BaseModel):
    category: Optional[str] = None
    priority: Optional[str] = None
    routing_team: Optional[str] = None
    model: str
    note: str


def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Model not found at {p}. Train first.")
    return joblib.load(p)


# For MVP: load two separate models if you trained them
PRIORITY_MODEL_PATH = "artifacts/baseline_priority/model.joblib"
CATEGORY_MODEL_PATH = "artifacts/baseline_category/model.joblib"

_priority_model = None
_category_model = None


@APP.get("/health")
def health():
    return {"status": "ok"}


@APP.on_event("startup")
def startup():
    global _priority_model, _category_model
    # Load if present; allow running even if one missing.
    try:
        _priority_model = _load_model(PRIORITY_MODEL_PATH)
    except Exception:
        _priority_model = None
    try:
        _category_model = _load_model(CATEGORY_MODEL_PATH)
    except Exception:
        _category_model = None


@APP.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = pd.DataFrame([{"text": req.text}])

    priority = None
    category = None

    if _priority_model is not None:
        priority = str(_priority_model.predict(X)[0])

    if _category_model is not None:
        category = str(_category_model.predict(X)[0])

    # MVP routing: simple mapping by category (fallback)
    routing_map = {
        "outage": "SRE",
        "latency": "SRE",
        "auth_issue": "Security",
        "payment_issue": "Payments",
        "data_issue": "Data",
        "deployment_issue": "Platform",
    }
    routing_team = routing_map.get(category, "Triage")

    note = "Train models to enable predictions. Missing model files will result in null outputs."
    return PredictResponse(
        category=category,
        priority=priority,
        routing_team=routing_team,
        model="tfidf+logreg",
        note=note,
    )
