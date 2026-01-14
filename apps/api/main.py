from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/baseline_priority/model.joblib")

app = FastAPI(title="Incident Ticket Triage API")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    target: str
    label: str
    proba: dict[str, float] | None = None


_model = None


@app.on_event("startup")
def load_model() -> None:
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}. Run train_baseline first.")
    _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = pd.DataFrame({"text": [req.text]})
    label = str(_model.predict(X)[0])

    proba = None
    if hasattr(_model, "predict_proba") and hasattr(_model, "classes_"):
        probs = _model.predict_proba(X)[0]
        proba = {str(c): float(p) for c, p in zip(_model.classes_, probs)}

    return PredictResponse(target="priority", label=label, proba=proba)
