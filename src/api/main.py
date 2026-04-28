"""
Smart Traffic Prediction — FastAPI Backend
Endpoints:
  GET  /                          → health check
  POST /predict                   → single prediction
  POST /predict/batch             → bulk predictions
  GET  /predict/realtime          → predict for current time
  GET  /models                    → list available models
  GET  /metrics/{model_name}      → return saved evaluation metrics
  GET  /traffic/heatmap           → hourly × day traffic heatmap data
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Make src importable ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # project root
sys.path.insert(0, str(ROOT / "src"))

from pipeline.model_pipeline import (
    load_pipeline,
    FEATURE_COLS,
    MODELS_DIR,
)

REPORTS_DIR = ROOT / "reports"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
FEATURE_DATA = ROOT / "data" / "processed" / "feature_data.csv"

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Traffic Prediction API",
    description=(
        "Predicts traffic speed using ML models trained on METR-LA sensor data. "
        "Features: hour, day_of_week, is_weekend, is_peak_hour."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Cache ──────────────────────────────────────────────────────────────
_model_cache: dict = {}

DEFAULT_MODEL = "xgboost"
AVAILABLE_MODELS = [
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]


def get_model(model_name: str):
    """Load model from cache or disk."""
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = load_pipeline(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Train it first via run_pipeline.py.",
            ) from exc
    return _model_cache[model_name]


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day: int = Field(..., ge=0, le=6, description="Day of week (0=Mon … 6=Sun)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend")
    is_peak_hour: int = Field(..., ge=0, le=1, description="1 if peak hour")
    model_name: str = Field(DEFAULT_MODEL, description="Model to use")

    model_config = {
        "json_schema_extra": {
            "example": {
                "hour": 8,
                "day": 0,
                "is_weekend": 0,
                "is_peak_hour": 1,
                "model_name": "xgboost",
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_speed: float
    model_used: str
    hour: int
    day: int
    is_weekend: bool
    is_peak_hour: bool
    congestion_level: str
    timestamp: str


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    count: int
    predictions: List[PredictResponse]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _congestion_label(speed: float) -> str:
    if speed >= 55:
        return "Free Flow"
    elif speed >= 40:
        return "Moderate"
    elif speed >= 25:
        return "Heavy"
    else:
        return "Severe"


def _build_response(req: PredictRequest, speed: float) -> PredictResponse:
    return PredictResponse(
        predicted_speed=round(speed, 4),
        model_used=req.model_name,
        hour=req.hour,
        day=req.day,
        is_weekend=bool(req.is_weekend),
        is_peak_hour=bool(req.is_peak_hour),
        congestion_level=_congestion_label(speed),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


def _derive_features_from_now() -> dict:
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    is_weekend = 1 if day >= 5 else 0
    is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    return dict(hour=hour, day=day, is_weekend=is_weekend, is_peak_hour=is_peak_hour)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Smart Traffic Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/models", tags=["Models"])
def list_models():
    """Return all models and which ones have a saved pipeline."""
    available = []
    for name in AVAILABLE_MODELS:
        path = MODELS_DIR / f"{name}_pipeline.pkl"
        available.append({"name": name, "trained": path.exists()})
    return {"models": available}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """Predict traffic speed for given time-based features."""
    model = get_model(req.model_name)
    X = pd.DataFrame(
        [[req.hour, req.day, req.is_weekend, req.is_peak_hour]],
        columns=FEATURE_COLS,
    )
    speed = float(model.predict(X)[0])
    return _build_response(req, speed)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(batch: BatchPredictRequest):
    """Predict for multiple records in one call."""
    responses = []
    for rec in batch.records:
        model = get_model(rec.model_name)
        X = pd.DataFrame(
            [[rec.hour, rec.day, rec.is_weekend, rec.is_peak_hour]],
            columns=FEATURE_COLS,
        )
        speed = float(model.predict(X)[0])
        responses.append(_build_response(rec, speed))
    return BatchPredictResponse(count=len(responses), predictions=responses)


@app.get("/predict/realtime", response_model=PredictResponse, tags=["Prediction"])
def predict_realtime(model_name: str = Query(DEFAULT_MODEL)):
    """Predict traffic for the current time (auto-derives features)."""
    feats = _derive_features_from_now()
    req = PredictRequest(**feats, model_name=model_name)
    model = get_model(model_name)
    X = pd.DataFrame([list(feats.values())], columns=FEATURE_COLS)
    speed = float(model.predict(X)[0])
    return _build_response(req, speed)


@app.get("/metrics/{model_name}", tags=["Evaluation"])
def get_metrics(model_name: str):
    """Return saved evaluation metrics for a trained model."""
    path = REPORTS_DIR / f"metrics_{model_name}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics for '{model_name}' not found. Run evaluate.py first.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/metrics", tags=["Evaluation"])
def get_all_metrics():
    """Return evaluation metrics for all trained models."""
    results = []
    for path in sorted(REPORTS_DIR.glob("metrics_*.json")):
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        raise HTTPException(status_code=404, detail="No metrics found yet.")
    return {"models": results}


@app.get("/traffic/heatmap", tags=["Analytics"])
def traffic_heatmap(model_name: str = Query(DEFAULT_MODEL)):
    """
    Returns a 24×7 matrix (hour × day) of predicted traffic speeds.
    Useful for rendering a heatmap on the frontend map.
    """
    model = get_model(model_name)
    rows = []
    for day in range(7):
        is_weekend = 1 if day >= 5 else 0
        for hour in range(24):
            is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            X = pd.DataFrame(
                [[hour, day, is_weekend, is_peak]],
                columns=FEATURE_COLS,
            )
            speed = float(model.predict(X)[0])
            rows.append(
                {
                    "hour": hour,
                    "day": day,
                    "speed": round(speed, 3),
                    "congestion": _congestion_label(speed),
                }
            )
    days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "model": model_name,
        "data": rows,
        "days": days_map,
    }


@app.get("/traffic/weekly-pattern", tags=["Analytics"])
def weekly_pattern(model_name: str = Query(DEFAULT_MODEL)):
    """Average predicted speed per day of week."""
    model = get_model(model_name)
    result = []
    days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in range(7):
        is_weekend = 1 if day >= 5 else 0
        speeds = []
        for hour in range(24):
            is_peak = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
            X = pd.DataFrame([[hour, day, is_weekend, is_peak]], columns=FEATURE_COLS)
            speeds.append(float(model.predict(X)[0]))
        result.append({"day": days_map[day], "avg_speed": round(np.mean(speeds), 3)})
    return {"model": model_name, "weekly_pattern": result}
