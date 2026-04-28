"""
Tests for FastAPI backend — run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_list_models():
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 4
    names = [m["name"] for m in models]
    assert "xgboost" in names


def test_predict_valid():
    payload = {
        "hour": 8,
        "day": 0,
        "is_weekend": 0,
        "is_peak_hour": 1,
        "model_name": "xgboost",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_speed" in data
    assert isinstance(data["predicted_speed"], float)
    assert "congestion_level" in data


def test_predict_invalid_hour():
    payload = {"hour": 25, "day": 0, "is_weekend": 0, "is_peak_hour": 0, "model_name": "xgboost"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422   # Validation error


def test_predict_realtime():
    r = client.get("/predict/realtime?model_name=xgboost")
    assert r.status_code == 200
    data = r.json()
    assert "predicted_speed" in data


def test_heatmap():
    r = client.get("/traffic/heatmap?model_name=xgboost")
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 24 * 7  # 168 cells


def test_weekly_pattern():
    r = client.get("/traffic/weekly-pattern?model_name=xgboost")
    assert r.status_code == 200
    wp = r.json()["weekly_pattern"]
    assert len(wp) == 7


def test_all_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) >= 1
    for m in models:
        assert "mae" in m
        assert "r2" in m
