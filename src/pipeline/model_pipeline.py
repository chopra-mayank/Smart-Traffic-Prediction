"""
Smart Traffic Prediction - Model Pipeline
Handles: preprocessing → feature engineering → multi-model training → save/load
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed" / "feature_data.csv"
DATA_PREDICTIONS = ROOT / "data" / "predictions"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PREDICTIONS.mkdir(parents=True, exist_ok=True)

# ─── Feature / Target Config ──────────────────────────────────────────────────
FEATURE_COLS = ["hour", "day", "is_weekend", "is_peak_hour"]
TARGET_COL = "traffic"

# ─── Model Definitions ────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    ),
    "xgboost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


# ─── Load & Prepare Data ──────────────────────────────────────────────────────
def load_feature_data(path: str = None) -> pd.DataFrame:
    """Load the feature-engineered CSV."""
    p = Path(path) if path else DATA_PROCESSED
    if not p.exists():
        raise FileNotFoundError(f"Feature data not found at: {p}")
    df = pd.read_csv(p)
    print(f"✅ Loaded feature data — shape: {df.shape}")
    return df


def prepare_xy(df: pd.DataFrame):
    """Return X (features) and y (target)."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ─── Build sklearn Pipeline ───────────────────────────────────────────────────
def build_pipeline(model_name: str = "xgboost") -> Pipeline:
    """
    Build a full sklearn Pipeline with StandardScaler + chosen estimator.
    The scaler is included so the pipeline is self-contained for inference.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    estimator = MODEL_REGISTRY[model_name]
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    return pipe


# ─── Train & Evaluate ─────────────────────────────────────────────────────────
def train_pipeline(
    model_name: str = "xgboost",
    test_size: float = 0.2,
    cross_validate: bool = True,
) -> dict:
    """
    Full training run:
      1. Load data
      2. Split train/test
      3. Build & fit pipeline
      4. Evaluate on test set
      5. (Optional) 5-fold CV on full data
      6. Save model artifact

    Returns a results dict with metrics + pipeline object.
    """
    df = load_feature_data()
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    pipe = build_pipeline(model_name)
    print(f"🚀 Training [{model_name}] …")
    pipe.fit(X_train, y_train)

    # ── Test-set metrics ──────────────────────────────────────────────────────
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R²   : {r2:.4f}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_scores = {}
    if cross_validate:
        print("   Running 5-fold CV …")
        cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
        cv_mae = -cross_val_score(
            pipe, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        cv_scores = {
            "cv_r2_mean": float(cv_r2.mean()),
            "cv_r2_std": float(cv_r2.std()),
            "cv_mae_mean": float(cv_mae.mean()),
            "cv_mae_std": float(cv_mae.std()),
        }
        print(
            f"   CV R² : {cv_scores['cv_r2_mean']:.4f} ± {cv_scores['cv_r2_std']:.4f}"
        )

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": preds,
            "hour": X_test["hour"].values,
            "day": X_test["day"].values,
            "is_weekend": X_test["is_weekend"].values,
            "is_peak_hour": X_test["is_peak_hour"].values,
        }
    )
    pred_path = DATA_PREDICTIONS / f"predictions_{model_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"✅ Predictions saved → {pred_path}")

    # ── Save pipeline ─────────────────────────────────────────────────────────
    model_path = save_pipeline(pipe, model_name)

    results = {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": str(model_path),
        "pipeline": pipe,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        **cv_scores,
    }
    return results


# ─── Save / Load ──────────────────────────────────────────────────────────────
def save_pipeline(pipe: Pipeline, model_name: str) -> Path:
    """Persist the fitted pipeline to disk."""
    path = MODELS_DIR / f"{model_name}_pipeline.pkl"
    joblib.dump(pipe, path)
    print(f"✅ Pipeline saved → {path}")
    return path


def load_pipeline(model_name: str = "xgboost") -> Pipeline:
    """Load a previously saved pipeline."""
    path = MODELS_DIR / f"{model_name}_pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved pipeline found for '{model_name}' at {path}. "
            "Run train_pipeline() first."
        )
    pipe = joblib.load(path)
    print(f"✅ Pipeline loaded ← {path}")
    return pipe


# ─── Inference Helper ─────────────────────────────────────────────────────────
def predict_traffic(
    hour: int,
    day: int,
    is_weekend: int,
    is_peak_hour: int,
    model_name: str = "xgboost",
) -> float:
    """
    Single-sample prediction using a saved pipeline.
    """
    pipe = load_pipeline(model_name)
    X = pd.DataFrame(
        [[hour, day, is_weekend, is_peak_hour]],
        columns=FEATURE_COLS,
    )
    pred = pipe.predict(X)[0]
    return float(pred)


# ─── Train All Models ─────────────────────────────────────────────────────────
def train_all_models() -> dict:
    """Train every registered model and return a comparison dict."""
    all_results = {}
    for name in MODEL_REGISTRY:
        print(f"\n{'='*55}")
        print(f"  Model: {name}")
        print(f"{'='*55}")
        res = train_pipeline(model_name=name, cross_validate=True)
        all_results[name] = {
            "mae": res["mae"],
            "rmse": res["rmse"],
            "r2": res["r2"],
            "cv_r2_mean": res.get("cv_r2_mean"),
            "cv_mae_mean": res.get("cv_mae_mean"),
        }
    return all_results


# ─── CLI Entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if model_arg == "all":
        results = train_all_models()
        print("\n\n📊 MODEL COMPARISON")
        print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
        print("-" * 57)
        for name, m in results.items():
            print(
                f"{name:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>8.4f}"
            )
    else:
        train_pipeline(model_name=model_arg)
