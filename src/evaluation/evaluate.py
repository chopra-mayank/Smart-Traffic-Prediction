"""
Smart Traffic Prediction - Evaluation Module
Generates comprehensive metrics + visualisation charts for all models.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ─── Core Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Return a dict of regression metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    metrics = {
        "model": model_name,
        "mae":   round(mae,  4),
        "mse":   round(mse,  4),
        "rmse":  round(rmse, 4),
        "r2":    round(r2,   4),
        "mape":  round(mape, 4),
    }
    return metrics


def print_metrics(metrics: dict):
    width = 45
    print(f"\n{'─'*width}")
    print(f"  📊  Evaluation — {metrics.get('model', 'Model')}")
    print(f"{'─'*width}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MSE  : {metrics['mse']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
    print(f"{'─'*width}\n")


def save_metrics_json(metrics: dict, model_name: str):
    path = REPORTS_DIR / f"metrics_{model_name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {path}")


# ─── Individual Model Plots ───────────────────────────────────────────────────
def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    sample: int = 500,
):
    """Scatter: actual vs predicted (sampled for readability)."""
    idx = np.random.choice(len(y_true), min(sample, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true[idx], y_pred[idx], alpha=0.4, s=18, color=PALETTE[0], label="Samples")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

    ax.set_xlabel("Actual Traffic Speed")
    ax.set_ylabel("Predicted Traffic Speed")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    fig.tight_layout()

    path = FIGURES_DIR / f"actual_vs_predicted_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📈  Saved: {path.name}")
    return str(path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """Residuals distribution and residuals vs predicted."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(residuals, bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Count")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.35, s=12, color=PALETTE[2])
    axes[1].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")

    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = FIGURES_DIR / f"residuals_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📈  Saved: {path.name}")
    return str(path)


def plot_time_series_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_points: int = 200,
):
    """Line chart of actual vs predicted over index (first n_points)."""
    n = min(n_points, len(y_true))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, y_true[:n], label="Actual",    color=PALETTE[0], lw=1.5)
    ax.plot(x, y_pred[:n], label="Predicted", color=PALETTE[1], lw=1.5, linestyle="--")
    ax.fill_between(x, y_true[:n], y_pred[:n], alpha=0.15, color=PALETTE[3])

    ax.set_title(f"Time-Series Prediction — {model_name}", fontsize=13)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Traffic Speed")
    ax.legend()
    fig.tight_layout()

    path = FIGURES_DIR / f"timeseries_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📈  Saved: {path.name}")
    return str(path)


def plot_feature_importance(pipeline, feature_names: list, model_name: str):
    """Bar chart of feature importances (tree-based models only)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"  ⚠️  {model_name} has no feature importances — skipping.")
        return None

    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"], color=PALETTE[0], edgecolor="white")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13)
    ax.set_xlabel("Importance")
    fig.tight_layout()

    path = FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📈  Saved: {path.name}")
    return str(path)


# ─── Comparison Plot ──────────────────────────────────────────────────────────
def plot_model_comparison(all_metrics: list):
    """
    Bar charts comparing all trained models on MAE, RMSE, R².
    all_metrics: list of dicts returned by compute_metrics().
    """
    if not all_metrics:
        return

    df = pd.DataFrame(all_metrics).set_index("model")
    metrics_to_plot = ["mae", "rmse", "r2"]
    titles = ["MAE (lower is better)", "RMSE (lower is better)", "R² (higher is better)"]
    colors = [PALETTE[1], PALETTE[3], PALETTE[0]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title, color in zip(axes, metrics_to_plot, titles, colors):
        bars = ax.bar(df.index, df[col], color=color, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col.upper())
        ax.tick_params(axis="x", rotation=20)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✅ Comparison chart saved → {path}")
    return str(path)


# ─── Full Evaluation Runner ───────────────────────────────────────────────────
def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    feature_names: list = None,
) -> dict:
    """
    End-to-end evaluation:
      • Compute metrics
      • Print them
      • Generate & save all plots
      • Save metrics JSON
    Returns metrics dict.
    """
    y_true = np.asarray(y_test)
    y_pred = pipeline.predict(X_test)

    metrics = compute_metrics(y_true, y_pred, model_name)
    print_metrics(metrics)
    save_metrics_json(metrics, model_name)

    plot_actual_vs_predicted(y_true, y_pred, model_name)
    plot_residuals(y_true, y_pred, model_name)
    plot_time_series_prediction(y_true, y_pred, model_name)

    if feature_names:
        plot_feature_importance(pipeline, feature_names, model_name)

    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick demo: loads saved predictions CSVs and re-evaluates + plots.
    """
    import sys
    from pathlib import Path

    PREDS_DIR = ROOT / "data" / "predictions"
    all_metrics = []

    for csv_path in sorted(PREDS_DIR.glob("predictions_*.csv")):
        model_name = csv_path.stem.replace("predictions_", "")
        df = pd.read_csv(csv_path)
        if "actual" not in df.columns or "predicted" not in df.columns:
            continue
        y_true = df["actual"].values
        y_pred = df["predicted"].values
        m = compute_metrics(y_true, y_pred, model_name)
        print_metrics(m)
        save_metrics_json(m, model_name)
        plot_actual_vs_predicted(y_true, y_pred, model_name)
        plot_residuals(y_true, y_pred, model_name)
        plot_time_series_prediction(y_true, y_pred, model_name)
        all_metrics.append(m)

    if all_metrics:
        plot_model_comparison(all_metrics)
        print("\n📊 Summary Table")
        print(pd.DataFrame(all_metrics).to_string(index=False))
