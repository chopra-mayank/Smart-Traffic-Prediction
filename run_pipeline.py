"""
run_pipeline.py — Master training + evaluation script
Usage:
    python run_pipeline.py           # train all models
    python run_pipeline.py xgboost   # train one model
"""

import sys
from pathlib import Path

# ── Make src importable ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline.model_pipeline import train_pipeline, train_all_models, FEATURE_COLS
from evaluation.evaluate import evaluate_model, plot_model_comparison


def run(model_name: str = "all"):
    all_metrics = []

    if model_name == "all":
        names = ["linear_regression", "random_forest", "gradient_boosting", "xgboost"]
    else:
        names = [model_name]

    for name in names:
        print(f"\n{'='*60}")
        print(f"  [TRAIN] {name}")
        print(f"{'='*60}")
        results = train_pipeline(model_name=name, cross_validate=True)

        metrics = evaluate_model(
            pipeline=results["pipeline"],
            X_test=results["X_test"],
            y_test=results["y_test"],
            model_name=name,
            feature_names=FEATURE_COLS,
        )
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        plot_model_comparison(all_metrics)
        print("\n\n📊 FINAL COMPARISON")
        import pandas as pd
        print(pd.DataFrame(all_metrics).to_string(index=False))

    print("\n✅ Pipeline complete! Models saved in /models, charts in /reports/figures")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    run(arg)
