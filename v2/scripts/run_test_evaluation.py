"""
V2 Pipeline – Phase 2: evaluate chosen models on the held-out test set.

Prerequisites: run run_tuning.py first to generate results/<model>_tuned.json.
You must explicitly name the models you want to commit to testing.

Run:
    conda run -n hull_tactical python v2/scripts/run_test_evaluation.py --models xgboost ridge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import mlflow

from pipeline.backtest import competition_score, compute_metrics
from pipeline.config import K_ROLL_WINDOW, RESULTS_DIR, TARGET_COL
from pipeline.diagnostics import plot_test_results

CV_DIR   = RESULTS_DIR / "cv"
TEST_DIR = RESULTS_DIR / "test"
TEST_DIR.mkdir(exist_ok=True)
from pipeline.feature_engineering import build_features
from pipeline.models import apply_scaler, get_model, map_positions
from pipeline.preprocessing import drop_high_na_cols, load_data, preprocess, train_test_split
from pipeline.walk_forward import train_final_model

MLFLOW_DB = Path(__file__).parents[1] / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

ALL_MODELS = ["linear", "ridge", "lasso", "elasticnet", "rf", "xgboost", "lightgbm", "lstm"]


def load_tuned(model_name: str) -> dict:
    path = CV_DIR / f"{model_name}_tuned.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No tuning results for '{model_name}'. Run run_tuning.py first.\n  Expected: {path}"
        )
    with open(path) as f:
        return json.load(f)


def evaluate_model(model_name: str, train_df, test_df, raw_to_eng: dict) -> dict:
    tuned = load_tuned(model_name)
    best_params = tuned["best_params"]
    best_k = tuned["k"]

    print(f"\n{'='*60}")
    print(f"  Model: {model_name.upper()}")
    print(f"  Params: k={best_k:.3f}  {best_params}")
    print(f"{'='*60}")
    t0 = time.time()

    factory = lambda: get_model(model_name, **best_params)
    final_model, eng_features, scaler, col_means = train_final_model(
        train_df, raw_to_eng, factory
    )

    X_test = test_df[eng_features].fillna(col_means)
    if scaler is not None:
        X_test = apply_scaler(scaler, X_test)
    pred_test = final_model.predict(X_test)

    # Warm-start rolling_std with last K_ROLL_WINDOW training predictions so
    # the test-set position mapping is consistent with production inference.
    X_ctx = train_df[eng_features].fillna(col_means).iloc[-K_ROLL_WINDOW:]
    if scaler is not None:
        X_ctx = apply_scaler(scaler, X_ctx)
    pred_ctx = final_model.predict(X_ctx)
    positions_full = map_positions(np.concatenate([pred_ctx, pred_test]), k=best_k)
    positions_test = positions_full[len(pred_ctx):]

    fwd_test = test_df["forward_returns"]
    rf_test  = test_df["risk_free_rate"]
    y_test   = test_df[TARGET_COL]

    bh_score = competition_score(fwd_test, rf_test, np.ones(len(test_df)))
    test_metrics = compute_metrics(
        fwd_test, rf_test, positions_test,
        pred=pred_test, target=y_test.values,
    )
    elapsed = time.time() - t0

    print(f"\n  Buy-and-hold benchmark:   {bh_score:.4f}")
    print(f"  Adjusted Sharpe (test):   {test_metrics['adjusted_sharpe']:.4f}")
    print(f"  Raw Sharpe:               {test_metrics['sharpe_raw']:.4f}")
    print(f"  Max drawdown:             {test_metrics['max_drawdown']:.4f}")
    print(f"  Win rate:                 {test_metrics['win_rate']:.4f}")
    print(f"  Runtime: {elapsed:.1f}s")

    plot_paths = plot_test_results(
        fwd_test, rf_test, positions_test, pred_test, y_test.values,
        model_name, save_dir=TEST_DIR,
    )
    for p in plot_paths:
        print(f"  Plot   → {p}")

    mlflow.set_experiment("v2_test_eval")
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("phase", "test_eval")
        mlflow.log_params({**best_params, "k": best_k})
        mlflow.log_metrics({
            "test_adjusted_sharpe":      test_metrics["adjusted_sharpe"],
            "test_sharpe_raw":           test_metrics["sharpe_raw"],
            "test_max_drawdown":         test_metrics["max_drawdown"],
            "test_win_rate":             test_metrics["win_rate"],
            "test_ann_vol_pct":          test_metrics["ann_vol_pct"],
            "benchmark_adjusted_sharpe": bh_score,
            "n_features_used":           len(eng_features),
            "runtime_sec":               elapsed,
        })
        for key in ("ic", "icir"):
            val = test_metrics.get(key)
            if val is not None and np.isfinite(float(val)):
                mlflow.log_metric(f"test_{key}", float(val))
        for p in plot_paths:
            mlflow.log_artifact(str(p))

    result = {
        "model":                model_name,
        "best_params":          {**best_params, "k": best_k},
        "cv_10fold_summary":    tuned["cv_10fold_summary"],
        "test_metrics": {
            k: (float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else None)
            for k, v in test_metrics.items()
            if isinstance(v, (int, float))
        },
        "benchmark_adjusted_sharpe": bh_score,
        "n_features_used":     len(eng_features),
        "runtime_sec":         elapsed,
    }
    out_path = TEST_DIR / f"{model_name}_test_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved  → {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="V2 Phase 2: evaluate chosen models on the held-out test set."
    )
    parser.add_argument(
        "--models", nargs="+", required=True, choices=ALL_MODELS,
        help="Models to evaluate. Must have run_tuning.py first for each.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("V2 Pipeline – Phase 2: Test Set Evaluation")
    print(f"Models: {args.models}")
    print("=" * 60)

    raw = load_data()
    cleaned = preprocess(raw)
    feat_df, raw_to_eng = build_features(cleaned)
    train_df, test_df = train_test_split(feat_df)
    train_df, test_df, _ = drop_high_na_cols(train_df, test_df, raw_to_eng)
    print(f"\n[Data ready] Train: {train_df.shape}, Test: {test_df.shape}")

    bh_score = competition_score(
        test_df["forward_returns"], test_df["risk_free_rate"], np.ones(len(test_df))
    )
    print(f"  Buy-and-hold test benchmark: {bh_score:.4f}")

    all_results = []
    for model_name in args.models:
        result = evaluate_model(model_name, train_df, test_df, raw_to_eng)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("TEST SET SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<12} {'CV-10fold':>10} {'Test AdjShr':>12} {'Beat BH':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        cv10 = r["cv_10fold_summary"]["adjusted_sharpe"]
        test_as = r["test_metrics"].get("adjusted_sharpe") or 0.0
        beat = "YES" if test_as > bh_score else "no"
        print(f"{r['model']:<12} {cv10:>10.4f} {test_as:>12.4f} {beat:>8}")

    print(f"\nMLflow UI: cd v2/ && mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
