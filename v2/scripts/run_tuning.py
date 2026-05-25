"""
V2 Pipeline – Phase 1: tune all models with Optuna, run 10-fold CV, stop before test set.

After reviewing CV results (results/*_tuned.json, learning curves, MLflow), run
run_test_evaluation.py with the model(s) you want to commit to testing.

Run all models:
    conda run -n hull_tactical python v2/scripts/run_tuning.py

Run specific models with more trials:
    conda run -n hull_tactical python v2/scripts/run_tuning.py --models ridge xgboost --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import optuna
import mlflow

from pipeline.backtest import compute_metrics
from pipeline.config import K_DEFAULT, N_FOLDS_EVAL, N_FOLDS_TUNE, RESULTS_DIR
from pipeline.diagnostics import plot_learning_curves, plot_residuals

CV_DIR = RESULTS_DIR / "cv"
CV_DIR.mkdir(exist_ok=True)
from pipeline.feature_engineering import build_features
from pipeline.models import get_model, suggest_params
from pipeline.preprocessing import drop_high_na_cols, load_data, preprocess, train_test_split
from pipeline.walk_forward import run_walk_forward_cv, walk_forward_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

MLFLOW_DB = Path(__file__).parents[1] / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

ALL_MODELS = ["linear", "ridge", "lasso", "elasticnet", "rf", "xgboost", "lightgbm", "lstm"]


def make_objective(train_df, raw_to_eng, model_name: str):
    def objective(trial) -> float:
        params = suggest_params(trial, model_name)
        k = trial.suggest_float("k", 0.5, 3.0)
        factory = lambda: get_model(model_name, **params)
        try:
            cv = run_walk_forward_cv(
                train_df, raw_to_eng, factory,
                n_folds=N_FOLDS_TUNE, k=k, verbose=False,
            )
            score = cv["summary"]["adjusted_sharpe"]
            return float(score) if np.isfinite(score) else -1e6
        except Exception:
            return -1e6
    return objective


def tune_model(model_name: str, train_df, raw_to_eng: dict, n_trials: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Model: {model_name.upper()}")
    print(f"{'='*60}")
    t0 = time.time()

    sampler = optuna.samplers.TPESampler(seed=abs(hash(model_name)) % 100_000)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_trials_actual = max(n_trials, 1)
    print(f"  Tuning ({n_trials_actual} trials, {N_FOLDS_TUNE}-fold CV)...")
    study.optimize(
        make_objective(train_df, raw_to_eng, model_name),
        n_trials=n_trials_actual,
        show_progress_bar=False,
    )

    best_params = dict(study.best_params)
    best_k = best_params.pop("k", K_DEFAULT)
    best_score_cv = study.best_value
    print(f"  Best CV adjusted_sharpe: {best_score_cv:.4f}")
    print(f"  Best params: k={best_k:.3f}  {best_params}")

    print(f"  Running {N_FOLDS_EVAL}-fold final evaluation...")
    factory = lambda: get_model(model_name, **best_params)
    cv_final = run_walk_forward_cv(
        train_df, raw_to_eng, factory,
        n_folds=N_FOLDS_EVAL, k=best_k, verbose=True,
        compute_train_score=True, collect_predictions=True,
    )

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.1f}s")

    ext_metrics: dict = {}
    if "all_fwd" in cv_final:
        _fwd = pd.Series(cv_final["all_fwd"])
        _rf = pd.Series(cv_final["all_rf"])
        _pos = cv_final["all_positions"]
        ext_metrics = compute_metrics(
            _fwd, _rf, _pos,
            pred=cv_final["all_preds"], target=cv_final["all_actuals"],
        )
        strat_ret = _rf.values * (1 - _pos) + _pos * _fwd.values
        ext_metrics["cum_return_pct"] = float((np.prod(1 + strat_ret) - 1) * 100)

    fold_csv = CV_DIR / f"{model_name}_folds.csv"
    cv_final["results_df"].to_csv(fold_csv, index=False)
    lc_path = plot_learning_curves(cv_final["results_df"], model_name, save_dir=CV_DIR)
    resid_path = None
    if "all_preds" in cv_final:
        resid_path = plot_residuals(
            cv_final["all_preds"], cv_final["all_actuals"], model_name, save_dir=CV_DIR
        )

    n_features_avg = int(cv_final["results_df"]["n_features"].mean().round())
    mlflow.set_experiment("v2_tuning")
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("phase", "tuning")
        mlflow.log_params({**best_params, "k": best_k, "n_trials": n_trials_actual})
        mlflow.log_metrics({
            "cv_adjusted_sharpe_5fold":  best_score_cv,
            "cv_adjusted_sharpe_10fold": cv_final["summary"]["adjusted_sharpe"],
            "cv_mae":          cv_final["summary"]["mae"],
            "cv_r2":           cv_final["summary"]["r2"],
            "n_features_avg":  n_features_avg,
            "runtime_sec":     elapsed,
        })
        mlflow.log_artifact(str(fold_csv))
        mlflow.log_artifact(str(lc_path))
        if resid_path is not None:
            mlflow.log_artifact(str(resid_path))

    def _safe_float(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    result = {
        "model":                   model_name,
        "best_params":             best_params,
        "k":                       best_k,
        "cv_5fold_adjusted_sharpe": best_score_cv,
        "cv_10fold_summary":       cv_final["summary"],
        "ext_metrics":             {k: _safe_float(v) for k, v in ext_metrics.items() if isinstance(v, (int, float))},
        "n_trials":                n_trials_actual,
        "runtime_sec":             elapsed,
    }
    out_path = CV_DIR / f"{model_name}_tuned.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")
    return result


def _save_cv_summary(all_results: list[dict], bh_metrics: dict, save_dir: Path) -> Path:
    """Build results/cv_summary.csv comparing all tuned models vs buy-and-hold."""
    metric_keys = [
        "adjusted_sharpe", "sharpe_raw", "sortino", "calmar",
        "max_drawdown", "win_rate", "ann_vol_pct", "cum_return_pct", "ic", "icir",
    ]
    rows = []

    bh_row: dict = {
        "model": "buy_and_hold",
        "cv5_adj_sharpe": None,
        "k": None,
        "n_trials": None,
        "runtime_sec": None,
    }
    for key in metric_keys:
        bh_row[f"cv10_{key}"] = bh_metrics.get(key)
    bh_row["cv10_adj_sharpe_fold_avg"] = None
    bh_row["cv10_mae"] = None
    bh_row["cv10_rmse"] = None
    bh_row["cv10_r2"] = None
    rows.append(bh_row)

    for r in all_results:
        ext = r.get("ext_metrics", {})
        cv10 = r["cv_10fold_summary"]
        row: dict = {
            "model": r["model"],
            "cv5_adj_sharpe": r.get("cv_5fold_adjusted_sharpe"),
            "k": r["k"],
            "n_trials": r["n_trials"],
            "runtime_sec": r["runtime_sec"],
        }
        for key in metric_keys:
            row[f"cv10_{key}"] = ext.get(key)
        row["cv10_adj_sharpe_fold_avg"] = cv10.get("adjusted_sharpe")
        row["cv10_mae"] = cv10.get("mae")
        row["cv10_rmse"] = cv10.get("rmse")
        row["cv10_r2"] = cv10.get("r2")
        rows.append(row)

    # Column order: identifiers, then metrics grouped logically
    cols = [
        "model", "cv5_adj_sharpe", "k", "n_trials", "runtime_sec",
        "cv10_adjusted_sharpe", "cv10_adj_sharpe_fold_avg",
        "cv10_sharpe_raw", "cv10_sortino", "cv10_calmar",
        "cv10_max_drawdown", "cv10_win_rate", "cv10_ann_vol_pct", "cv10_cum_return_pct",
        "cv10_ic", "cv10_icir",
        "cv10_mae", "cv10_rmse", "cv10_r2",
    ]
    out_path = save_dir / "cv_summary.csv"
    pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="V2 Phase 1: tune models and run 10-fold CV. Stops before test set."
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
                        metavar="MODEL", help=f"Models to tune (default: all). Choices: {ALL_MODELS}")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    args = parser.parse_args()

    print("=" * 60)
    print("V2 Pipeline – Phase 1: Tuning")
    print(f"Models: {args.models}  |  Trials per model: {args.n_trials}")
    print("=" * 60)

    raw = load_data()
    cleaned = preprocess(raw)
    feat_df, raw_to_eng = build_features(cleaned)
    train_df, test_df = train_test_split(feat_df)
    train_df, test_df, dropped = drop_high_na_cols(train_df, test_df, raw_to_eng)
    print(f"\n[Data ready] Train: {train_df.shape}, Test (held-out, not touched): {test_df.shape}")
    print(f"  Dropped high-NA cols: {len(dropped)}")

    all_results = []
    for model_name in args.models:
        result = tune_model(model_name, train_df, raw_to_eng, args.n_trials)
        all_results.append(result)

    # Merge with any previously tuned models not in this run
    run_models = {r["model"] for r in all_results}
    for model_name in ALL_MODELS:
        if model_name in run_models:
            continue
        cached = CV_DIR / f"{model_name}_tuned.json"
        if cached.exists():
            with open(cached) as f:
                all_results.append(json.load(f))

    print(f"\n{'='*60}")
    print("TUNING SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<12} {'CV-5fold':>10} {'CV-10fold':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['model']:<12} {r['cv_5fold_adjusted_sharpe']:>10.4f} {r['cv_10fold_summary']['adjusted_sharpe']:>10.4f}")

    # Compute buy-and-hold metrics on the same 10-fold validation windows
    bh_splits = walk_forward_splits(len(train_df), n_folds=N_FOLDS_EVAL)
    bh_fwd = np.concatenate([train_df.iloc[val_idx]["forward_returns"].values for _, val_idx in bh_splits])
    bh_rf = np.concatenate([train_df.iloc[val_idx]["risk_free_rate"].values for _, val_idx in bh_splits])
    bh_pos = np.ones(len(bh_fwd))
    bh_metrics = compute_metrics(pd.Series(bh_fwd), pd.Series(bh_rf), bh_pos)
    # For BH positions=1, strat_ret = fwd_returns, so cum_return = prod(1+fwd)-1
    bh_metrics["cum_return_pct"] = float((np.prod(1 + bh_fwd) - 1) * 100)

    summary_path = _save_cv_summary(all_results, bh_metrics, CV_DIR)
    print(f"\nCV summary saved → {summary_path}")
    print(f"\nReview results in: {CV_DIR}")
    print(f"MLflow UI:  cd v2/ && mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"\nWhen ready: python v2/scripts/run_test_evaluation.py --models <chosen_models>")


if __name__ == "__main__":
    main()
