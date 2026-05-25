import json
from pathlib import Path

import joblib

from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_engineering import build_features
from hull_tactical.backtest import run_backtest
from hull_tactical.config import TARGET_COL
from hull_tactical.paths import RESULTS_DIR, ARTIFACTS_DIR


def main():
    print("=== Backtest: start ===")

    # 1) Rebuild cleaned and feature data (same pipeline as training)
    train, _ = load_train_test()                              # raw train.csv

    full_cleaned, *_ = build_cleaned_data(train)              # cleaned full data

    print("\nRebuilding feature sets for backtest...")
    full_feat_df, feat_train_set, feat_val_set, feat_test_set, new_cols = (
        build_features(full_cleaned)                          # engineered features
    )

    # 2) Load trained model and model config
    model_path = ARTIFACTS_DIR / "lgbm_directional.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path} (run training first)")

    print("\nLoading trained model from:", model_path)
    best_model = joblib.load(model_path)

    config_path = RESULTS_DIR / "lgbm_halving_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path} (run training first)")

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    feature_cols = model_config["feature_cols"]               # same features used in training
    print("Number of model feature columns:", len(feature_cols))

    # 3) Build test matrices for prediction and backtest
    X_test = feat_test_set[feature_cols].copy()
    y_test = feat_test_set[TARGET_COL].copy()
    risk_free_test = feat_test_set["risk_free_rate"].values   # rf path for backtest

    print("\nShapes for backtest:")
    print("  X_test:", X_test.shape)
    print("  y_test:", y_test.shape)

    # 4) Predict probabilities on test set
    proba_test = best_model.predict_proba(X_test)[:, 1]       # p(up) for each day

    # 5) Run strategy backtest (grid over mapping + vol targeting)
    print("\nRunning strategy backtest grid search...")
    bt_res = run_backtest(
        proba_test,
        y_test.values,
        risk_free_test,
    )

    print("\n=== Backtest: completed ===")
    print("Baseline adjusted Sharpe:",
          bt_res["baseline"]["adjusted_sharpe"])
    print("Best strategy name:",
          bt_res["best"]["name"])
    print("Best strategy adjusted Sharpe:",
          bt_res["best"]["kaggle_details"]["adjusted_sharpe"])

    print("\nBacktest finished. You can inspect `bt_res` in a notebook if needed.")
    print("=== Backtest: done ===")


if __name__ == "__main__":
    main()
