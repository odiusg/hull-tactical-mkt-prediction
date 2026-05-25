from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_engineering import build_features
from hull_tactical.modelling import train_directional_model
from hull_tactical.backtest import run_backtest
from hull_tactical.paths import RESULTS_DIR, ARTIFACTS_DIR
from hull_tactical.config import TARGET_COL


def main():
    print("=== Full Pipeline: start ===")

    # 1) Load raw data
    print("\n[1] Loading raw train.csv ...")
    train, _ = load_train_test()                             # raw train.csv only

    # 2) Preprocessing
    print("\n[2] Running preprocessing ...")
    full_cleaned, train_clean, val_clean, test_clean, high_na_cols = build_cleaned_data(train)

    print("Preprocessing completed.")
    print("  Full cleaned shape :", full_cleaned.shape)
    print("  Train cleaned shape:", train_clean.shape)
    print("  Val cleaned shape  :", val_clean.shape)
    print("  Test cleaned shape :", test_clean.shape)
    print("  High-NA cols with *_missing flags:", len(high_na_cols))

    # 3) Feature engineering
    print("\n[3] Running feature engineering ...")
    full_feat_df, feat_train_set, feat_val_set, feat_test_set, new_cols = build_features(full_cleaned)

    print("Feature engineering completed.")
    print("  Full feat shape :", full_feat_df.shape)
    print("  Train feat shape:", feat_train_set.shape)
    print("  Val feat shape  :", feat_val_set.shape)
    print("  Test feat shape :", feat_test_set.shape)
    print("  New engineered features:", len(new_cols))

    # 4) Model training
    print("\n[4] Training model (LightGBM + HalvingRandomSearchCV) ...")
    results = train_directional_model(
        full_feat_df,
        feat_train_set,
        feat_val_set,
        feat_test_set,
    )

    print("Model training completed.")
    print("  Best CV AUC:", results["best_cv_auc"])
    test_metrics = results["metrics"]["test"]
    print("  Test AUC:", test_metrics["auc"])
    print("  Test ACC:", test_metrics["acc"])

    # 5) Backtest
    print("\n[5] Running backtest on test set ...")

    best_model = results["model"]
    X_test = results["X_test"]
    y_test = results["y_test"]                               # forward_returns on test
    proba_test = best_model.predict_proba(X_test)[:, 1]      # p(up) for each date

    risk_free_test = feat_test_set["risk_free_rate"].values  # rf path for test

    bt_res = run_backtest(
        proba_test,
        y_test.values,
        risk_free_test,
    )

    print("\nBacktest completed.")
    print("  Baseline adjusted Sharpe:",
          bt_res["baseline"]["adjusted_sharpe"])
    print("  Best strategy name:",
          bt_res["best"]["name"])
    print("  Best strategy adjusted Sharpe:",
          bt_res["best"]["kaggle_details"]["adjusted_sharpe"])

    print("\n=== Summary ===")
    print("Artifacts saved under:", ARTIFACTS_DIR)
    print("Results & reports under:", RESULTS_DIR)
    print("=== Full Pipeline: done ===")


if __name__ == "__main__":
    main()
