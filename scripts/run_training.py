from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_engineering import build_features
from hull_tactical.modelling import train_directional_model
from hull_tactical.paths import RESULTS_DIR, ARTIFACTS_DIR


def main():
    print("=== Model Training: start ===")

    train, _ = load_train_test()

    full_cleaned, *_ = build_cleaned_data(train)

    print("\nRunning feature engineering...")
    full_feat_df, feat_train_set, feat_val_set, feat_test_set, new_cols = (
        build_features(full_cleaned)
    )

    print("\nRunning model training (LightGBM + HalvingRandomSearchCV)...")
    results = train_directional_model(
        full_feat_df,
        feat_train_set,
        feat_val_set,
        feat_test_set,
    )

    print("\n=== Model Training Completed ===")
    print("Best CV AUC:", results["best_cv_auc"])

    test_metrics = results["metrics"]["test"]
    print("Test AUC:", test_metrics["auc"])
    print("Test ACC:", test_metrics["acc"])

    print("\nArtifacts saved under:", ARTIFACTS_DIR)
    print("Reports saved under:", RESULTS_DIR)
    print("=== Model Training: done ===")


if __name__ == "__main__":
    main()
