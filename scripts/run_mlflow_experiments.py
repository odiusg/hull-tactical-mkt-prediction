from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_engineering import build_features
from hull_tactical.feature_selection import run_feature_selection
from hull_tactical.mlflow_experiments import run_all_experiments
from hull_tactical.mlflow_experiments import EXPERIMENT_NAME, _DB_URI


def main():
    print("=== MLflow Experiment Pipeline: start ===")

    # 1) Load and preprocess
    train, _ = load_train_test()
    full_cleaned, *_ = build_cleaned_data(train)

    # 2) Feature engineering
    full_feat_df, feat_train_set, feat_val_set, feat_test_set, _ = build_features(full_cleaned)

    # 3) Feature selection — use selected features for all models
    fs_result = run_feature_selection(full_cleaned)
    feature_cols = fs_result["selected_features"]
    print(f"\nUsing {len(feature_cols)} selected features for all models.")

    # 4) Run all 6 regression experiments
    results = run_all_experiments(feat_train_set, feat_val_set, feat_test_set, feature_cols)

    print("\n=== MLflow Experiment Pipeline: done ===")
    print(f"View results: mlflow ui --backend-store-uri {_DB_URI}")


if __name__ == "__main__":
    main()
