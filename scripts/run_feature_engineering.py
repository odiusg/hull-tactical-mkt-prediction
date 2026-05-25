from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_engineering import build_features
from hull_tactical.paths import RESULTS_DIR


def main():
    print("=== Feature Engineering: start ===")

    train, _ = load_train_test()                   # load raw train.csv

    full_cleaned, *_ = build_cleaned_data(train)   # cleaned data for FE

    print("\nRunning feature engineering...")
    full_feat_df, feat_train_set, feat_val_set, feat_test_set, new_cols = (
        build_features(full_cleaned)
    )

    print("\n=== Feature Engineering Completed ===")
    print("Total new engineered features:", len(new_cols))

    print("\nFeature sets saved under:", RESULTS_DIR)
    print("=== Feature Engineering: done ===")


if __name__ == "__main__":
    main()
