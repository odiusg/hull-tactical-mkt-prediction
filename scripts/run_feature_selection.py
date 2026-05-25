from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.feature_selection import run_feature_selection
from hull_tactical.paths import RESULTS_DIR


def main():
    print("=== Feature Selection: start ===")

    train, _ = load_train_test()

    full_cleaned, *_ = build_cleaned_data(train)

    print("\nRunning feature selection on cleaned data...")
    fs_res = run_feature_selection(full_cleaned)

    selected = fs_res["selected_features"]
    print("\n=== Feature Selection Completed ===")
    print("Number of selected features:", len(selected))
    print("First 5 selected features:", selected[:5])

    print("\nSaved outputs under:", RESULTS_DIR)
    print("=== Feature Selection: done ===")


if __name__ == "__main__":
    main()
