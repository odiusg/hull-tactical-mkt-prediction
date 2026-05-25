from hull_tactical.data_loading import load_train_test
from hull_tactical.preprocessing import build_cleaned_data
from hull_tactical.paths import RESULTS_DIR


def main():
    print("=== Preprocessing: start ===")

    train, _ = load_train_test() # ignore test

    full_cleaned, train_clean, val_clean, test_clean, high_na_cols = build_cleaned_data(train)

    print("\n=== Preprocessing: summary ===")
    print("Full cleaned shape :", full_cleaned.shape)
    print("Train cleaned shape:", train_clean.shape)
    print("Val cleaned shape  :", val_clean.shape)
    print("Test cleaned shape :", test_clean.shape)

    print("\nHigh-NA columns with *_missing flags created:")
    print(high_na_cols)

    print("\nCleaned data saved under:", RESULTS_DIR)
    print("=== Preprocessing: done ===")


if __name__ == "__main__":
    main()