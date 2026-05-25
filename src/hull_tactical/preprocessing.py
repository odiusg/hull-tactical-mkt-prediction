import numpy as np
import pandas as pd

from .config import NON_FEATURE_COLS, TARGET_COL
from .paths import INTERIM_DIR, RESULTS_DIR
from .utils import get_feature_cols


def clean_extreme_values(df, cols, quantile=0.005, ratio=10, name="Dataset"):
    """Set isolated spikes to NaN: outside [quantile, 1-quantile] AND ≥ratio× both neighbours."""
    cleaned_df = df.copy()
    dirty_info = {}

    cols = [
        c for c in cols
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and c != TARGET_COL
    ]

    for col in cols:
        s = cleaned_df[col]

        lower_thr = s.quantile(quantile)
        upper_thr = s.quantile(1 - quantile)
        candidate = (s < lower_thr) | (s > upper_thr)

        dirty = np.zeros(len(s), dtype=bool)

        for i in range(1, len(s) - 1):
            if not candidate.iloc[i] or pd.isna(s.iloc[i]):
                continue

            val = abs(s.iloc[i])
            prev_val = abs(s.iloc[i - 1])
            next_val = abs(s.iloc[i + 1])

            if pd.isna(prev_val) or pd.isna(next_val):
                continue
            if candidate.iloc[i - 1] or candidate.iloc[i + 1]:
                continue
            if prev_val > 0 and next_val > 0:
                if (val > prev_val * ratio) and (val > next_val * ratio):
                    dirty[i] = True

        cleaned_df.loc[dirty, col] = np.nan

        dirty_count = int(dirty.sum())
        dirty_info[col] = {
            "dirty_count": dirty_count,
            "dirty_rate": float(dirty_count / len(s)),
            "lower_thr": float(lower_thr),
            "upper_thr": float(upper_thr),
        }

    dirty_summary = (
        pd.DataFrame.from_dict(dirty_info, orient="index")
          .sort_values("dirty_rate", ascending=False)
    )

    print(f"\nDirty extreme cleaning summary for {name}:")
    print(dirty_summary.head(10))

    return cleaned_df, dirty_summary


def add_missing_flags(df, high_na_cols):
    """Add a 0/1 `<col>_missing` column for every high-NA column."""
    out = df.copy()
    for col in high_na_cols:
        out[f"{col}_missing"] = df[col].isna().astype(int)
    return out


def build_cleaned_data(
    train: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    quantile: float = 0.005,
    ratio: float = 10.0,
    high_na_threshold: float = 0.40,
    save: bool = True,
    excel_name: str = "cleaned_train.xlsx",
):
    if "date_id" not in train.columns:
        raise ValueError("train must contain a 'date_id' column.")

    df = train.sort_values("date_id").reset_index(drop=True).copy()

    feature_cols = get_feature_cols(df)
    cleaned, dirty_summary = clean_extreme_values(
        df, feature_cols, quantile=quantile, ratio=ratio, name="Train"
    )

    # decide high-NA columns on train portion only to avoid target leakage
    n = len(cleaned)
    train_end = int(train_ratio * n)
    train_part = cleaned.iloc[:train_end]
    missing_rate_train = train_part.isna().mean()
    high_na_cols = missing_rate_train[
        missing_rate_train >= high_na_threshold
    ].index.tolist()
    high_na_cols = [c for c in high_na_cols if c not in NON_FEATURE_COLS]

    print(f"\nHigh-NA columns (>= {high_na_threshold:.0%}, decided on train): "
          f"{len(high_na_cols)}")

    full_cleaned = add_missing_flags(cleaned, high_na_cols)

    val_end = int((train_ratio + val_ratio) * n)
    train_clean = full_cleaned.iloc[:train_end].reset_index(drop=True)
    val_clean = full_cleaned.iloc[train_end:val_end].reset_index(drop=True)
    test_clean = full_cleaned.iloc[val_end:].reset_index(drop=True)

    print("\n===== Cleaned splits =====")
    print("Full cleaned :", full_cleaned.shape)
    print("Train cleaned:", train_clean.shape)
    print("Val cleaned  :", val_clean.shape)
    print("Test cleaned :", test_clean.shape)

    if save:
        out_path = INTERIM_DIR / excel_name
        with pd.ExcelWriter(out_path) as writer:
            full_cleaned.to_excel(
                writer, sheet_name="train_cleaned_with_flags", index=False
            )
            dirty_summary.to_excel(writer, sheet_name="dirty_summary")
            pd.Series(high_na_cols, name="high_na_cols").to_frame().to_excel(
                writer, sheet_name="high_na_cols", index=False
            )
        print("Saved cleaned data to:", out_path)

    return full_cleaned, train_clean, val_clean, test_clean, high_na_cols
