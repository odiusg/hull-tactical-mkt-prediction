import numpy as np
import pandas as pd

from .data_loading import time_based_split
from .config import (
    MACRO_COLS,
    MACRO_WINDOWS,
    MD_COLS,
    MOM_WINDOWS,
    RANK_WINDOW,
    PRICE_COLS,
    MR_WINDOWS,
    SENT_COLS,
    SENT_SMOOTH_WINDOW,
    SENT_SPIKE_WINDOW,
)
from .paths import RESULTS_DIR


# Rolling rank helper
def rolling_rank_percentile(arr):
    order = np.argsort(arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks[-1] / len(arr)


# Main feature engineering
def build_features(full_cleaned,
                   train_ratio=0.7,
                   val_ratio=0.2,
                   excel_name="cleaned_feature_all.xlsx"):
    if "date_id" not in full_cleaned.columns:
        raise ValueError("full_cleaned must contain 'date_id'")

    full_feat_df = full_cleaned.sort_values("date_id").reset_index(drop=True).copy()
    new_feature_cols = []

    # Macro features
    for col in MACRO_COLS:
        if col not in full_feat_df.columns:
            print(f"[WARN] Macro col {col} not in DataFrame, skip.")
            continue

        s = full_feat_df[col]

        for w in MACRO_WINDOWS:
            roll_mean = s.rolling(window=w, min_periods=1).mean()
            denom = roll_mean.replace(0, np.nan)
            dev_ratio = (s - roll_mean) / denom

            new_col = f"{col}_dev_ratio_w{w}"
            full_feat_df[new_col] = dev_ratio
            new_feature_cols.append(new_col)

    # Market dynamics / momentum
    for col in MD_COLS:
        if col not in full_feat_df.columns:
            print(f"[WARN] Market dynamic col {col} not in DataFrame, skip momentum.")
            continue

        s = full_feat_df[col]

        for w in MOM_WINDOWS:
            prev = s.shift(w)
            denom = prev.replace(0, np.nan)
            mom = (s / denom) - 1

            new_col = f"{col}_mom_w{w}"
            full_feat_df[new_col] = mom
            new_feature_cols.append(new_col)

    # Rolling rank percentile
    for col in MD_COLS:
        if col not in full_feat_df.columns:
            continue

        s = full_feat_df[col]
        rank_series = s.rolling(window=RANK_WINDOW, min_periods=5).apply(
            rolling_rank_percentile,
            raw=True,
        )

        new_col = f"{col}_rank_pct_w{RANK_WINDOW}"
        full_feat_df[new_col] = rank_series
        new_feature_cols.append(new_col)

    # Price / valuation mean-reversion
    for col in PRICE_COLS:
        if col not in full_feat_df.columns:
            print(f"[WARN] Price/valuation col {col} not in DataFrame, skip mean-reversion.")
            continue

        s = full_feat_df[col]

        for w in MR_WINDOWS:
            roll_mean = s.rolling(window=w, min_periods=5).mean()
            roll_std = s.rolling(window=w, min_periods=5).std(ddof=0)
            denom = roll_std.replace(0, np.nan)
            zscore = (s - roll_mean) / denom

            new_col = f"{col}_mr_z_w{w}"
            full_feat_df[new_col] = zscore
            new_feature_cols.append(new_col)

    # Sentiment smoothing + spike
    for col in SENT_COLS:
        if col not in full_feat_df.columns:
            print(f"[WARN] Sentiment col {col} not in DataFrame, skip smoothing/spike.")
            continue

        s = full_feat_df[col]

        smooth = s.rolling(window=SENT_SMOOTH_WINDOW, min_periods=1).mean()
        new_col_smooth = f"{col}_smooth_w{SENT_SMOOTH_WINDOW}"
        full_feat_df[new_col_smooth] = smooth
        new_feature_cols.append(new_col_smooth)

        roll_mean = s.rolling(window=SENT_SPIKE_WINDOW, min_periods=3).mean()
        roll_std = s.rolling(window=SENT_SPIKE_WINDOW, min_periods=3).std(ddof=0)
        denom = roll_std.replace(0, np.nan)
        spike_z = (s - roll_mean) / denom

        new_col_spike = f"{col}_spike_z_w{SENT_SPIKE_WINDOW}"
        full_feat_df[new_col_spike] = spike_z
        new_feature_cols.append(new_col_spike)

    print(f"\nTotal new engineered features created (full set): {len(new_feature_cols)}")
    print(new_feature_cols)

    feat_train_set, feat_val_set, feat_test_set = time_based_split(
        full_feat_df, train_ratio, val_ratio, sort_by=None
    )

    print("\n===== Feature sets with engineered features =====")
    print("Train with features:", feat_train_set.shape)
    print("Val with features  :", feat_val_set.shape)
    print("Test with features :", feat_test_set.shape)

    feature_all_path = RESULTS_DIR / excel_name

    with pd.ExcelWriter(feature_all_path) as writer:
        full_feat_df.to_excel(writer,
                              sheet_name="full_with_features",
                              index=False)
        feat_train_set.to_excel(writer,
                                sheet_name="train_with_features",
                                index=False)
        feat_val_set.to_excel(writer,
                              sheet_name="val_with_features",
                              index=False)
        feat_test_set.to_excel(writer,
                               sheet_name="test_with_features",
                               index=False)

    print(f"\nSaved full/train/val/test feature sets to: {feature_all_path}")

    return full_feat_df, feat_train_set, feat_val_set, feat_test_set, new_feature_cols
