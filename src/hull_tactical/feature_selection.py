"""
Feature-selection pipeline for the Hull Tactical Market Prediction project.

Three-stage funnel:
    1. Drop dummy {-1, 0, 1} columns and columns with >40% missing on train.
    2. Variance filter (drop bottom-quartile variance).
    3. Pairwise-correlation filter (uses the EDA-stage correlation report).
    4. LightGBM gain-based selection (keep features that cumulatively
       contribute up to 90% of total gain).

Returns the selected feature list plus the importance DataFrame for plotting.

Extracted from `final_version.ipynb` / `code.md` and refactored into a
reusable module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import TARGET_COL
from .paths import RESULTS_DIR
from .utils import (
    get_feature_cols,
    drop_dummy_cols,
    compute_feature_pair_corr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _time_split(df, train_ratio=0.7, val_ratio=0.2):
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def _is_dummy_like(series):
    """True if the column only takes values in {-1, 0, 1} (ignoring NaN)."""
    vals = set(series.dropna().unique())
    return vals.issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_feature_selection(
    full_cleaned: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    high_na_threshold: float = 0.40,
    var_quantile: float = 0.25,
    top_corr_pairs: int = 200,
    cum_gain_keep: float = 0.90,
    correlation_excel: str = "correlation_stats_trainset.xlsx",
):
    """
    Run the full feature-selection funnel and return the result dict.

    Returns
    -------
    dict with keys
        selected_features : list[str]   final feature list
        importance_df     : pd.DataFrame LightGBM gain table
        intermediate      : dict        sizes after each filter
    """
    print("===== Feature selection pipeline =====")

    train_clean, val_clean, test_clean = _time_split(
        full_cleaned, train_ratio, val_ratio
    )

    # ---- Step 0.1: start from all non-meta / non-finance columns
    all_features = get_feature_cols(full_cleaned)
    print(f"Initial feature pool: {len(all_features)}")

    # ---- Step 0.2: drop dummy {-1, 0, 1} columns (but keep *_missing flags)
    feature_no_dummy = [
        c for c in all_features
        if not (_is_dummy_like(train_clean[c]) and not c.endswith("_missing"))
    ]
    print(f"After dropping dummies: {len(feature_no_dummy)}")

    # ---- Step 0.3: drop columns with > high_na_threshold missing on train
    miss_rate = train_clean[feature_no_dummy].isna().mean()
    high_na = miss_rate[miss_rate > high_na_threshold].index.tolist()
    feature_after_missing = [c for c in feature_no_dummy if c not in high_na]
    print(f"Dropped high-NA cols (> {high_na_threshold:.0%}): {len(high_na)}")
    print(f"After NA filter: {len(feature_after_missing)}")

    # ---- Step 1: variance filter (drop bottom var_quantile of variances)
    var_series = train_clean[feature_after_missing].var(ddof=0)
    var_sorted = var_series.sort_values()
    var_threshold = var_sorted.quantile(var_quantile)
    low_var_cols = var_sorted[var_sorted <= var_threshold].index.tolist()
    feature_after_var = [
        c for c in feature_after_missing if c not in low_var_cols
    ]
    print(f"\nStep 1 | variance threshold = {var_threshold:.2e}")
    print(f"        dropped {len(low_var_cols)}, remaining {len(feature_after_var)}")

    # ---- Step 2: pairwise-correlation filter
    # Try to load a pre-computed correlation report; if not present, compute it.
    corr_path = RESULTS_DIR / correlation_excel
    if corr_path.exists():
        top_pairs_df = pd.read_excel(corr_path, sheet_name="Feature_Feature_Top")
        print(f"\nStep 2 | loaded correlation pairs from {corr_path.name}")
    else:
        print(f"\nStep 2 | correlation file not found, computing on the fly...")
        corr_res = compute_feature_pair_corr(
            train_clean,
            feature_cols=feature_after_var,
            top_n=top_corr_pairs,
            excel_name=correlation_excel,
        )
        top_pairs_df = corr_res["top_pairs_df"]

    if "abs_corr" in top_pairs_df.columns:
        top_pairs_df = top_pairs_df.sort_values("abs_corr", ascending=False)

    candidate = list(feature_after_var)
    corr_drop = set()
    for _, row in top_pairs_df.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        if f1 not in candidate or f2 not in candidate:
            continue
        v1 = var_series.get(f1, np.nan)
        v2 = var_series.get(f2, np.nan)
        if np.isnan(v1) or np.isnan(v2):
            continue
        drop_col = f2 if v1 >= v2 else f1
        corr_drop.add(drop_col)
        candidate.remove(drop_col)

    feature_after_corr = candidate
    print(f"        dropped {len(corr_drop)} correlated cols, "
          f"remaining {len(feature_after_corr)}")

    # ---- Step 3: LightGBM-importance selection (top cum_gain_keep of gain)
    print(f"\nStep 3 | training a LightGBM baseline for gain-based selection")

    X_train = train_clean[feature_after_corr].copy()
    y_train = train_clean[TARGET_COL].copy()
    X_val = val_clean[feature_after_corr].copy()
    y_val = val_clean[TARGET_COL].copy()

    # Drop rows where target is NaN
    mtr = y_train.notna()
    mva = y_val.notna()
    X_train, y_train = X_train[mtr].reset_index(drop=True), y_train[mtr].reset_index(drop=True)
    X_val, y_val = X_val[mva].reset_index(drop=True), y_val[mva].reset_index(drop=True)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=300,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    gain = model.feature_importance(importance_type="gain")
    importance_df = (
        pd.DataFrame({"feature": feature_after_corr, "gain": gain})
          .sort_values("gain", ascending=False)
          .reset_index(drop=True)
    )

    total_gain = importance_df["gain"].sum()
    importance_df["cum_gain"] = importance_df["gain"].cumsum()
    selected_top_gain = importance_df.loc[
        importance_df["cum_gain"] <= cum_gain_keep * total_gain, "feature"
    ].tolist()
    if not selected_top_gain:                       # always keep at least one
        selected_top_gain = importance_df.head(1)["feature"].tolist()

    print(f"        kept top {cum_gain_keep:.0%} cumulative gain: "
          f"{len(selected_top_gain)} features")

    # ---- Save selection summary
    out_path = RESULTS_DIR / "feature_selection_summary.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        importance_df.to_excel(writer, sheet_name="lgbm_importance", index=False)
        pd.Series(selected_top_gain, name="selected_features").to_frame().to_excel(
            writer, sheet_name="selected_features", index=False
        )
        pd.Series(high_na, name="high_na_cols").to_frame().to_excel(
            writer, sheet_name="high_na_cols", index=False
        )
        pd.Series(low_var_cols, name="low_var_cols").to_frame().to_excel(
            writer, sheet_name="low_var_cols", index=False
        )
        pd.Series(sorted(corr_drop), name="corr_dropped").to_frame().to_excel(
            writer, sheet_name="corr_dropped", index=False
        )
    print(f"\nSaved feature-selection summary to: {out_path}")

    return {
        "selected_features": selected_top_gain,
        "importance_df": importance_df,
        "intermediate": {
            "initial":           len(all_features),
            "after_dummy":       len(feature_no_dummy),
            "after_missing":     len(feature_after_missing),
            "after_variance":    len(feature_after_var),
            "after_correlation": len(feature_after_corr),
            "after_lgbm_gain":   len(selected_top_gain),
        },
        "high_na_cols": high_na,
        "low_var_cols": low_var_cols,
        "corr_drop_cols": sorted(corr_drop),
    }
