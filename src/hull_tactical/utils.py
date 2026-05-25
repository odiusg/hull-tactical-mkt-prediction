import numpy as np
import pandas as pd

from .config import NON_FEATURE_COLS, TARGET_COL
from .paths import RESULTS_DIR


# Feature helpers
def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]   # drop meta/finance cols


# Correlation with target
def corr_with_target(df, target_col=TARGET_COL, features=None, method="pearson"):
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in DataFrame")

    if features is None:
        features = get_feature_cols(df)

    numeric = df[features].select_dtypes(include=[np.number])
    y = df[target_col]

    corr_series = numeric.corrwith(y, method=method)
    corr_df = corr_series.to_frame("corr").dropna().reset_index()
    corr_df = corr_df.rename(columns={"index": "feature"})
    corr_df["abs_corr"] = corr_df["corr"].abs()

    corr_df = corr_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return corr_df


# Pairwise feature correlation and Excel export
def compute_feature_pair_corr(df,
                              feature_cols=None,
                              target_col=TARGET_COL,
                              top_n=200,
                              method="pearson",
                              excel_name="correlation_stats_trainset.xlsx"):
    if feature_cols is None:
        feature_cols = get_feature_cols(df)

    feat_df = df[feature_cols].select_dtypes(include=[np.number])
    corr_mat = feat_df.corr(method=method)

    pairs = []
    cols = corr_mat.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            f1 = cols[i]
            f2 = cols[j]
            val = corr_mat.iloc[i, j]
            pairs.append((f1, f2, val, abs(val)))

    pairs_df = pd.DataFrame(pairs,
                            columns=["feature_1", "feature_2", "corr", "abs_corr"])
    pairs_df = pairs_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    top_pairs_df = pairs_df.head(top_n)

    target_corr_df = None
    if target_col in df.columns:
        target_corr_df = corr_with_target(df,
                                          target_col=target_col,
                                          features=feature_cols,
                                          method=method)

    excel_path = RESULTS_DIR / excel_name

    with pd.ExcelWriter(excel_path) as writer:
        top_pairs_df.to_excel(writer,
                              sheet_name="Feature_Feature_Top",
                              index=False)
        if target_corr_df is not None:
            target_corr_df.to_excel(writer,
                                    sheet_name="Feature_Target",
                                    index=False)

    print(f"Saved correlation stats to: {excel_path}")
    return {
        "corr_matrix": corr_mat,
        "pairs_df": pairs_df,
        "top_pairs_df": top_pairs_df,
        "target_corr_df": target_corr_df,
    }
