import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

from .config import NON_FEATURE_COLS
from .utils import get_feature_cols
from .paths import RESULTS_DIR


# Column statistics
def feature_count(df):
    return df.shape[1] 


def na_rate(df):
    na = df.isna().mean().to_frame("na_rate")
    na["na_count"] = df.isna().sum()
    return na.sort_values("na_rate", ascending=False)


def outlier_rate_iqr(df, k=1.5):
    rates = {}
    numeric = df.select_dtypes(include=[np.number])

    for col in numeric:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        outliers = ((numeric[col] < lower) | (numeric[col] > upper)).mean()
        rates[col] = outliers

    return pd.DataFrame({"outlier_rate": rates}).sort_values("outlier_rate", ascending=False)


# Missing heatmap
def plot_missing_heatmap(df, figsize=(10, 6), save_name=None):
    plt.figure(figsize=figsize)
    msno.heatmap(df)
    if save_name:
        plt.savefig(RESULTS_DIR / save_name, bbox_inches="tight")
    plt.close()


# Distribution plots
def plot_distribution_grid(df, cols, n_cols=4, figsize=(16, 12), save_name=None):
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax)
        ax.set_title(col)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    if save_name:
        fig.savefig(RESULTS_DIR / save_name, bbox_inches="tight")
    plt.close()


# One-line EDA runner
def run_basic_eda(train_set, test_set=None, top_n=30):
    print("Train shape:", train_set.shape)
    if test_set is not None:
        print("Test shape:", test_set.shape)

    # ---- missing rate ----
    na_df = na_rate(train_set)
    na_df.to_excel(RESULTS_DIR / "missing_rate_train.xlsx")  # save missing summary

    # ---- outlier rate ----
    out_df = outlier_rate_iqr(train_set)
    out_df.to_excel(RESULTS_DIR / "outlier_rate_train.xlsx")

    # ---- heatmap ----
    plot_missing_heatmap(train_set, save_name="missing_heatmap_train.png")

    # ---- top-k distributions ----
    feat_cols = get_feature_cols(train_set)
    top_cols = out_df.head(top_n).index.tolist()
    plot_distribution_grid(train_set, top_cols, save_name="top_outlier_distributions.png")

    print("EDA completed. Outputs saved to:", RESULTS_DIR)
