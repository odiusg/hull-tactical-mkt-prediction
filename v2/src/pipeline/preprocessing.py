from __future__ import annotations

import pandas as pd

from .config import (
    DATA_RAW,
    HIGH_NA_THRESHOLD,
    META_COLS,
    PREFIX_GROUPS,
    TARGET_COL,
    TEST_RATIO,
)


def load_data(path=None) -> pd.DataFrame:
    p = path or DATA_RAW / "train.csv"
    df = pd.read_csv(p)
    df = df.sort_values("date_id").reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill non-target, non-meta columns across the full dataset.

    Drop decisions are intentionally deferred to drop_high_na_cols, which is
    called after build_features + train_test_split so the NA rate is computed
    on the actual (post-burn-in) training rows only.
    """
    cleaned = df.copy()
    fill_cols = [c for c in cleaned.columns if c not in META_COLS + [TARGET_COL]]
    cleaned[fill_cols] = cleaned[fill_cols].ffill()
    return cleaned


def drop_high_na_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_to_eng: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop high-NA raw feature columns (and their derived columns) from both splits.

    NA rate is computed on train_df (post-burn-in) only, avoiding the ~12 pp
    inflation from all-NaN header rows that preprocess sees.

    If raw_to_eng is provided, engineered columns derived from each dropped raw
    column are also removed.  This keeps _fit_on_train's eng_candidates list
    self-consistent: candidates are filtered to `c in all_columns`, and columns
    removed here are absent from all_columns, so they never reach IC selection.
    Without this cleanup those derived columns would silently remain in the
    DataFrame and be passed as dead candidates on every fold.
    """
    all_raw = {c for group in PREFIX_GROUPS.values() for c in group}
    raw_cols = [c for c in train_df.columns if c in all_raw]
    na_rate = train_df[raw_cols].isna().mean()
    dropped = na_rate[na_rate >= HIGH_NA_THRESHOLD].index.tolist()

    to_drop = list(dropped)
    if raw_to_eng is not None:
        derived = [eng for raw in dropped for eng in raw_to_eng.get(raw, [])]
        to_drop.extend(derived)

    return (
        train_df.drop(columns=to_drop, errors="ignore"),
        test_df.drop(columns=to_drop, errors="ignore"),
        dropped,
    )


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 90/10 split.  Test set is the final TEST_RATIO of rows."""
    n = len(df)
    split = int((1 - TEST_RATIO) * n)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test
