from __future__ import annotations

import pandas as pd

from .config import (
    E_DIFF_WINDOWS,
    I_DIFF_WINDOW,
    I_LAG_WINDOW,
    M_ROLL_WINDOW,
    MAX_LOOKBACK,
    P_ROLL_WINDOW,
    PREFIX_GROUPS,
    RATE_REGIME_THRESHOLD,
    V_ROLL_WINDOW,
)


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Construct all engineered features from raw columns.  All operations are
    backward-looking, so this can be applied to the full dataset without leakage.

    Returns:
        df_out      – DataFrame with original columns + engineered columns
        raw_to_eng  – mapping {raw_col: [engineered_col, ...]}
    """
    out = df.copy()
    raw_to_eng: dict[str, list[str]] = {}

    # M* → 5-day rolling mean
    for col in _present(df, PREFIX_GROUPS["M"]):
        name = f"{col}_roll{M_ROLL_WINDOW}"
        out[name] = df[col].rolling(M_ROLL_WINDOW, min_periods=1).mean()
        raw_to_eng.setdefault(col, []).append(name)

    # E* → 21-day and 63-day difference
    for col in _present(df, PREFIX_GROUPS["E"]):
        for w in E_DIFF_WINDOWS:
            name = f"{col}_diff{w}"
            out[name] = df[col].diff(w)
            raw_to_eng.setdefault(col, []).append(name)

    # I* → 1-period diff then lag 5 days
    for col in _present(df, PREFIX_GROUPS["I"]):
        name = f"{col}_diff{I_DIFF_WINDOW}_lag{I_LAG_WINDOW}"
        out[name] = df[col].diff(I_DIFF_WINDOW).shift(I_LAG_WINDOW)
        raw_to_eng.setdefault(col, []).append(name)

    # P* → deviation from 21-day rolling mean
    for col in _present(df, PREFIX_GROUPS["P"]):
        roll_mean = df[col].rolling(P_ROLL_WINDOW, min_periods=5).mean()
        name = f"{col}_dev{P_ROLL_WINDOW}"
        out[name] = df[col] - roll_mean
        raw_to_eng.setdefault(col, []).append(name)

    # V* → 21-day rolling mean
    for col in _present(df, PREFIX_GROUPS["V"]):
        name = f"{col}_roll{V_ROLL_WINDOW}"
        out[name] = df[col].rolling(V_ROLL_WINDOW, min_periods=1).mean()
        raw_to_eng.setdefault(col, []).append(name)

    # S* → lag 1 day
    for col in _present(df, PREFIX_GROUPS["S"]):
        name = f"{col}_lag1"
        out[name] = df[col].shift(1)
        raw_to_eng.setdefault(col, []).append(name)

    # Interest rate regime dummy (global, always included)
    out["rate_regime"] = (df["risk_free_rate"] >= RATE_REGIME_THRESHOLD).astype(int)

    # Drop rows from the head that have no usable engineered features.
    # This covers both the rolling-window burn-in (MAX_LOOKBACK) and the
    # pre-feature-availability period present in this dataset (~first 1006 rows).
    all_eng = [c for cols in raw_to_eng.values() for c in cols]
    has_any_eng = out[all_eng].notna().any(axis=1)
    first_valid = int(has_any_eng.idxmax()) if has_any_eng.any() else len(out)
    out = out.iloc[first_valid:].reset_index(drop=True)

    return out, raw_to_eng
