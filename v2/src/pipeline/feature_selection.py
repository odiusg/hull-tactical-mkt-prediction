from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import (
    CORR_THRESHOLD,
    IC_T_STAT_MIN,
    PREFIX_GROUPS,
    TARGET_COL,
    VAR_QUANTILE,
)


def get_raw_feature_cols(df: pd.DataFrame) -> list[str]:
    all_raw = [c for group in PREFIX_GROUPS.values() for c in group]
    return [c for c in all_raw if c in df.columns]

_DUMMY_PREFIXES = {"D"}


def _ic_t_stat(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return np.nan, np.nan
    n = len(x)
    denom = np.sqrt(max(1 - ic**2, 1e-12))
    t = ic * np.sqrt(max(n - 2, 1)) / denom
    return float(ic), float(t)


def select_raw_features(
    df_train: pd.DataFrame,
    candidate_raw_cols: list[str],
    target_col: str = TARGET_COL,
    ic_t_min: float = IC_T_STAT_MIN,
    corr_thr: float = CORR_THRESHOLD,
    var_q: float = VAR_QUANTILE,
) -> tuple[list[str], dict]:
    df = df_train[[c for c in candidate_raw_cols + [target_col] if c in df_train.columns]].copy()
    df = df[df[target_col].notna()].reset_index(drop=True)

    non_dummy = [c for c in candidate_raw_cols if c[:1] not in _DUMMY_PREFIXES]

    y = df[target_col].values
    ic_stats: dict[str, dict] = {}
    for col in non_dummy:
        s = df[col]
        valid = s.notna()
        if valid.sum() < 10:
            continue
        x_valid = s[valid].values
        if len(np.unique(x_valid)) <= 1:
            continue
        ic, t = _ic_t_stat(x_valid, y[valid])
        ic_stats[col] = {"ic": ic, "t_stat": t, "abs_t": abs(t)}

    ic_passed = [c for c, v in ic_stats.items() if v["abs_t"] >= ic_t_min]

    var_series = df[ic_passed].var(ddof=0)
    groups: dict[str, list[str]] = defaultdict(list)
    for col in ic_passed:
        groups[col[0]].append(col)

    corr_kept: list[str] = []
    for _, group_cols in groups.items():
        if len(group_cols) <= 1:
            corr_kept.extend(group_cols)
            continue
        ordered = sorted(group_cols, key=lambda c: var_series.get(c, 0.0), reverse=True)
        kept: list[str] = []
        for col in ordered:
            if all(abs(df[col].corr(df[k])) < corr_thr for k in kept):
                kept.append(col)
        corr_kept.extend(kept)

    if corr_kept:
        var_kept = var_series.reindex(corr_kept).dropna()
        threshold = var_kept.quantile(var_q)
        selected = [c for c in corr_kept if var_series.get(c, 0.0) > threshold]
    else:
        selected = corr_kept

    stats = {
        "n_non_dummy": len(non_dummy),
        "n_ic_passed": len(ic_passed),
        "n_corr_kept": len(corr_kept),
        "n_selected": len(selected),
        "ic_stats": ic_stats,
    }
    return selected, stats


def resolve_engineered_features(
    selected_raw: list[str],
    raw_to_eng: dict[str, list[str]],
    extra: list[str] | None = None,
) -> list[str]:
    eng: list[str] = []
    for raw in selected_raw:
        eng.extend(raw_to_eng.get(raw, []))
    if extra:
        eng.extend(extra)
    return list(dict.fromkeys(eng))


def select_engineered_features(
    df_train: pd.DataFrame,
    candidate_eng_cols: list[str],
    target_col: str = TARGET_COL,
    ic_t_min: float = IC_T_STAT_MIN,
    corr_thr: float = CORR_THRESHOLD,
    var_q: float = VAR_QUANTILE,
) -> tuple[list[str], dict]:
    """IC/correlation/variance selection directly on engineered (or any) feature columns.

    D* dummy columns are naturally absent from candidate_eng_cols (they have no
    engineered versions in raw_to_eng), so no explicit dummy exclusion is needed.
    Groups for intra-group correlation filtering are formed by the first character
    of each column name (matches the raw-source prefix letter).
    """
    df = df_train[[c for c in candidate_eng_cols + [target_col] if c in df_train.columns]].copy()
    df = df[df[target_col].notna()].reset_index(drop=True)

    y = df[target_col].values
    ic_stats: dict[str, dict] = {}
    for col in candidate_eng_cols:
        if col not in df.columns:
            continue
        s = df[col]
        valid = s.notna()
        if valid.sum() < 10:
            continue
        x_valid = s[valid].values
        if len(np.unique(x_valid)) <= 1:
            continue
        ic, t = _ic_t_stat(x_valid, y[valid])
        ic_stats[col] = {"ic": ic, "t_stat": t, "abs_t": abs(t)}

    ic_passed = [c for c, v in ic_stats.items() if v["abs_t"] >= ic_t_min]

    var_series = df[ic_passed].var(ddof=0) if ic_passed else pd.Series(dtype=float)

    # Group by raw source column name (e.g. "E1" for "E1_diff21", "E1_diff63").
    # Using only col[0] would bucket all E* derivatives together, causing valid
    # features from different raw sources to compete and eliminate each other.
    _raw_src = re.compile(r"^([A-Z]+\d+)")
    groups: dict[str, list[str]] = defaultdict(list)
    for col in ic_passed:
        m = _raw_src.match(col)
        groups[m.group(1) if m else col[0]].append(col)

    corr_kept: list[str] = []
    for _, group_cols in groups.items():
        if len(group_cols) <= 1:
            corr_kept.extend(group_cols)
            continue
        ordered = sorted(group_cols, key=lambda c: var_series.get(c, 0.0), reverse=True)
        kept: list[str] = []
        for col in ordered:
            if all(abs(df[col].corr(df[k])) < corr_thr for k in kept):
                kept.append(col)
        corr_kept.extend(kept)

    if corr_kept:
        var_kept = var_series.reindex(corr_kept).dropna()
        threshold = var_kept.quantile(var_q)
        selected = [c for c in corr_kept if var_series.get(c, 0.0) > threshold]
    else:
        selected = corr_kept

    stats = {
        "n_candidates": len(candidate_eng_cols),
        "n_ic_passed": len(ic_passed),
        "n_corr_kept": len(corr_kept),
        "n_selected": len(selected),
    }
    return selected, stats
