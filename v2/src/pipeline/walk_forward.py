from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .backtest import competition_score
from .config import INITIAL_TRAIN_DAYS, K_DEFAULT, K_ROLL_WINDOW, N_FOLDS_TUNE, TARGET_COL
from .feature_selection import select_engineered_features
from .models import apply_scaler, fit_scaler, map_positions, needs_scaling


def walk_forward_splits(
    n: int,
    initial_train: int = INITIAL_TRAIN_DAYS,
    n_folds: int = N_FOLDS_TUNE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    remaining = n - initial_train
    if remaining <= 0:
        raise ValueError(f"Dataset too small: n={n}, initial_train={initial_train}")
    val_size = remaining // n_folds
    if val_size == 0:
        raise ValueError(
            f"val_size=0: not enough data for {n_folds} folds "
            f"(n={n}, initial_train={initial_train})"
        )
    # Distribute remainder (+1 row) to the first `remainder` folds so all data
    # is covered and the maximum size difference between any two folds is 1 row.
    remainder = remaining % n_folds
    splits = []
    pos = initial_train
    for fold in range(n_folds):
        fold_size = val_size + (1 if fold < remainder else 0)
        splits.append((np.arange(0, pos), np.arange(pos, pos + fold_size)))
        pos += fold_size
    return splits


def _fit_on_train(
    df_train: pd.DataFrame,
    raw_to_eng: dict[str, list[str]],
    all_columns: list[str],
    model_factory,
    verbose: bool = False,
) -> tuple:
    """Feature-select on engineered features, prep X/y, optionally scale, fit."""
    # Gather all engineered candidates that exist in this fold's data.
    # D* raw columns have no engineered versions (not in raw_to_eng), so they
    # are naturally excluded.  rate_regime participates in IC filtering like any
    # other candidate rather than being injected unconditionally.
    eng_candidates = [c for cols in raw_to_eng.values() for c in cols if c in all_columns]
    if "rate_regime" in all_columns:
        eng_candidates.append("rate_regime")

    eng_features, stats = select_engineered_features(
        df_train, candidate_eng_cols=eng_candidates
    )
    eng_features = [c for c in eng_features if c in all_columns]

    if not eng_features:
        return None, [], None, pd.Series(dtype=float), stats

    mask = df_train[TARGET_COL].notna()
    X = df_train.loc[mask, eng_features].copy()
    y = df_train.loc[mask, TARGET_COL]
    col_means = X.mean()

    nan_count = int(X.isna().sum().sum())
    if verbose and nan_count > 0:
        n_cols_with_nan = int((X.isna().any()).sum())
        print(f"    [NaN fill] {nan_count} cells in {n_cols_with_nan}/{len(eng_features)} features → col_means")

    X = X.fillna(col_means)

    model = model_factory()
    scaler = None
    if needs_scaling(model):
        scaler = fit_scaler(X)
        X = apply_scaler(scaler, X)

    model.fit(X, y)
    return model, eng_features, scaler, col_means, stats


def run_walk_forward_cv(
    feature_df: pd.DataFrame,
    raw_to_eng: dict[str, list[str]],
    model_factory,
    n_folds: int = N_FOLDS_TUNE,
    k: float = K_DEFAULT,
    verbose: bool = True,
    compute_train_score: bool = True,
    collect_predictions: bool = False,
) -> dict:
    splits = walk_forward_splits(len(feature_df), n_folds=n_folds)
    all_cols = list(feature_df.columns)
    fold_results = []
    all_preds: list[np.ndarray] = []
    all_actuals: list[np.ndarray] = []
    all_positions_list: list[np.ndarray] = []
    all_fwd_list: list[np.ndarray] = []
    all_rf_list: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        df_train = feature_df.iloc[train_idx].reset_index(drop=True)
        df_val = feature_df.iloc[val_idx].reset_index(drop=True)

        model, eng_features, scaler, col_means, sel_stats = _fit_on_train(
            df_train, raw_to_eng, all_cols, model_factory, verbose=verbose
        )
        if not eng_features:
            if verbose:
                print(f"  Fold {fold}: no features selected, skipping.")
            continue

        va_mask = df_val[TARGET_COL].notna()
        X_val = df_val.loc[va_mask, eng_features].fillna(col_means)
        y_val = df_val.loc[va_mask, TARGET_COL]
        if scaler is not None:
            X_val = apply_scaler(scaler, X_val)
        pred_val = model.predict(X_val)

        mae = mean_absolute_error(y_val, pred_val)
        mse = mean_squared_error(y_val, pred_val)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_val, pred_val)

        train_mae = train_rmse = train_r2 = np.nan
        if compute_train_score:
            tr_mask = df_train[TARGET_COL].notna()
            X_tr = df_train.loc[tr_mask, eng_features].fillna(col_means)
            y_tr = df_train.loc[tr_mask, TARGET_COL]
            if scaler is not None:
                X_tr = apply_scaler(scaler, X_tr)
            pred_tr = model.predict(X_tr)
            train_mae = mean_absolute_error(y_tr, pred_tr)
            train_rmse = float(np.sqrt(mean_squared_error(y_tr, pred_tr)))
            train_r2 = r2_score(y_tr, pred_tr)

        # Prepend the last K_ROLL_WINDOW training predictions so that rolling_std
        # in map_positions starts from a warm state rather than cold-starting on
        # the validation set alone (which would cause expanding-std distortion for
        # the first min_periods rows of each fold).
        tr_mask_ctx = df_train[TARGET_COL].notna()
        X_ctx = df_train.loc[tr_mask_ctx, eng_features].fillna(col_means).iloc[-K_ROLL_WINDOW:]
        if scaler is not None:
            X_ctx = apply_scaler(scaler, X_ctx)
        pred_ctx = model.predict(X_ctx)
        positions_full = map_positions(np.concatenate([pred_ctx, pred_val]), k=k)
        positions = positions_full[len(pred_ctx):]
        rf_val = df_val.loc[va_mask, "risk_free_rate"].reset_index(drop=True)
        fwd_val = df_val.loc[va_mask, "forward_returns"].reset_index(drop=True)

        if collect_predictions:
            all_preds.append(pred_val)
            all_actuals.append(y_val.values)
            all_positions_list.append(positions)
            all_fwd_list.append(fwd_val.values)
            all_rf_list.append(rf_val.values)

        try:
            adj_sharpe = competition_score(fwd_val, rf_val, positions)
        except Exception as e:
            adj_sharpe = np.nan
            if verbose:
                print(f"  Fold {fold}: competition_score error: {e}")

        result = {
            "fold": fold,
            "train_size": int(df_train[TARGET_COL].notna().sum()),
            "val_size": int(va_mask.sum()),
            "n_features": len(eng_features),
            "n_eng_selected": sel_stats["n_selected"],
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "adjusted_sharpe": adj_sharpe,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
        }
        fold_results.append(result)

        if verbose:
            print(
                f"  Fold {fold:2d} | train={result['train_size']:5d} val={result['val_size']:5d} "
                f"feat={result['n_features']:3d} | "
                f"MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}  "
                f"AdjSharpe={adj_sharpe:.4f}"
            )

    results_df = pd.DataFrame(fold_results)
    summary = {c: results_df[c].mean() for c in ["mae", "mse", "rmse", "r2", "adjusted_sharpe"]}

    if verbose:
        print(f"\n  CV averages:")
        for k_name, v in summary.items():
            print(f"    {k_name}: {v:.6f}")

    out = {"folds": fold_results, "summary": summary, "results_df": results_df}
    if collect_predictions and all_preds:
        out["all_preds"] = np.concatenate(all_preds)
        out["all_actuals"] = np.concatenate(all_actuals)
        out["all_positions"] = np.concatenate(all_positions_list)
        out["all_fwd"] = np.concatenate(all_fwd_list)
        out["all_rf"] = np.concatenate(all_rf_list)
    return out


def train_final_model(
    feature_df: pd.DataFrame,
    raw_to_eng: dict[str, list[str]],
    model_factory,
) -> tuple:
    model, eng_features, scaler, col_means, _ = _fit_on_train(
        feature_df, raw_to_eng, list(feature_df.columns), model_factory
    )
    return model, eng_features, scaler, col_means
