from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from .config import K_DEFAULT, K_MIN_PERIODS, K_ROLL_WINDOW

RANDOM_STATE = 42


def needs_scaling(model) -> bool:
    return isinstance(model, (LinearRegression, Ridge, Lasso, ElasticNet)) or getattr(model, "needs_scaling", False)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


def map_positions(
    predictions: pd.Series | np.ndarray,
    k: float = K_DEFAULT,
    window: int = K_ROLL_WINDOW,
    min_periods: int = K_MIN_PERIODS,
) -> np.ndarray:
    """
    position = clip(1 + k * pred / rolling_std(pred, window), 0, 2)
    Returns neutral (1.0) when predictions are constant (std ≈ 0).
    """
    pred = pd.Series(predictions, dtype=float).reset_index(drop=True)
    roll_std = pred.rolling(window, min_periods=min_periods).std()
    roll_std = roll_std.fillna(pred.expanding(min_periods=1).std())
    std_near_zero = roll_std.abs() < 1e-12
    if std_near_zero.all():
        return np.ones(len(pred))
    roll_std = roll_std.where(~std_near_zero, np.nan).ffill().bfill().fillna(1.0)
    pos = 1.0 + k * pred / roll_std
    return np.clip(pos.values, 0.0, 2.0)


def get_model(name: str, **kwargs):
    registry = {
        "linear":     lambda: LinearRegression(),
        "ridge":      lambda: Ridge(random_state=RANDOM_STATE, **kwargs),
        "lasso":      lambda: Lasso(random_state=RANDOM_STATE, max_iter=5000, **kwargs),
        "elasticnet": lambda: ElasticNet(random_state=RANDOM_STATE, max_iter=5000, **kwargs),
        "rf":         lambda: RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, **kwargs
        ),
        "xgboost":    lambda: xgb.XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=0, eval_metric="rmse", **kwargs
        ),
        "lightgbm":   lambda: lgb.LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=-1,
            verbose=-1, **kwargs
        ),
        "lstm": lambda: __import_lstm(**kwargs),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(registry)}")
    return registry[name]()


def __import_lstm(**kwargs):
    from .lstm_model import LSTMWrapper
    return LSTMWrapper(**kwargs)


def suggest_params(trial, model_name: str) -> dict:
    if model_name == "linear":
        return {}
    if model_name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
    if model_name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1.0, log=True)}
    if model_name == "elasticnet":
        return {
            "alpha":    trial.suggest_float("alpha", 1e-4, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "rf":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 200),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":    1,
            "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
    if model_name == "lstm":
        return {
            "lookback":    trial.suggest_int("lookback", 10, 63),
            "hidden_size": trial.suggest_int("hidden_size", 32, 128, step=16),
            "n_layers":    trial.suggest_int("n_layers", 1, 2),
            "dropout":     trial.suggest_float("dropout", 0.0, 0.3),
            "epochs":      trial.suggest_int("epochs", 20, 80, step=10),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
    raise ValueError(f"No param space defined for '{model_name}'")
