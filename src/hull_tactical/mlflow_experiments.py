import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb

from .config import TARGET_COL
from .paths import ARTIFACTS_DIR

EXPERIMENT_NAME = "hull_tactical_regression"
LSTM_LOOKBACK = 20
_DB_URI = f"sqlite:///{ARTIFACTS_DIR}/mlflow.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_mse": float(mean_squared_error(y_true, y_pred)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def _drop_nan_target(X, y):
    mask = ~np.isnan(y)
    return X[mask], y[mask]


def _prepare_linear(feat_train, feat_val, feat_test, feature_cols):
    """Impute (train-fit) + scale (train-fit). Returns 6 arrays + transformers."""
    raw = [df[feature_cols].values for df in (feat_train, feat_val, feat_test)]
    imputer = SimpleImputer(strategy="mean").fit(raw[0])
    imputed = [imputer.transform(x) for x in raw]
    scaler = StandardScaler().fit(imputed[0])
    scaled = [scaler.transform(x) for x in imputed]

    out = []
    for X, df in zip(scaled, (feat_train, feat_val, feat_test)):
        y = df[TARGET_COL].values
        out.extend(_drop_nan_target(X, y))
    return tuple(out), imputer, scaler


def _prepare_trees(feat_train, feat_val, feat_test, feature_cols):
    """Tree models handle NaN natively — extract arrays and drop NaN targets only."""
    out = []
    for df in (feat_train, feat_val, feat_test):
        X = df[feature_cols].values
        y = df[TARGET_COL].values
        out.extend(_drop_nan_target(X, y))
    return tuple(out)


def _create_sequences(X, y, lookback):
    """(N, F), (N,) → (N-L, L, F), (N-L,)  — uses sliding window view for efficiency."""
    n = len(y)
    idx = np.arange(n - lookback)
    X_seq = np.stack([X[i: i + lookback] for i in idx]).astype(np.float32)
    return X_seq, y[lookback:].astype(np.float32)


# ── LSTM architecture ─────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Per-model runners ─────────────────────────────────────────────────────────

def _run_sklearn(name, model, params, X_tr, y_tr, X_va, y_va, X_te, y_te):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        model.fit(X_tr, y_tr)
        m = {}
        for prefix, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            m.update(_metrics(y, model.predict(X), prefix))
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(model, "model")
    print(f"  [{name:20s}] val_r2={m['val_r2']:+.4f}  test_r2={m['test_r2']:+.4f}")
    return m


def _run_xgboost(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "early_stopping_rounds": 30, "random_state": 42, "n_jobs": -1}
    with mlflow.start_run(run_name="XGBoost"):
        log_p = {k: v for k, v in params.items()
                 if k not in ("early_stopping_rounds", "n_jobs")}
        mlflow.log_params(log_p)
        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        m = {}
        for prefix, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            m.update(_metrics(y, model.predict(X), prefix))
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(model, "model")
    print(f"  [{'XGBoost':20s}] val_r2={m['val_r2']:+.4f}  test_r2={m['test_r2']:+.4f}")
    return m


def _run_lgbm(X_tr, y_tr, X_va, y_va, X_te, y_te):
    params = {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05,
              "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
              "verbose": -1, "random_state": 42}
    with mlflow.start_run(run_name="LightGBM"):
        mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        m = {}
        for prefix, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            m.update(_metrics(y, model.predict(X), prefix))
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(model, "model")
    print(f"  [{'LightGBM':20s}] val_r2={m['val_r2']:+.4f}  test_r2={m['test_r2']:+.4f}")
    return m


def _run_lstm(feat_train, feat_val, feat_test, feature_cols, imputer, scaler,
              lookback=LSTM_LOOKBACK, hidden=64, layers=2, dropout=0.2,
              epochs=80, batch_size=64, lr=1e-3, patience=10):
    # Build full sequential array (impute+scale fitted on train only)
    full = pd.concat([feat_train, feat_val, feat_test], ignore_index=True)
    X_full = scaler.transform(imputer.transform(full[feature_cols].values))
    y_full = full[TARGET_COL].values

    X_seq, y_seq = _create_sequences(X_full, y_full, lookback)

    # Split boundaries: y_seq[i] predicts y at position i+lookback in the full array
    n_tr = len(feat_train) - lookback
    n_va = len(feat_val)

    def _loader(X_s, y_s, shuffle):
        mask = ~np.isnan(y_s)
        ds = TensorDataset(torch.from_numpy(X_s[mask]),
                           torch.from_numpy(y_s[mask]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = _loader(X_seq[:n_tr], y_seq[:n_tr], shuffle=True)
    va_loader = _loader(X_seq[n_tr:n_tr + n_va], y_seq[n_tr:n_tr + n_va], shuffle=False)
    te_loader = _loader(X_seq[n_tr + n_va:], y_seq[n_tr + n_va:], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _LSTMNet(X_seq.shape[2], hidden, layers, dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(net(xb.to(device)), yb.to(device)).item() * len(xb)
                for xb, yb in va_loader
            ) / len(va_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    early stop at epoch {epoch + 1}")
                break

    net.load_state_dict(best_state)
    net.eval()

    def _predict(loader):
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds.append(net(xb.to(device)).cpu().numpy())
                trues.append(yb.numpy())
        return np.concatenate(trues), np.concatenate(preds)

    params = dict(lookback=lookback, hidden=hidden, layers=layers,
                  dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr)
    with mlflow.start_run(run_name="LSTM"):
        mlflow.log_params(params)
        m = {}
        for prefix, loader in [("train", tr_loader), ("val", va_loader), ("test", te_loader)]:
            yt, yp = _predict(loader)
            m.update(_metrics(yt, yp, prefix))
        mlflow.log_metrics(m)
        mlflow.pytorch.log_model(net, "model")
    print(f"  [{'LSTM':20s}] val_r2={m['val_r2']:+.4f}  test_r2={m['test_r2']:+.4f}")
    return m


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_experiments(feat_train_set, feat_val_set, feat_test_set, feature_cols,
                        lookback=LSTM_LOOKBACK):
    """Run 6 regression models and log results to MLflow. Returns per-model metrics dict."""
    mlflow.set_tracking_uri(_DB_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n===== MLflow: {EXPERIMENT_NAME} (tracking: {_DB_URI}) =====")

    (X_tr, y_tr, X_va, y_va, X_te, y_te), imputer, scaler = _prepare_linear(
        feat_train_set, feat_val_set, feat_test_set, feature_cols
    )
    X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t = _prepare_trees(
        feat_train_set, feat_val_set, feat_test_set, feature_cols
    )

    results = {}
    results["LinearRegression"] = _run_sklearn(
        "LinearRegression", LinearRegression(), {},
        X_tr, y_tr, X_va, y_va, X_te, y_te,
    )
    results["Ridge"] = _run_sklearn(
        "Ridge", Ridge(alpha=1.0), {"alpha": 1.0},
        X_tr, y_tr, X_va, y_va, X_te, y_te,
    )
    results["Lasso"] = _run_sklearn(
        "Lasso", Lasso(alpha=0.001), {"alpha": 0.001},
        X_tr, y_tr, X_va, y_va, X_te, y_te,
    )
    results["XGBoost"] = _run_xgboost(X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t)
    results["LightGBM"] = _run_lgbm(X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t)
    results["LSTM"] = _run_lstm(
        feat_train_set, feat_val_set, feat_test_set, feature_cols,
        imputer, scaler, lookback=lookback,
    )

    print("\n===== Summary (sorted by test_r2) =====")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["test_r2"], reverse=True)
    for name, m in ranked:
        print(f"  {name:20s}  test_r2={m['test_r2']:+.4f}  "
              f"test_mse={m['test_mse']:.6f}  test_mae={m['test_mae']:.6f}")

    print(f"\nMLflow UI: mlflow ui --backend-store-uri {_DB_URI}")
    return results
