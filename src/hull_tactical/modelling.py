import json
import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from .config import TARGET_COL, RANDOM_STATE, N_SPLITS, PARAM_DISTRIBUTIONS
from .paths import RESULTS_DIR, ARTIFACTS_DIR

# Helpers
def drop_nan_target(X, y, name="Dataset"):
    mask = y.notna()
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[{name}] Dropped {dropped} rows with NaN target.")
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def make_direction_labels(y, threshold=0.0):
    return (y > threshold).astype(int)                    # >0 as long direction


def evaluate_classifier(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred_label = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, pred_label)

    print(f"[{name}] AUC={auc:.4f}  ACC={acc:.4f}")
    return {"auc": float(auc), "acc": float(acc)}


# LGBM + HalvingRandomSearchCV
class LGBMClassifierPatched(LGBMClassifier):
    def set_params(self, **params):
        if "max_depth" in params and isinstance(params["max_depth"], (int, np.integer)):
            if params["max_depth"] == 0:
                params["max_depth"] = -1                  # LightGBM uses -1 as "no limit"
        return super().set_params(**params)


def build_model_feature_cols(full_feat_df, extra_drop=None):
    drop_cols = [TARGET_COL, "market_forward_excess_returns"]  # remove target + leakage
    if extra_drop is not None:
        drop_cols = drop_cols + list(extra_drop)

    model_feature_cols = [c for c in full_feat_df.columns if c not in drop_cols]
    model_feature_cols = sorted(model_feature_cols)
    print(f"Final features for model: {len(model_feature_cols)}")
    return model_feature_cols


def train_directional_model(full_feat_df,
                            feat_train_set,
                            feat_val_set,
                            feat_test_set,
                            extra_drop=None):
    model_feature_cols = build_model_feature_cols(full_feat_df, extra_drop=extra_drop)

    X_train = feat_train_set[model_feature_cols].copy()
    y_train = feat_train_set[TARGET_COL].copy()

    X_val = feat_val_set[model_feature_cols].copy()
    y_val = feat_val_set[TARGET_COL].copy()

    X_test = feat_test_set[model_feature_cols].copy()
    y_test = feat_test_set[TARGET_COL].copy()

    print("Shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val  :", X_val.shape,   " y_val  :", y_val.shape)
    print("  X_test :", X_test.shape,  " y_test :", y_test.shape)

    X_train, y_train = drop_nan_target(X_train, y_train, "Train")
    X_val, y_val = drop_nan_target(X_val, y_val, "Validation")
    X_test, y_test = drop_nan_target(X_test, y_test, "Test")

    y_train_cls = make_direction_labels(y_train)
    y_val_cls = make_direction_labels(y_val)
    y_test_cls = make_direction_labels(y_test)

    print("\nLabel distribution (Train):")
    print(y_train_cls.value_counts(normalize=True))

    base_estimator = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_distributions = PARAM_DISTRIBUTIONS               # from config.py
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)               # time-series CV

    base_estimator_patched = LGBMClassifierPatched(**base_estimator.get_params())

    halving_cv = HalvingRandomSearchCV(
        estimator=base_estimator_patched,
        param_distributions=param_distributions,
        resource="n_estimators",
        max_resources=500,
        min_resources=50,
        factor=3,
        cv=tscv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_candidates=30,
        verbose=2,
        n_jobs=-1,
        refit=True,
    )

    print("\n===== Run HalvingRandomSearchCV (train set only) =====")
    halving_cv.fit(X_train, y_train_cls)

    best_model = halving_cv.best_estimator_
    best_params = halving_cv.best_params_
    best_cv_score = halving_cv.best_score_

    print("\n===== Best params from HalvingRandomSearchCV =====")
    for k, v in best_params.items():
        print(f"{k:20s}: {v}")
    print(f"\nBest CV AUC (TimeSeriesSplit, train only): {best_cv_score:.6f}")

    print("\n===== Evaluation on hold-out sets (classification) =====")
    metrics_train = evaluate_classifier(best_model, X_train, y_train_cls, "Train")
    metrics_val = evaluate_classifier(best_model, X_val, y_val_cls, "Validation")
    metrics_test = evaluate_classifier(best_model, X_test, y_test_cls, "Test")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "lgbm_directional.joblib"
    joblib.dump(best_model, model_path)
    print("Saved trained model to:", model_path)

    model_config = {
        "model_type": "LGBMClassifier",
        "best_params": best_params,
        "feature_cols": model_feature_cols,
        "n_features": len(model_feature_cols),
        "target_col": TARGET_COL,
        "cv": {
            "type": "TimeSeriesSplit",
            "n_splits": tscv.n_splits,
        },
        "search": {
            "search_type": "HalvingRandomSearchCV",
            "resource": "n_estimators",
            "min_resources": 50,
            "max_resources": 500,
            "factor": 3,
            "n_candidates": 30,
            "scoring": "roc_auc",
        },
    }

    config_path = RESULTS_DIR / "lgbm_halving_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    print("Saved model config to:", config_path)

    eval_report = {
        "cv": {"best_auc": float(best_cv_score)},
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
    }

    eval_path = RESULTS_DIR / "lgbm_halving_eval_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print("Saved evaluation report to:", eval_path)

    print("\n===== Model training pipeline finished (directional classifier). =====")

    return {
        "model": best_model,
        "feature_cols": model_feature_cols,
        "best_params": best_params,
        "best_cv_auc": float(best_cv_score),
        "metrics": eval_report["metrics"],
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_cls": y_train_cls,
        "y_val_cls": y_val_cls,
        "y_test_cls": y_test_cls,
    }
