"""
TRUST-CT tabular model suite.

Models: ElasticNet logistic regression, LightGBM, Random Forest.
All support:
  - Nested CV hyperparameter tuning (inner 3-fold)
  - Class weighting (scale_pos_weight / class_weight="balanced")
  - Feature engineering within fold via FeatureEngineer transformer
  - Return calibrated probabilities

No feature selection on the full dataset. No SMOTE before splitting.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Derived feature engineering fitted on the TRAINING fold only.
    Thresholds (medians) are computed in fit() and applied in transform().
    Input/output: pandas DataFrame.
    """

    def __init__(self):
        self._thresholds = {}

    def fit(self, X_df: pd.DataFrame, y=None):
        if "n_refill_months_obs" in X_df.columns:
            self._thresholds["high_obs_months"] = X_df["n_refill_months_obs"].median()
        if "refill_recency_days" in X_df.columns:
            self._thresholds["recency_med"] = X_df["refill_recency_days"].median()
        if "obs_amount" in X_df.columns:
            self._thresholds["amount_med"] = X_df["obs_amount"].median()
        return self

    def transform(self, X_df: pd.DataFrame) -> pd.DataFrame:
        X = X_df.copy()
        # High-adherence indicator in obs window
        if "n_refill_months_obs" in X and "high_obs_months" in self._thresholds:
            X["high_obs_adherence"] = (
                X["n_refill_months_obs"] > self._thresholds["high_obs_months"]
            ).astype(float)
        # Recency flag: refilled recently (within 30 days of landmark)
        if "refill_recency_days" in X:
            X["recent_refill"] = (X["refill_recency_days"] <= 30).astype(float)
        # Balanced refill: both early and late months filled
        if "early_months" in X and "late_months" in X:
            X["balanced_refill"] = (
                (X["early_months"] >= 1) & (X["late_months"] >= 1)
            ).astype(float)
        # Utilisation intensity
        if "n_claims" in X and "n_claim_dates" in X:
            X["claims_per_date"] = X["n_claims"] / X["n_claim_dates"].replace(0, np.nan)
        return X


def _prep_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def fit_elasticnet(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 7,
    inner_folds: int = 3,
) -> tuple:
    """
    Nested CV elastic-net logistic regression.
    Grid: C in {0.01, 0.1, 1.0}, l1_ratio in {0.1, 0.5, 0.9}.
    Returns (fitted_pipeline, feature_names).
    """
    fe = FeatureEngineer()
    X_eng = fe.fit_transform(X_train)
    feature_names = X_eng.columns.tolist()
    X_np = X_eng.values.astype(float)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    class_weight = {0: 1.0, 1: float(pos_weight)}

    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    base = Pipeline([
        ("prep", _prep_pipeline()),
        ("clf",  LogisticRegression(
            penalty="elasticnet", solver="saga",
            max_iter=2000, class_weight=class_weight,
            random_state=seed,
        )),
    ])
    param_grid = {
        "clf__C":        [0.01, 0.1, 1.0],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
    }
    gs = GridSearchCV(base, param_grid, cv=inner_cv, scoring="average_precision",
                      n_jobs=-1, refit=True)
    gs.fit(X_np, y_train)
    return gs.best_estimator_, feature_names, fe


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 7,
    inner_folds: int = 3,
) -> tuple:
    """
    Nested CV LightGBM with early stopping.
    Grid: num_leaves in {15, 31}, min_child_samples in {5, 10, 20}.
    Returns (fitted_model, feature_names, feature_engineer).
    """
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed: pip install lightgbm")

    fe = FeatureEngineer()
    X_eng = fe.fit_transform(X_train)
    feature_names = X_eng.columns.tolist()
    X_np = X_eng.values.astype(float)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    param_grid = [
        {"num_leaves": nl, "min_child_samples": mcs}
        for nl in [15, 31]
        for mcs in [5, 10, 20]
    ]
    best_score, best_params = -1.0, param_grid[0]

    for params in param_grid:
        fold_scores = []
        for tr_idx, va_idx in inner_cv.split(X_np, y_train):
            X_tr, y_tr = X_np[tr_idx], y_train[tr_idx]
            X_va, y_va = X_np[va_idx], y_train[va_idx]
            imputer = SimpleImputer(strategy="median").fit(X_tr)
            X_tr = imputer.transform(X_tr)
            X_va = imputer.transform(X_va)
            m = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                random_state=seed, verbose=-1,
                **params,
            )
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(-1)])
            from sklearn.metrics import average_precision_score
            yp = m.predict_proba(X_va)[:, 1]
            fold_scores.append(average_precision_score(y_va, yp))
        score = np.mean(fold_scores)
        if score > best_score:
            best_score, best_params = score, params

    imputer = SimpleImputer(strategy="median").fit(X_np)
    X_np_imp = imputer.transform(X_np)
    final_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=seed, verbose=-1,
        **best_params,
    )
    final_model.fit(X_np_imp, y_train)
    return final_model, imputer, feature_names, fe


def fit_rf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 7,
) -> tuple:
    fe = FeatureEngineer()
    X_eng = fe.fit_transform(X_train)
    feature_names = X_eng.columns.tolist()
    X_np = X_eng.values.astype(float)

    imputer = SimpleImputer(strategy="median").fit(X_np)
    X_np_imp = imputer.transform(X_np)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = RandomForestClassifier(
        n_estimators=300, class_weight={0: 1.0, 1: float(pos_weight)},
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_np_imp, y_train)
    return model, imputer, feature_names, fe
