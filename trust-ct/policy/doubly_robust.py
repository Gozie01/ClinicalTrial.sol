"""
TRUST-CT Policy — Cross-fitted Doubly Robust CATE estimation.

Formulation:
    tau(x) = E[Y | A=1, X=x] - E[Y | A=0, X=x]

BETTER-BP RCT: randomisation probability e(x) = P(A=1) = 2/3 (known, fixed).

Cross-fitted DR estimator (Chernozhukov et al. 2018):
    psi_i = mu1(X_i) - mu0(X_i)
            + A_i/e * (Y_i - mu1(X_i))
            - (1-A_i)/(1-e) * (Y_i - mu0(X_i))

where mu1, mu0 are nuisance outcome models fitted on the other K-1 folds.

Budget-constrained allocation:
    Given budget B and predicted uplift tau_i per patient,
    treat patients in descending tau order until budget is exhausted.
    Ties broken by higher baseline risk (mu0).

NOT: prospective clinical validation.
NOT: Qini/uplift on an RCT holdout (we use BETTER-BP for offline evaluation only).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _make_outcome_model(seed: int = 7) -> Pipeline:
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scl",   StandardScaler()),
        ("clf",   LogisticRegression(C=0.1, max_iter=1000,
                                      class_weight="balanced",
                                      random_state=seed)),
    ])


def cross_fit_dr(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    e: float = 2 / 3,
    n_folds: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Cross-fitted DR pseudo-outcomes for individual treatment effect estimation.

    Args:
        X: (n, p) feature matrix
        y: (n,)   binary outcome (e.g. nonattendance_12m)
        a: (n,)   binary treatment indicator (1 = incentive arm)
        e: known propensity score P(A=1) — fixed by randomisation
        n_folds:  number of cross-fitting folds

    Returns:
        DataFrame with columns:
            mu1, mu0   — outcome model predictions under each arm
            tau_dr     — DR pseudo-outcome (individual-level CATE estimate)
            psi        — same as tau_dr (alias)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    a = np.asarray(a, dtype=int)
    n = len(y)

    mu1_hat = np.full(n, np.nan)
    mu0_hat = np.full(n, np.nan)

    # Stratify on (y, a) jointly where possible
    strat = y * 2 + a
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in cv.split(X, strat):
        X_tr, y_tr, a_tr = X[train_idx], y[train_idx], a[train_idx]
        X_te              = X[test_idx]

        # Fit outcome model on treated and control separately
        m1 = _make_outcome_model(seed).fit(X_tr[a_tr == 1], y_tr[a_tr == 1])
        m0 = _make_outcome_model(seed).fit(X_tr[a_tr == 0], y_tr[a_tr == 0])

        mu1_hat[test_idx] = m1.predict_proba(X_te)[:, 1]
        mu0_hat[test_idx] = m0.predict_proba(X_te)[:, 1]

    # DR pseudo-outcome
    psi = (
        mu1_hat - mu0_hat
        + a / e * (y - mu1_hat)
        - (1 - a) / (1 - e) * (y - mu0_hat)
    )

    return pd.DataFrame({
        "mu1":    mu1_hat,
        "mu0":    mu0_hat,
        "tau_dr": psi,
        "psi":    psi,
    })


def fit_cate_model(
    X_train: np.ndarray,
    tau_train: np.ndarray,
    seed: int = 7,
) -> Pipeline:
    """
    Meta-learner: fit E[tau | X] using the DR pseudo-outcomes as regression targets.
    Uses logistic regression on sign(tau) as a simple classifier-based S-learner.
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
    ])
    # Binarise uplift for classification-based meta-learner
    tau_bin = (tau_train > 0).astype(int)
    pipe.fit(X_train, tau_bin)
    return pipe


def budget_allocation(
    tau: np.ndarray,
    mu0: np.ndarray,
    budget_n: int,
) -> np.ndarray:
    """
    Allocate incentives to top-budget_n patients by uplift tau.
    Ties broken by higher baseline risk (mu0).

    Returns binary allocation vector (1 = incentivised).
    """
    n = len(tau)
    score = tau + 1e-6 * mu0   # tie-break by baseline risk
    alloc = np.zeros(n, dtype=int)
    top   = np.argsort(-score)[:budget_n]
    alloc[top] = 1
    return alloc


def evaluate_policy(
    y: np.ndarray,
    a: np.ndarray,
    alloc: np.ndarray,
    e: float = 2 / 3,
) -> dict:
    """
    Inverse-probability-weighted (IPW) policy value estimate.
    V(pi) = E[ Y_i * I(pi(X_i)=A_i) / e(A_i | X_i) ]

    This is a semi-synthetic closed-loop replay, NOT prospective clinical validation.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=int)
    alloc = np.asarray(alloc, dtype=int)

    propensity = np.where(a == 1, e, 1 - e)
    treated_correctly = (alloc == a).astype(float)

    ipw_value = (y * treated_correctly / propensity).mean()
    treat_rate = alloc.mean()

    return {
        "ipw_value":    float(ipw_value),
        "treat_rate":   float(treat_rate),
        "n_treated":    int(alloc.sum()),
        "n_total":      int(len(alloc)),
        "note": "Semi-synthetic closed-loop replay. NOT prospective clinical validation.",
    }
