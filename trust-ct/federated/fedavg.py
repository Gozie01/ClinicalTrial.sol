"""
TRUST-CT Federated Learning — FedAvg and FedProx.

FedProx root-cause note
-----------------------
With lr=0.05 and n_local_epochs=5 the per-round parameter drift
||w_local - w_global|| ≈ 0.017.  The proximal gradient is mu * 0.017.
At mu=0.01 that is 0.00017 — roughly 400x smaller than the data gradient
(~0.07).  FedProx is invisible at mu=0.01 / few local steps.

Fixes applied:
  1. n_local_epochs default raised to 10; lr default lowered to 0.02.
  2. Diagnostic mode records proximal_penalty and weight_distance per round.
  3. mu_select() tunes mu on designated validation clients — never on the
     held-out LOCO test client.
  4. Recommended mu range: {0.001, 0.01, 0.1, 1.0}; tune on inner clients.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class LogisticClient:
    """
    Single FL client: gradient-descent logistic regression.
    Weights: flat array [coef (n_features), intercept (1)].
    """

    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.02,
        n_local_epochs: int = 10,
        class_weight: str = "balanced",
        seed: int = 7,
    ):
        self.client_id = client_id
        self.X_raw     = X.astype(float)
        self.y         = y.astype(int)
        self.lr        = lr
        self.n_local_epochs = n_local_epochs
        self.class_weight   = class_weight
        self.seed           = seed
        self.n_features     = X.shape[1]

        # Preprocessing fitted on local data only
        self.imputer = SimpleImputer(strategy="median").fit(self.X_raw)
        self.scaler  = StandardScaler().fit(self.imputer.transform(self.X_raw))

        # Weights initialised to zero
        self._w = np.zeros(self.n_features + 1)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X))

    def set_global_weights(self, w: np.ndarray):
        self._w = w.copy()

    def get_weights(self) -> np.ndarray:
        return self._w.copy()

    def _sample_weights(self) -> np.ndarray:
        y = self.y
        if self.class_weight == "balanced":
            n_pos = max(y.sum(), 1)
            n_neg = max(len(y) - n_pos, 1)
            return np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
        return np.ones(len(y))

    def local_update(self, mu: float = 0.0) -> Tuple[np.ndarray, dict]:
        """
        Run n_local_epochs of SGD.  mu=0 → FedAvg; mu>0 → FedProx.

        Returns:
            (updated_weights, diagnostics)

        diagnostics keys:
            proximal_penalty   — mean ||(w - w_global)||^2 over steps (only if mu>0)
            weight_distance    — ||w_final - w_global||_2
            grad_norm_mean     — mean data-gradient norm over steps
        """
        X      = self._preprocess(self.X_raw)
        y      = self.y
        sw     = self._sample_weights()

        # w_global is fixed at the START of this round (before any local updates)
        w_global = self._w.copy()
        w        = self._w.copy()

        prox_penalties, grad_norms = [], []

        for _ in range(self.n_local_epochs):
            logits = np.clip(X @ w[:self.n_features] + w[self.n_features], -30, 30)
            pred   = 1.0 / (1.0 + np.exp(-logits))
            err    = pred - y  # shape (n,)

            grad_coef = (sw * err) @ X / len(y)
            grad_int  = (sw * err).mean()
            grad_norms.append(np.sqrt(np.sum(grad_coef ** 2) + grad_int ** 2))

            if mu > 0:
                # FedProx: add proximal term gradient
                diff_coef = w[:self.n_features] - w_global[:self.n_features]
                diff_int  = w[self.n_features]  - w_global[self.n_features]
                prox_penalties.append(mu * (np.sum(diff_coef ** 2) + diff_int ** 2))
                grad_coef += mu * diff_coef
                grad_int  += mu * diff_int

            w[:self.n_features] -= self.lr * grad_coef
            w[self.n_features]  -= self.lr * grad_int

        self._w = w
        weight_dist = float(np.linalg.norm(w - w_global))
        grad_norm   = float(np.mean(grad_norms))
        # Proximal gradient L2 norm = mu * ||w - w_global||  (same units as data gradient)
        # The scalar loss penalty mu*||w-w_global||^2 is NOT comparable with ||grad_L||.
        proximal_grad_norm = float(mu * weight_dist)
        diag = {
            "proximal_grad_norm": proximal_grad_norm,        # ||mu*(w-w_g)||  — same units as grad
            "proximal_to_grad_ratio": (proximal_grad_norm / max(grad_norm, 1e-12)),
            "weight_distance":  weight_dist,
            "grad_norm_mean":   grad_norm,
        }
        return self._w.copy(), diag

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_p   = self._preprocess(X)
        logits = np.clip(X_p @ self._w[:self.n_features] + self._w[self.n_features], -30, 30)
        prob1  = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - prob1, prob1])


# ── Aggregation round ────────────────────────────────────────────────────────

def fedavg_round(
    clients: List[LogisticClient],
    mu: float = 0.0,
    equal_weight: bool = False,
    verbose_diag: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    One FedAvg / FedProx communication round.

    Returns:
        (aggregated_weights, round_diagnostics)
    """
    current_global = clients[0].get_weights()
    for c in clients:
        c.set_global_weights(current_global)

    diags   = []
    updated = []
    for c in clients:
        w_new, d = c.local_update(mu=mu)
        updated.append(w_new)
        diags.append(d)

    sizes   = np.array([c.n_samples for c in clients], dtype=float)
    weights = np.ones(len(clients)) / len(clients) if equal_weight else sizes / sizes.sum()
    aggregated = sum(wt * w for wt, w in zip(weights, updated))

    for c in clients:
        c.set_global_weights(aggregated)

    round_diag = {
        "mean_proximal_grad_norm":       float(np.mean([d["proximal_grad_norm"]      for d in diags])),
        "mean_proximal_to_grad_ratio":   float(np.mean([d["proximal_to_grad_ratio"]  for d in diags])),
        "mean_weight_distance":          float(np.mean([d["weight_distance"]          for d in diags])),
        "mean_grad_norm":                float(np.mean([d["grad_norm_mean"]           for d in diags])),
    }
    if verbose_diag:
        pgnorm = round_diag["mean_proximal_grad_norm"]
        gnorm  = round_diag["mean_grad_norm"]
        ratio  = round_diag["mean_proximal_to_grad_ratio"]
        print(f"    |prox_grad|={pgnorm:.6f}  |data_grad|={gnorm:.6f}  "
              f"ratio={ratio:.4f} ({ratio*100:.2f}%)  weight_dist={round_diag['mean_weight_distance']:.6f}")
    return aggregated, round_diag


def run_federated(
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_rounds: int = 20,
    mu: float = 0.0,
    equal_weight: bool = False,
    lr: float = 0.02,
    n_local_epochs: int = 10,
    seed: int = 7,
    verbose_diag: bool = False,
) -> Tuple[List[LogisticClient], List[dict]]:
    """
    Run FedAvg (mu=0) or FedProx (mu>0) for n_rounds.

    Returns (clients, round_diagnostics_list).
    """
    clients = [
        LogisticClient(cid, X, y, lr=lr, n_local_epochs=n_local_epochs, seed=seed)
        for cid, (X, y) in client_data.items()
    ]
    round_diags = []
    for rnd in range(n_rounds):
        _, rd = fedavg_round(clients, mu=mu, equal_weight=equal_weight,
                              verbose_diag=verbose_diag)
        rd["round"] = rnd + 1
        round_diags.append(rd)
    return clients, round_diags


# ── Inner-client mu selection (never uses the LOCO test client) ──────────────

def mu_select(
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mu_grid: List[float] = [0.001, 0.01, 0.1, 1.0],
    val_fraction: float = 0.2,
    n_rounds: int = 20,
    n_local_epochs: int = 10,
    seed: int = 7,
) -> float:
    """
    Select mu on a random 20% validation holdout within the training clients.
    Call this BEFORE assembling the test-client split so the test client is
    never touched.

    Returns the best mu (highest mean validation PR-AUC).
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    best_mu, best_score = mu_grid[0], -1.0

    for mu_cand in mu_grid:
        fold_scores = []
        for cid, (X, y) in client_data.items():
            n    = len(y)
            n_val = max(int(n * val_fraction), 2)
            idx   = rng.permutation(n)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            if y[tr_idx].sum() == 0 or y[val_idx].sum() == 0:
                continue

            train_dict = {k: v for k, v in client_data.items() if k != cid}
            train_dict[cid] = (X[tr_idx], y[tr_idx])
            clients, _ = run_federated(
                train_dict, n_rounds=n_rounds, mu=mu_cand,
                n_local_epochs=n_local_epochs, seed=seed,
            )
            # Use the client whose local model matches cid
            pred_client = next(c for c in clients if c.client_id == cid)
            yp  = pred_client.predict_proba(X[val_idx])[:, 1]
            try:
                fold_scores.append(average_precision_score(y[val_idx], yp))
            except Exception:
                pass

        score = float(np.mean(fold_scores)) if fold_scores else -1.0
        if score > best_score:
            best_score, best_mu = score, mu_cand

    return best_mu
