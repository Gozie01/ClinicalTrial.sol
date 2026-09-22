"""
SCAFFOLD federated learning (Karimireddy et al., 2020).

SCAFFOLD corrects for client drift by maintaining per-client control variates
c_i and a global control variate c.

Update rule (client i):
    w_i ← w_i - lr * (g_i - c_i + c)
    c_i ← c_i - c + (w_global - w_i) / (n_steps * lr)
    c   ← c + (1/N) * sum(c_i_new - c_i_old)

Implementation: logistic-regression gradient descent matching fedavg.py style.
"""

import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ScaffoldClient:
    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.05,
        n_local_steps: int = 5,
        seed: int = 7,
    ):
        self.client_id = client_id
        self.X_raw = X.astype(float)
        self.y     = y.astype(int)
        self.lr    = lr
        self.n_local_steps = n_local_steps
        self.n_features = X.shape[1]

        self.imputer = SimpleImputer(strategy="median").fit(self.X_raw)
        self.scaler  = StandardScaler().fit(self.imputer.transform(self.X_raw))

        self._w  = np.zeros(self.n_features + 1)
        self._ci = np.zeros(self.n_features + 1)  # local control variate

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def _preprocess(self, X):
        return self.scaler.transform(self.imputer.transform(X))

    def local_update(self, w_global: np.ndarray, c_global: np.ndarray) -> tuple:
        X   = self._preprocess(self.X_raw)
        y   = self.y
        w   = w_global.copy()
        ci_old = self._ci.copy()

        n_pos = max(y.sum(), 1)
        n_neg = max(len(y) - n_pos, 1)
        sw = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

        for _ in range(self.n_local_steps):
            logits = np.clip(X @ w[:self.n_features] + w[self.n_features], -30, 30)
            pred   = 1 / (1 + np.exp(-logits))
            err    = pred - y

            grad = np.empty_like(w)
            grad[:self.n_features] = (sw * err) @ X / len(y)
            grad[self.n_features]  = (sw * err).mean()

            # SCAFFOLD correction
            corrected_grad = grad - self._ci + c_global
            w -= self.lr * corrected_grad

        # Update local control variate
        ci_new = ci_old - c_global + (w_global - w) / (self.n_local_steps * self.lr)
        self._w  = w
        self._ci = ci_new
        return w.copy(), (ci_new - ci_old)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_p = self._preprocess(X)
        logits = np.clip(X_p @ self._w[:self.n_features] + self._w[self.n_features], -30, 30)
        prob1  = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - prob1, prob1])


def run_scaffold(
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_rounds: int = 20,
    lr: float = 0.05,
    n_local_steps: int = 5,
    seed: int = 7,
) -> List[ScaffoldClient]:
    clients = [
        ScaffoldClient(cid, X, y, lr=lr, n_local_steps=n_local_steps, seed=seed)
        for cid, (X, y) in client_data.items()
    ]
    n     = len(clients)
    w_g   = np.zeros(clients[0].n_features + 1)
    c_g   = np.zeros_like(w_g)

    for _ in range(n_rounds):
        delta_c_sum = np.zeros_like(c_g)
        w_updates   = []
        for cl in clients:
            w_new, delta_c = cl.local_update(w_g, c_g)
            w_updates.append(w_new)
            delta_c_sum += delta_c

        sizes = np.array([cl.n_samples for cl in clients], dtype=float)
        weights = sizes / sizes.sum()
        w_g = sum(wt * wu for wt, wu in zip(weights, w_updates))
        c_g = c_g + delta_c_sum / n

        for cl in clients:
            cl._w = w_g.copy()

    return clients
