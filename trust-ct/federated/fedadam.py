"""
TRUST-CT Federated Learning — FedAdam (Reddi et al., 2020).

Server-side adaptive optimisation: the server applies an Adam step to the
aggregated pseudo-gradient (negative client-update direction) rather than
simple averaging.

Client-side: same LogisticClient local updates as FedAvg.
Server-side:
    delta_t  = w_global_t - weighted_average(w_local_i_t)   # pseudo-gradient
    m_t      = beta1 * m_{t-1} + (1 - beta1) * delta_t
    v_t      = beta2 * v_{t-1} + (1 - beta2) * delta_t^2
    w_global_{t+1} = w_global_t - server_lr * m_t / (sqrt(v_t) + epsilon)

Recommended defaults (Reddi et al.):
    server_lr = 0.01,  beta1 = 0.9,  beta2 = 0.99,  epsilon = 1e-3

Note: the sign convention is that delta_t points in the descent direction
(from global toward client average), so the server subtracts.
"""

import numpy as np
from typing import Dict, List, Tuple

from .fedavg import LogisticClient


def run_fedadam(
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_rounds:     int   = 20,
    server_lr:    float = 0.01,
    beta1:        float = 0.9,
    beta2:        float = 0.99,
    epsilon:      float = 1e-3,
    lr:           float = 0.02,
    n_local_epochs: int = 10,
    equal_weight: bool  = False,
    seed:         int   = 7,
    verbose_diag: bool  = False,
) -> Tuple[List[LogisticClient], List[dict]]:
    """
    Run FedAdam for n_rounds.

    Returns (clients, round_diagnostics_list).
    """
    clients = [
        LogisticClient(cid, X, y, lr=lr, n_local_epochs=n_local_epochs, seed=seed)
        for cid, (X, y) in client_data.items()
    ]

    n_dim  = clients[0].n_features + 1
    w_g    = np.zeros(n_dim)
    m      = np.zeros(n_dim)   # 1st moment
    v      = np.zeros(n_dim)   # 2nd moment

    sizes   = np.array([c.n_samples for c in clients], dtype=float)
    agg_w   = sizes / sizes.sum() if not equal_weight else np.ones(len(clients)) / len(clients)

    round_diags = []
    for rnd in range(n_rounds):
        # Broadcast global weights
        for c in clients:
            c.set_global_weights(w_g)

        # Local updates
        local_ws = []
        diags    = []
        for c in clients:
            w_new, d = c.local_update(mu=0.0)   # FedAdam: no proximal on clients
            local_ws.append(w_new)
            diags.append(d)

        # Pseudo-gradient: direction from global toward client average
        w_avg   = sum(wt * w for wt, w in zip(agg_w, local_ws))
        delta_t = w_g - w_avg    # descent direction

        # Adam server update
        m = beta1 * m + (1 - beta1) * delta_t
        v = beta2 * v + (1 - beta2) * (delta_t ** 2)
        m_hat = m / (1 - beta1 ** (rnd + 1))
        v_hat = v / (1 - beta2 ** (rnd + 1))
        w_g   = w_g - server_lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # Push new global back
        for c in clients:
            c.set_global_weights(w_g)

        rd = {
            "round":             rnd + 1,
            "pseudo_grad_norm":  float(np.linalg.norm(delta_t)),
            "mean_weight_dist":  float(np.mean([d["weight_distance"] for d in diags])),
            "mean_grad_norm":    float(np.mean([d["grad_norm_mean"]  for d in diags])),
        }
        if verbose_diag:
            print(f"  R{rnd+1:02d}  |delta|={rd['pseudo_grad_norm']:.6f}  "
                  f"dist={rd['mean_weight_dist']:.6f}")
        round_diags.append(rd)

    return clients, round_diags
