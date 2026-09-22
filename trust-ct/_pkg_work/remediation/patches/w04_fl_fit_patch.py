"""
W04 Remediation Patch — fl_fit state-carryover fix
====================================================
Bug:  run_phase5_v3.py::fl_fit creates fresh zero-initialised LogisticClient
      objects every call, discarding the accumulated global weights `w` from
      the outer closed-loop.  The outer loop carries `w` via `w = new_w`, but
      never feeds it back into fl_fit.  Sequential / continuous learning claims
      are therefore not supported by the v3 implementation.

Fix:  Add w_init parameter.  If provided, every client's weights are set to
      w_init before the inner FL rounds begin.

Scope: This patch covers the fl_fit function and the two call sites in the
       outer BETTER-BP loop and the Cimas warm/cold-start loops.
       Double-preprocessing (raw X_bp vs X_bp_pp) is a companion issue noted
       under W04b and is flagged separately in this file.

Status: PATCH READY — pending author approval of W04 rerun (see rerun_plan.md).
"""

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from federated.fedavg import LogisticClient, fedavg_round

# ── Reproduce the hash helper from run_phase5_v3.py ──────────────────────────
import hashlib
def sha256_arr(arr):
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    ).hexdigest()[:16]

N_FL = 5   # must match run_phase5_v3.py constant


# ═══════════════════════════════════════════════════════════════════════════════
# PATCHED FUNCTION  (drop-in replacement for fl_fit in run_phase5_v3.py)
# ═══════════════════════════════════════════════════════════════════════════════

def fl_fit(client_data, n_rounds, seed=7, w_init=None):
    """
    Federated logistic regression — patched for W04.

    Parameters
    ----------
    client_data : dict  {client_id: (X, y)}   X and y are numpy arrays.
    n_rounds    : int   Number of inner FedAvg aggregation rounds.
    seed        : int   Per-client seed passed to LogisticClient.
    w_init      : ndarray or None
        If provided, ALL client weights are set to w_init before training begins.
        When None (default), clients start from their zero-initialised state —
        correct for cold-start warm initialisation; INCORRECT for sequential
        outer-round continuation (that is the W04 bug being patched).

    Returns
    -------
    w     : ndarray   Averaged global weight vector after n_rounds.
    hashes: list[str] Per-client weight hashes (truncated SHA-256).
    """
    valid = {k: (X, y) for k, (X, y) in client_data.items()
             if len(np.unique(y)) > 1 and len(y) >= 4}
    if not valid:
        return None, []

    clients = [
        LogisticClient(cid, X, y, lr=0.02, n_local_epochs=N_FL, seed=seed)
        for cid, (X, y) in valid.items()
    ]

    # ── W04 FIX: broadcast incoming global weights before any local update ──
    if w_init is not None:
        for c in clients:
            c.set_global_weights(w_init)
    # ── end fix ─────────────────────────────────────────────────────────────

    for _ in range(n_rounds):
        fedavg_round(clients, mu=0.0)

    w = clients[0].get_weights()
    return w, [sha256_arr(c.get_weights()) for c in clients]


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED CHANGES TO THE OUTER LOOP IN run_phase5_v3.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# BETTER-BP outer loop (lines 316-376 of run_phase5_v3.py):
#   BEFORE:
#       new_w, upd = fl_fit(cli_data, n_rounds=N_FL, seed=ms)
#   AFTER:
#       new_w, upd = fl_fit(cli_data, n_rounds=N_FL, seed=ms, w_init=w)
#
# Cimas warm-start outer loop:
#   BEFORE:
#       new_w, upd = fl_fit(cli_data_c, n_rounds=N_FL, seed=7)
#   AFTER:
#       new_w, upd = fl_fit(cli_data_c, n_rounds=N_FL, seed=7, w_init=w)
#
# Cimas cold-start outer loop:
#   BEFORE:
#       new_w, upd = fl_fit(cli_data_c, n_rounds=N_FL, seed=ms)
#   AFTER:
#       new_w, upd = fl_fit(cli_data_c, n_rounds=N_FL, seed=ms, w_init=w)
#
# Warm-init call (start of each model-seed loop) is UNCHANGED:
#       init_w_bp, _ = fl_fit(init_data, n_rounds=10, seed=ms)  # w_init=None OK
#
# ═══════════════════════════════════════════════════════════════════════════════
# W04b — DOUBLE-PREPROCESSING NOTE (companion issue)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The outer loop calls fl_fit with raw X slices (e.g., X_bp[mask]) while
# LogisticClient applies its own local imputer + StandardScaler.  Performance
# metrics are then computed via fast_metrics(w, X_bp_pp, Y_bp) where X_bp_pp
# uses a GLOBAL imputer+scaler.
#
# Consequence: the weight vector w is optimised for locally-scaled features
# but evaluated on globally-scaled features.  Under similar site distributions
# the error is small but the inconsistency is real.
#
# Corrective options (author decision required):
#   Option 1: pass X_bp_pp[mask] to fl_fit and configure LogisticClient to
#             skip local preprocessing when data is already standardised.
#   Option 2: expose the global imputer/scaler to fl_fit and apply it to raw
#             slices before creating clients.
#
# Until the author decides, the W04 state fix above is applied in isolation.
# The double-preprocessing issue is documented here but NOT patched, as it
# requires a design decision about the LogisticClient API.
