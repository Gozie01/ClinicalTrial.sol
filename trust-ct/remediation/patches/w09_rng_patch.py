"""
W09 Remediation Patch — Attack RNG reproducibility fixes
=========================================================
Three separate RNG issues identified in run_phase6.py:

  W09a  atk_random_gaussian (line 185):
        Uses np.random.default_rng() with no seed — non-reproducible across
        runs and seeds.

  W09b  Clipped-Gaussian noise in 6C-i utility sweep (lines 609-612):
        Noise RNG seeded with (seed * 1000 + t) — the same key for ALL clients
        in round t.  Every client receives an identical noise vector per round,
        collapsing per-client privacy accounting and understating noise variance.

  W09c  atk_backdoor (line 206):
        Uses np.random.default_rng(42) — fixed seed independent of outer run
        seed.  Backdoor poisoning index is identical across all outer seeds,
        breaking seed-level independence.

Status: PATCH READY — random_gauss configurations should be rerun after this
        fix is applied (see rerun_plan.md W09 decision item).
        Backdoor and noise fixes are also listed in the rerun plan.
"""

import numpy as np

# ── Module-level RNG handles, seeded before each attack dispatch ──────────────
_GAUSS_RNG  = np.random.default_rng(0)   # seeded per (seed, round, client)
_BDOOR_RNG  = np.random.default_rng(0)   # seeded per (seed, round, client)


def seed_attack_rngs(outer_seed: int, round_t: int, client_i: int) -> None:
    """
    Call this immediately before dispatching any attack function inside run_fl.
    Ensures all stochastic attacks are reproducible and client-specific.

    Key: outer_seed * 10_000_000 + round_t * 10_000 + client_i * 10 + attack_offset
    The large multipliers ensure no collisions across reasonable parameter ranges
    (outer_seed < 1000, round_t < 1000, client_i < 100).
    """
    global _GAUSS_RNG, _BDOOR_RNG
    base = outer_seed * 10_000_000 + round_t * 10_000 + client_i * 10
    _GAUSS_RNG = np.random.default_rng(base + 1)
    _BDOOR_RNG = np.random.default_rng(base + 2)


# ═══════════════════════════════════════════════════════════════════════════════
# W09a PATCH — atk_random_gaussian
# ═══════════════════════════════════════════════════════════════════════════════

ATTACK_SCALE = 3.0   # must match run_phase6.py

def atk_random_gaussian_fixed(delta, w_global, X, y):
    """
    W09a fix: uses module-level _GAUSS_RNG seeded by seed_attack_rngs().
    Caller MUST invoke seed_attack_rngs(seed, t, i) before calling this.
    """
    scale = ATTACK_SCALE * np.linalg.norm(delta) + 1e-8
    return _GAUSS_RNG.normal(0, scale, size=delta.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# W09b PATCH — per-client noise in 6C-i utility sweep
# ═══════════════════════════════════════════════════════════════════════════════

CLIP_NORM = 1.0   # must match run_phase6.py

def add_clipped_gaussian_noise_fixed(delta, alpha, outer_seed, round_t, client_idx):
    """
    W09b fix: every (seed, round, client) triple gets a unique noise draw.

    ORIGINAL (broken):
        noise = np.random.default_rng(seed * 1000 + t).normal(
            0, alpha * CLIP_NORM, size=delta.shape)
        # All clients in round t share the same noise vector.

    FIXED:
        noise = np.random.default_rng(
            outer_seed * 1_000_000 + round_t * 1_000 + client_idx
        ).normal(0, alpha * CLIP_NORM, size=delta.shape)

    Drop this function's body directly into the inner client loop in Section 6C-i.
    """
    if alpha == 0:
        return delta.copy()
    rng   = np.random.default_rng(
        outer_seed * 1_000_000 + round_t * 1_000 + client_idx
    )
    noise = rng.normal(0, alpha * CLIP_NORM, size=delta.shape)
    return delta + noise


# ═══════════════════════════════════════════════════════════════════════════════
# W09c PATCH — atk_backdoor
# ═══════════════════════════════════════════════════════════════════════════════

BDOOR_FRAC   = 0.30
TRIGGER_FEAT = 0
TRIGGER_VAL  = 3.0
BDOOR_TARGET = 0

def local_sgd_placeholder(w0, X, y, n_epochs=5, lr=0.02):
    """Placeholder — replace with actual local_sgd from run_phase6.py."""
    raise NotImplementedError("Import local_sgd from run_phase6.py")

def atk_backdoor_fixed(delta, w_global, X, y):
    """
    W09c fix: uses module-level _BDOOR_RNG seeded by seed_attack_rngs().
    Caller MUST invoke seed_attack_rngs(seed, t, i) before calling this.

    ORIGINAL (broken):
        idx_poison = np.random.default_rng(42).choice(...)   # always seed=42

    FIXED:
        idx_poison = _BDOOR_RNG.choice(...)   # seed = f(outer_seed, t, i)
    """
    n_poison   = max(1, int(BDOOR_FRAC * len(y)))
    idx_poison = _BDOOR_RNG.choice(len(y), size=n_poison, replace=False)
    X_p, y_p   = X.copy(), y.copy()
    X_p[idx_poison, TRIGGER_FEAT] = TRIGGER_VAL
    y_p[idx_poison]               = BDOOR_TARGET
    # NOTE: replace local_sgd_placeholder with the actual local_sgd from run_phase6.py
    w_local = local_sgd_placeholder(w_global.copy(), X_p, y_p)
    return w_local - w_global


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED CHANGES TO run_phase6.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. In run_fl() — add call to seed_attack_rngs before dispatch:
#
#   BEFORE (lines ~279-286):
#       if i in mal_idx:
#           if is_alie:
#               delta = atk_alie(...)
#           elif attack_name == "model_replace":
#               delta = atk_model_replace(...)
#           else:
#               delta = atk_fn(delta, w_global, X_c, y_c)
#
#   AFTER:
#       if i in mal_idx:
#           seed_attack_rngs(seed, t, i)   # <-- insert this line
#           if is_alie:
#               delta = atk_alie(...)
#           elif attack_name == "model_replace":
#               delta = atk_model_replace(...)
#           else:
#               delta = atk_fn(delta, w_global, X_c, y_c)
#
# 2. Replace atk_random_gaussian with atk_random_gaussian_fixed in ATTACK_FNS.
#
# 3. Replace atk_backdoor with atk_backdoor_fixed in ATTACK_FNS.
#
# 4. In Section 6C-i inner client loop (lines ~599-613):
#   BEFORE:
#       if alpha > 0:
#           noise = np.random.default_rng(seed * 1000 + t).normal(
#               0, alpha * CLIP_NORM, size=delta.shape)
#           delta = delta + noise
#
#   AFTER:
#       delta = add_clipped_gaussian_noise_fixed(delta, alpha, seed, t, ci_idx)
#   where ci_idx is the enumeration index over eicu_clients in the inner loop.
#
# ═══════════════════════════════════════════════════════════════════════════════
# OPERATING-POINT NOTE (W09 related)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The operating point (alpha=0.01) selection in Phase 6D uses X_eicu_te (test
# set) for the AUROC comparison that determines alpha.  If alpha selection is
# intended to be exploratory / historical (retrospective reporting), this is
# acceptable but must be stated explicitly.  If it is intended as prospective
# specification, the alpha must have been fixed before final test evaluation.
# AUTHOR DECISION REQUIRED — see rerun_plan.md item W09-OP.
