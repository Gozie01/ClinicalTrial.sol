"""
TRUST-CT Security — Byzantine attack simulations on federated aggregation.

Attack types (matching existing eICU experiments in federated_mlp_run.py):
  1. label_flip   — attacker flips labels before local training
  2. gaussian     — attacker adds Gaussian noise to gradients
  3. sign_flip    — attacker negates gradient direction
  4. scaling      — attacker amplifies gradient by scale_factor
  5. min_max      — gradient replacement to maximise worst-case deviation

Defence baselines:
  - Coordinate-wise median
  - Trimmed mean (alpha% clipped)
  - Krum / Multi-Krum

Note on update perturbation:
  The "DP AUC" column in prior eICU results is relabelled here as
  'empirical_leakage_mitigation': clipped Gaussian perturbation intended to
  reduce reconstruction leakage.  NO formal (epsilon, delta)-DP guarantee is
  made — no privacy accountant was applied.
"""

import numpy as np
from typing import List


def label_flip_attack(y: np.ndarray, flip_fraction: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_adv = y.copy()
    n_flip = int(len(y) * flip_fraction)
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y_adv[idx] = 1 - y_adv[idx]
    return y_adv


def gaussian_attack(gradient: np.ndarray, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return gradient + rng.normal(0, sigma, size=gradient.shape)


def sign_flip_attack(gradient: np.ndarray) -> np.ndarray:
    return -gradient


def scaling_attack(gradient: np.ndarray, scale: float = 10.0) -> np.ndarray:
    return gradient * scale


def min_max_attack(
    gradient: np.ndarray,
    honest_gradients: List[np.ndarray],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Minimise similarity to other clients while maximising gradient norm.
    Simple implementation: perturb in direction that maximises distance
    from the mean of honest gradients.
    """
    mean_h = np.mean(honest_gradients, axis=0)
    direction = gradient - mean_h
    norm = np.linalg.norm(direction) + 1e-9
    return gradient + gamma * direction / norm


# ── Robust aggregation ──────────────────────────────────────────────────────

def coordinate_median(gradients: List[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(gradients, axis=0), axis=0)


def trimmed_mean(gradients: List[np.ndarray], alpha: float = 0.1) -> np.ndarray:
    G = np.stack(gradients, axis=0)  # (n_clients, d)
    n = G.shape[0]
    k = max(1, int(np.floor(alpha * n)))
    G_sorted = np.sort(G, axis=0)
    return G_sorted[k: n - k].mean(axis=0)


def krum(gradients: List[np.ndarray], f: int = 1) -> np.ndarray:
    """
    Multi-Krum: select the n-f-2 gradients with smallest sum of distances
    to their nearest neighbours, then average them.
    f = number of assumed Byzantine clients.
    """
    G = np.stack(gradients, axis=0)
    n = G.shape[0]
    select = max(1, n - f - 2)
    scores = np.zeros(n)
    for i in range(n):
        dists = np.sum((G - G[i]) ** 2, axis=1)
        dists[i] = np.inf
        scores[i] = np.sort(dists)[: n - f - 1].sum()
    selected = np.argsort(scores)[:select]
    return G[selected].mean(axis=0)


# ── Empirical leakage mitigation (NOT formal DP) ───────────────────────────

def empirical_update_perturbation(
    gradient: np.ndarray,
    clip_norm: float = 1.0,
    noise_sigma: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    Clipped Gaussian perturbation intended to reduce gradient reconstruction
    leakage.

    Description: clip gradient to L2 norm <= clip_norm, then add zero-mean
    Gaussian noise with std=noise_sigma.

    This is NOT differential privacy: no formal (epsilon, delta) accounting
    is performed, no per-sample gradient computation, no privacy amplification
    by subsampling. Label: 'empirical_leakage_mitigation'.
    """
    norm = np.linalg.norm(gradient)
    if norm > clip_norm:
        gradient = gradient * clip_norm / norm
    rng = np.random.default_rng(seed)
    return gradient + rng.normal(0, noise_sigma, size=gradient.shape)
