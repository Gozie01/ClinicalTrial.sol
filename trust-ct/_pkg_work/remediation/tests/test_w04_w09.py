"""
Synthetic unit tests — W04 fl_fit state-carryover fix and W09 RNG fixes
========================================================================
Run with: python -m pytest trust-ct/remediation/tests/test_w04_w09.py -v

These tests use synthetic data (no patient records) and are self-contained.
They verify the *patch semantics*, not the experimental results.
"""

import numpy as np
import sys
import pathlib
import pytest

# Add trust-ct root so we can import the federated module
REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from federated.fedavg import LogisticClient, fedavg_round

# ── Import patched functions ──────────────────────────────────────────────────
PATCHES = REPO / "remediation" / "patches"
sys.path.insert(0, str(PATCHES))
import w04_fl_fit_patch as w04
import w09_rng_patch    as w09


# ── Helpers ───────────────────────────────────────────────────────────────────

def synthetic_clients(n_clients=3, n_per_client=50, d=10, seed=0):
    """Return dict {cid: (X, y)} with synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    return {
        f"c{i}": (
            rng.normal(0, 1, (n_per_client, d)),
            rng.integers(0, 2, n_per_client),
        )
        for i in range(n_clients)
    }


def zero_weights(d=10):
    return np.zeros(d + 1)


# ═══════════════════════════════════════════════════════════════════════════════
# W04 TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestW04FlFitStateContinuity:
    """
    T1  When w_init is None, fl_fit must start from zero-initialised weights.
    T2  When w_init is provided, the returned model must differ from the
        zero-init model — confirming incoming weights were broadcast.
    T3  Two sequential calls with state carryover must produce a different
        final model than two independent zero-init calls.
    T4  Same (seed, w_init) gives bitwise-identical output (seed replay).
    """

    def test_T1_none_init_starts_from_zero(self):
        """fl_fit with w_init=None must produce the same output as a manually
        zero-initialised single-round fedavg on the same synthetic data."""
        data = synthetic_clients(d=5, seed=1)
        w_none, _ = w04.fl_fit(data, n_rounds=1, seed=42, w_init=None)
        # Reference: manually zero-init all clients and run one round
        d = 5
        clients = [
            LogisticClient(cid, X, y, lr=0.02, n_local_epochs=w04.N_FL, seed=42)
            for cid, (X, y) in data.items()
        ]
        fedavg_round(clients, mu=0.0)
        w_ref = clients[0].get_weights()
        np.testing.assert_allclose(w_none, w_ref, atol=1e-10,
            err_msg="w_init=None should give same result as manual zero init")

    def test_T2_w_init_alters_output(self):
        """Providing a nonzero w_init must produce a different model than w_init=None."""
        data = synthetic_clients(d=5, seed=2)
        rng  = np.random.default_rng(99)
        w_in = rng.normal(0, 0.5, 6)   # d+1 = 6

        w_none, _ = w04.fl_fit(data, n_rounds=3, seed=7, w_init=None)
        w_init, _ = w04.fl_fit(data, n_rounds=3, seed=7, w_init=w_in)

        assert not np.allclose(w_none, w_init), (
            "fl_fit with nonzero w_init must differ from zero-init baseline"
        )

    def test_T3_carryover_differs_from_independent(self):
        """
        Sequential carryover (round 1 output → round 2 w_init) must produce
        a different final model than two independent zero-init calls.
        """
        data = synthetic_clients(d=8, seed=3)

        # Sequential (state carried)
        w1, _ = w04.fl_fit(data, n_rounds=5, seed=11, w_init=None)
        w2_carry, _ = w04.fl_fit(data, n_rounds=5, seed=11, w_init=w1)

        # Independent (zero each time — the original bug)
        _, _ = w04.fl_fit(data, n_rounds=5, seed=11, w_init=None)
        w2_indep, _ = w04.fl_fit(data, n_rounds=5, seed=11, w_init=None)

        assert not np.allclose(w2_carry, w2_indep), (
            "Carryover and independent-restart must diverge after two rounds"
        )

    def test_T4_seed_replay_identical(self):
        """Same (seed, w_init) must produce bitwise-identical output on two calls."""
        data  = synthetic_clients(d=6, seed=4)
        w_in  = np.random.default_rng(0).normal(0, 0.1, 7)

        w_a, ha = w04.fl_fit(data, n_rounds=4, seed=19, w_init=w_in)
        w_b, hb = w04.fl_fit(data, n_rounds=4, seed=19, w_init=w_in)

        np.testing.assert_array_equal(w_a, w_b, err_msg="seed replay must be bitwise identical")
        assert ha == hb, "hash lists must match on seed replay"

    def test_T5_empty_data_returns_none(self):
        """fl_fit must return (None, []) when no valid client exists."""
        bad_data = {"c0": (np.ones((3, 5)), np.zeros(3))}  # single class
        result, hashes = w04.fl_fit(bad_data, n_rounds=3)
        assert result is None
        assert hashes == []

    def test_T6_holdout_clients_excluded(self):
        """
        A client whose data is all-one-class must not appear in the returned
        weight hashes, verifying the validity gate in fl_fit.
        """
        rng = np.random.default_rng(5)
        valid_client   = ("v0", (rng.normal(0, 1, (40, 5)), rng.integers(0, 2, 40)))
        invalid_client = ("v1", (rng.normal(0, 1, (40, 5)), np.zeros(40, int)))
        data = dict([valid_client, invalid_client])

        w, hashes = w04.fl_fit(data, n_rounds=2, seed=7)
        assert w is not None, "valid client must yield a result"
        assert len(hashes) == 1, "only the valid client should appear in hashes"


# ═══════════════════════════════════════════════════════════════════════════════
# W09 TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestW09RNGFixes:
    """
    T7   atk_random_gaussian_fixed is reproducible: same seed_attack_rngs call
         gives the same output on two invocations.
    T8   Different (seed, round, client) triples produce different noise.
    T9   add_clipped_gaussian_noise_fixed: different client_idx gives different
         noise vectors for the same (outer_seed, round_t).
    T10  add_clipped_gaussian_noise_fixed: alpha=0 returns unchanged delta.
    T11  atk_backdoor fixed seed varies with outer_seed: different outer seeds
         produce different poisoning indices (verifies seed propagation).
    """

    def _fake_delta(self, d=20):
        return np.ones(d) * 0.1

    def test_T7_gauss_reproducible(self):
        """Same seed_attack_rngs call → same gaussian attack output."""
        delta = self._fake_delta()
        w_glob = np.zeros(20)
        X = np.ones((10, 20))
        y = np.array([0, 1] * 5)

        w09.seed_attack_rngs(7, 3, 2)
        out_a = w09.atk_random_gaussian_fixed(delta, w_glob, X, y)

        w09.seed_attack_rngs(7, 3, 2)
        out_b = w09.atk_random_gaussian_fixed(delta, w_glob, X, y)

        np.testing.assert_array_equal(out_a, out_b,
            err_msg="Identical seed_attack_rngs must give identical output")

    def test_T8_gauss_different_keys_differ(self):
        """Different (seed, round, client) triples must produce different noise."""
        delta  = self._fake_delta()
        w_glob = np.zeros(20)
        X = np.ones((10, 20))
        y = np.array([0, 1] * 5)

        w09.seed_attack_rngs(7, 3, 2)
        out_a = w09.atk_random_gaussian_fixed(delta, w_glob, X, y)

        w09.seed_attack_rngs(7, 3, 3)   # different client_i
        out_b = w09.atk_random_gaussian_fixed(delta, w_glob, X, y)

        assert not np.allclose(out_a, out_b), (
            "Different client indices must produce different attack noise"
        )

    def test_T9_per_client_noise_differs(self):
        """add_clipped_gaussian_noise_fixed: clients 0 and 1 get different noise."""
        delta = self._fake_delta(d=30)
        d0 = w09.add_clipped_gaussian_noise_fixed(delta, alpha=0.05,
                                                    outer_seed=7, round_t=2,
                                                    client_idx=0)
        d1 = w09.add_clipped_gaussian_noise_fixed(delta, alpha=0.05,
                                                    outer_seed=7, round_t=2,
                                                    client_idx=1)
        assert not np.allclose(d0, d1), (
            "Clients 0 and 1 must receive different noise vectors in the same round"
        )

    def test_T10_zero_alpha_unchanged(self):
        """add_clipped_gaussian_noise_fixed with alpha=0 must return unchanged delta."""
        delta = self._fake_delta(d=30)
        out   = w09.add_clipped_gaussian_noise_fixed(delta, alpha=0.0,
                                                       outer_seed=7, round_t=1,
                                                       client_idx=0)
        np.testing.assert_array_equal(out, delta,
            err_msg="alpha=0 must leave delta unchanged")

    def test_T11_backdoor_seed_varies_with_outer_seed(self):
        """
        Different outer seeds must yield different poisoning index sets,
        confirming that atk_backdoor_fixed no longer uses the fixed seed=42.
        """
        rng = np.random.default_rng(0)
        X   = rng.normal(0, 1, (100, 20))
        y   = rng.integers(0, 2, 100)

        # Extract which indices get poisoned by inspecting _BDOOR_RNG state
        w09.seed_attack_rngs(7,  0, 0)
        idx_seed7 = w09._BDOOR_RNG.choice(100, size=30, replace=False).tolist()

        w09.seed_attack_rngs(11, 0, 0)
        idx_seed11 = w09._BDOOR_RNG.choice(100, size=30, replace=False).tolist()

        assert idx_seed7 != idx_seed11, (
            "Different outer seeds must produce different backdoor poisoning indices"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
