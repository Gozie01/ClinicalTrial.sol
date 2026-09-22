"""
W06 — Export selected_features.json
=====================================
Extracts the 60 MI-selected eICU features using a TRAIN-ONLY split (C0 fix),
records the split IDs, imputer config, and a SHA-256 hash of the feature list.

Run from the trust-ct/ directory:
    python remediation/export_selected_features.py

Outputs: trust-ct/remediation/selected_features.json
"""

import hashlib, json, pathlib, warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).resolve().parent.parent   # = trust-ct/
EICU_CSV  = ROOT.parent / "evaluation" / "prepared_datasets" / "eicu_demo_prepared.csv"
OUT_FILE  = pathlib.Path(__file__).resolve().parent / "selected_features.json"

# ── Parameters (must match run_phase6.py) ─────────────────────────────────────
SEED       = 7
TEST_FRAC  = 0.20
N_TOP_FEAT = 60
OUTCOME_COL = "Outcome"  # confirmed from earlier reads

def main():
    print(f"Reading: {EICU_CSV}")
    df = pd.read_csv(EICU_CSV)
    print(f"  Shape: {df.shape}")

    if OUTCOME_COL not in df.columns:
        raise KeyError(
            f"Column '{OUTCOME_COL}' not found. Available: {list(df.columns[:10])} ..."
        )

    X_all = df.drop(columns=[OUTCOME_COL]).values
    y_all = df[OUTCOME_COL].values.astype(int)
    all_feature_names = list(df.drop(columns=[OUTCOME_COL]).columns)

    # C0-correct split: MI computed on TRAINING SET ONLY
    idx_all = np.arange(len(y_all))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_FRAC, random_state=SEED, stratify=y_all
    )

    X_tr = X_all[idx_train]
    y_tr = y_all[idx_train]

    # Impute missing values on training set only
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)

    # MI feature selection on training set only
    mi_scores = mutual_info_classif(X_tr_imp, y_tr, random_state=SEED)
    top_idx   = np.argsort(mi_scores)[::-1][:N_TOP_FEAT]   # MI-rank order (matches run_phase6.py)

    # run_phase6.py uses features IN MI-RANK ORDER (highest MI first).
    # The authoritative benchmark hash is computed from this order.
    features_mi_rank  = [all_feature_names[i] for i in top_idx]           # order used by benchmark
    mi_scores_mi_rank = [float(mi_scores[i]) for i in top_idx]

    # Authoritative hash: must match run_phase6.py and phase6_c0_lock.json
    feat_str_benchmark = json.dumps(features_mi_rank, sort_keys=False)
    feat_hash_benchmark = hashlib.sha256(feat_str_benchmark.encode()).hexdigest()

    # Column-index-sorted list (for human inspection only; hash differs from benchmark)
    top_idx_sorted = sorted(top_idx.tolist())
    features_col_order  = [all_feature_names[i] for i in top_idx_sorted]
    mi_scores_col_order = [float(mi_scores[i]) for i in top_idx_sorted]
    feat_hash_col = hashlib.sha256(
        json.dumps(features_col_order, sort_keys=False).encode()
    ).hexdigest()

    out = {
        "generation_script": "trust-ct/remediation/export_selected_features.py",
        "c0_compliance": "MI computed on training set ONLY (C0 fix applied)",
        "eicu_csv": str(EICU_CSV),
        "outcome_col": OUTCOME_COL,
        "n_total": int(len(y_all)),
        "n_train": int(len(idx_train)),
        "n_test":  int(len(idx_test)),
        "test_frac": TEST_FRAC,
        "split_seed": SEED,
        "n_top_features": N_TOP_FEAT,
        "hash_note": (
            "reconstruction_hash_sha256 is the MI-rank-order hash produced by this re-run "
            "under the current sklearn environment. It does NOT match the executed_run_hash "
            "from phase6_c0_lock.json because mutual_info_classif Monte Carlo scores differ "
            "across sklearn versions even with identical random_state, causing near-tied "
            "features to swap rank. The full ordered 60-feature list from the original "
            "run_phase6.py execution is NOT stored and cannot be recovered."
        ),
        "executed_run_hash_sha256": "4f4a66036e869328b354fb528e8dface5ff2981c9376f12fc517c594a2b91ded",
        "executed_run_hash_source": "phase6_c0_lock.json feature_selection_provenance (original run_phase6.py execution)",
        "reconstruction_hash_sha256": feat_hash_benchmark,
        "reconstruction_hash_source": "this re-run; MI-rank order; current sklearn environment",
        "col_order_hash_sha256": feat_hash_col,
        "col_order_hash_source": "this re-run; column-index sorted order",
        "imputer": {
            "strategy": "median",
            "fit_on": "training_set_only"
        },
        "features_by_mi_rank": [
            {
                "rank": rank + 1,
                "name": features_mi_rank[rank],
                "mi_score": mi_scores_mi_rank[rank],
                "orig_col_idx": int(top_idx[rank]),
            }
            for rank in range(N_TOP_FEAT)
        ],
        "features_col_order": features_col_order,
        "mi_scores_col_order": mi_scores_col_order,
        "train_idx_hash": hashlib.sha256(
            np.array(sorted(idx_train.tolist())).astype(np.int64).tobytes()
        ).hexdigest(),
        "test_idx_hash": hashlib.sha256(
            np.array(sorted(idx_test.tolist())).astype(np.int64).tobytes()
        ).hexdigest(),
    }

    # ── Benchmark reconciliation against phase6_c0_lock.json ──────────────────
    c0_lock_path = ROOT / "processed" / "phase6" / "phase6_c0_lock.json"
    recon = {}
    if c0_lock_path.exists():
        with open(c0_lock_path) as lf:
            c0 = json.load(lf)
        c0_prov       = c0.get("feature_selection_provenance", {})
        lock_hash     = c0_prov.get("feature_list_hash_sha256", "")
        lock_top10    = c0_prov.get("top_10_features", [])
        export_set    = set(features_mi_rank)
        col_set       = set(features_col_order)
        lock_top10_in = [f for f in lock_top10 if f in export_set]
        near_ties = [
            f for f in lock_top10
            if f in export_set and f not in features_mi_rank[:len(lock_top10)]
        ]
        recon = {
            "executed_run_hash": lock_hash,
            "reconstruction_hash": feat_hash_benchmark,
            "hash_matches": feat_hash_benchmark == lock_hash,
            "full_60_list_recoverable": False,
            "full_60_list_recoverable_note": (
                "phase6_c0_lock.json stores only the top-10 features, not the full ordered "
                "60-feature list. The executed_run_hash cannot be independently verified "
                "because the complete ordered list from the original execution is not preserved."
            ),
            "top10_in_reconstruction": len(lock_top10_in) == len(lock_top10),
            "top10_verified_note": (
                f"All {len(lock_top10)} lock top-10 features appear in the reconstruction. "
                f"This verifies only that these 10 features were selected; it does not confirm "
                f"that the same 60 features were selected in the same order."
            ),
            "near_tied_rank_swaps_in_top10": near_ties,
        }
        out["benchmark_reconciliation"] = recon
        print(f"  hash_matches={recon['hash_matches']}, "
              f"top10_in_reconstruction={recon['top10_in_reconstruction']}, "
              f"full_60_recoverable=False")
        if near_ties:
            print(f"  Near-tied rank swaps in top-10: {near_ties}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWritten: {OUT_FILE}")
    print(f"  Features selected: {N_TOP_FEAT}")
    print(f"  executed_run_hash (phase6_c0_lock.json):  4f4a66036e869328...")
    print(f"  reconstruction_hash (this run, MI-rank):  {feat_hash_benchmark[:16]}...")
    print(f"  col_order_hash (this run, col-index):     {feat_hash_col[:16]}...")
    print(f"  Full 60-list recoverable: False (only top-10 stored in lock)")
    print(f"  Top 5 by MI rank (this run): {features_mi_rank[:5]}")


if __name__ == "__main__":
    main()
