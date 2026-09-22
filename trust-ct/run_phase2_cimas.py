"""
TRUST-CT Phase 3 — Cimas HTN Centralized and Federated Baselines.

Loads the landmark cohort produced by cohort/cimas_landmark.py and runs:
  1. Centralized baselines: ElasticNet, LightGBM, RandomForest
     (repeated 5-fold CV, 5 seeds, nested hyperparameter tuning)
  2. Federated: FedAvg, FedProx, Equal-weight FedAvg ablation
     (23 primary providers as clients, same 5 seeds)
  3. Noninferiority test: AUROC_FL vs AUROC_central
  4. Per-client breakdown table

Primary metric: PR-AUC
Secondary:      AUROC, Brier, calibration slope

Run:
    cd "C:/Users/Gozie/Desktop/Blockchain in Clinical Trial"
    python trust-ct/run_phase2_cimas.py
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE   = pathlib.Path(__file__).resolve().parent
PROC    = _HERE / "processed" / "cimas"
OUT_DIR = _HERE / "processed" / "cimas" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_PATH = PROC / "cimas_htn_landmark_6m.parquet"

# ── Import modules from trust-ct ───────────────────────────────────────────
import sys
sys.path.insert(0, str(_HERE))

from models.tabular import fit_elasticnet, fit_lgbm, fit_rf, FeatureEngineer
from federated.fedavg import run_federated
from evaluation.metrics import full_metrics, noninferiority_test, metrics_table

SEEDS    = [7, 11, 19, 23, 37]
CV_FOLDS = 5

FEATURE_COLS = [
    "age", "sex_female",
    "scheme_type_ord", "cover_type_bin", "annual_contrib_log",
    "n_refill_months_obs", "refill_recency_days",
    "early_months", "late_months", "has_2m_gap", "first_claim_month",
    "n_claims", "n_claim_dates", "n_products", "n_providers_vis",
    "obs_amount", "obs_units", "amount_per_claim", "units_per_claim",
    "n_networks",
]


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    lm = pd.read_parquet(LANDMARK_PATH)
    available = [c for c in FEATURE_COLS if c in lm.columns]
    X_df  = lm[available].copy()
    y     = lm["Y_adh"].values.astype(int)
    client_labels = lm["client"].values
    return lm, X_df, y, client_labels


def run_centralized(X_df: pd.DataFrame, y: np.ndarray) -> list[dict]:
    """Repeated 5-fold CV over 5 seeds for EN, GBM, RF."""
    results = []
    X_np = X_df.values.astype(float)

    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_np, y)):
            X_tr_df = X_df.iloc[tr_idx].reset_index(drop=True)
            X_te_df = X_df.iloc[te_idx].reset_index(drop=True)
            y_tr, y_te = y[tr_idx], y[te_idx]

            # ElasticNet
            try:
                pipe_en, feat_names, fe_en = fit_elasticnet(X_tr_df, y_tr, seed=seed)
                X_te_eng = fe_en.transform(X_te_df).values.astype(float)
                yp_en = pipe_en.predict_proba(X_te_eng)[:, 1]
                m = full_metrics(y_te, yp_en, label="ElasticNet")
                m.update({"seed": seed, "fold": fold_i})
                results.append(m)
            except Exception as exc:
                print(f"  EN seed={seed} fold={fold_i} failed: {exc}")

            # LightGBM
            try:
                lgb_m, lgb_imp, feat_names_lgb, fe_lgb = fit_lgbm(X_tr_df, y_tr, seed=seed)
                X_te_eng_lgb = fe_lgb.transform(X_te_df).values.astype(float)
                X_te_imp = lgb_imp.transform(X_te_eng_lgb)
                yp_lgb = lgb_m.predict_proba(X_te_imp)[:, 1]
                m = full_metrics(y_te, yp_lgb, label="LightGBM")
                m.update({"seed": seed, "fold": fold_i})
                results.append(m)
            except Exception as exc:
                print(f"  LGB seed={seed} fold={fold_i} failed: {exc}")

            # Random Forest
            try:
                rf_m, rf_imp, _, fe_rf = fit_rf(X_tr_df, y_tr, seed=seed)
                X_te_eng_rf = fe_rf.transform(X_te_df).values.astype(float)
                X_te_imp_rf = rf_imp.transform(X_te_eng_rf)
                yp_rf = rf_m.predict_proba(X_te_imp_rf)[:, 1]
                m = full_metrics(y_te, yp_rf, label="RandomForest")
                m.update({"seed": seed, "fold": fold_i})
                results.append(m)
            except Exception as exc:
                print(f"  RF seed={seed} fold={fold_i} failed: {exc}")

    return results


def run_fl(
    X_df: pd.DataFrame,
    y: np.ndarray,
    client_labels: np.ndarray,
) -> list[dict]:
    """
    Federated baselines: FedAvg, FedProx, Equal-weight FedAvg.
    Train on all-but-test-client; evaluate on test client.
    Repeated over 5 seeds.
    """
    results = []
    X_np = X_df.values.astype(float)
    named_clients = [c for c in np.unique(client_labels) if c != "other"]

    fl_configs = [
        ("FedAvg",       dict(mu=0.0, equal_weight=False)),
        ("FedProx",      dict(mu=0.01, equal_weight=False)),
        ("FedAvg-equal", dict(mu=0.0, equal_weight=True)),
    ]

    for seed in SEEDS:
        # Leave-one-client-out evaluation
        for test_client in named_clients:
            test_mask  = client_labels == test_client
            train_mask = (client_labels != test_client) & (client_labels != "other")

            X_te, y_te = X_np[test_mask], y[test_mask]
            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue  # skip if test client has only one class

            # Build per-client training data
            train_clients = [c for c in named_clients if c != test_client]
            client_data = {}
            for tc in train_clients:
                mask = client_labels == tc
                if mask.sum() < 10 or y[mask].sum() == 0:
                    continue
                client_data[tc] = (X_np[mask], y[mask])

            if len(client_data) < 2:
                continue

            for fl_name, fl_kwargs in fl_configs:
                try:
                    clients = run_federated(
                        client_data, n_rounds=20, lr=0.05,
                        n_local_epochs=5, seed=seed, **fl_kwargs
                    )
                    # Use the first client's weights for prediction
                    # (all clients have global weights after federation)
                    yp = clients[0].predict_proba(X_te)[:, 1]
                    m = full_metrics(y_te, yp, label=fl_name)
                    m.update({"seed": seed, "test_client": test_client})
                    results.append(m)
                except Exception as exc:
                    print(f"  {fl_name} seed={seed} client={test_client} failed: {exc}")

    return results


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("label")[["pr_auc", "auroc", "brier", "cal_slope"]]
          .agg(["mean", "std"])
          .round(4)
    )


def main():
    print("=== TRUST-CT Phase 3: Cimas Centralized + FL Baselines ===\n")

    if not LANDMARK_PATH.exists():
        raise FileNotFoundError(
            f"Run cohort/cimas_landmark.py first to produce {LANDMARK_PATH}"
        )

    lm, X_df, y, client_labels = load_data()
    print(f"Cohort: N={len(y):,}, Y_adh prevalence={y.mean():.1%}")
    print(f"Features: {X_df.shape[1]} | Clients: {pd.Series(client_labels).nunique()}")
    print()

    # --- Centralized ---
    print("Running centralized baselines (5 seeds x 5 folds)...")
    central_results = run_centralized(X_df, y)
    central_df = pd.DataFrame(central_results)
    central_df.to_csv(OUT_DIR / "central_results.csv", index=False)
    print("Centralized summary:")
    print(summarise(central_df))
    print()

    # --- Federated ---
    print("Running federated baselines (5 seeds x LOCO evaluation)...")
    fl_results = run_fl(X_df, y, client_labels)
    fl_df = pd.DataFrame(fl_results)
    if len(fl_df) > 0:
        fl_df.to_csv(OUT_DIR / "fl_results.csv", index=False)
        print("FL summary:")
        print(summarise(fl_df))
        print()

        # Noninferiority test: best FL vs best central
        # Use pooled predictions from FedAvg (approximate — full NI test requires
        # matched per-observation predictions; use aggregate AUROC here as proxy)
        best_central_auroc = central_df[central_df["label"] == "LightGBM"]["auroc"].mean()
        best_fl_auroc      = fl_df[fl_df["label"] == "FedAvg"]["auroc"].mean()
        delta = best_fl_auroc - best_central_auroc
        print(f"Noninferiority proxy: AUROC_FL={best_fl_auroc:.4f}, "
              f"AUROC_central={best_central_auroc:.4f}, "
              f"delta={delta:.4f} (NI margin=-0.02)")
        print(f"  -> {'NONINFERIOR' if delta > -0.02 else 'INFERIOR'} (aggregate AUROC proxy)")
    else:
        print("No FL results produced (insufficient clients or data).")

    print("\n=== Phase 3 complete. Results in processed/cimas/results/ ===")


if __name__ == "__main__":
    main()
