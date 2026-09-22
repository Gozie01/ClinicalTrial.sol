"""
TRUST-CT Phase 3 Complete — All evaluation modes.

Four evaluation modes per held-out client k:
  A. Local-only      : train + 5-fold CV within client k only
  B. Central-LOCO    : pool 22 training clients, test on k (matched architecture)
  C. FL-LOCO         : FedAvg / FedProx / SCAFFOLD / FedAdam on 22 clients, test on k
  D. Long-tail ext.  : train on all 23 primary clients, test on "other" cohort

Paired NI quantity:
  D_{k,s} = AUROC_FL_{k,s} - AUROC_CentralLOCO_{k,s}
  NI declared iff LCB_95pct(D) > -0.02

Primary metric: PR-AUC for Y_risk = 1 - Y_adh  (nonadherence targeting)
Secondary:      PR-AUC for Y_adh, AUROC (same for both directions)

Run:
    cd "C:/Users/Gozie/Desktop/Blockchain in Clinical Trial"
    python trust-ct/run_phase3_complete.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from federated.fedavg    import LogisticClient, run_federated
from federated.scaffold  import run_scaffold, ScaffoldClient
from federated.fedadam   import run_fedadam
from evaluation.metrics  import (
    full_metrics, full_metrics_both_directions,
    noninferiority_test, boot_ci, hierarchical_boot_ci,
    format_ci, format_table_row, auroc, pr_auc,
    site_metrics,
)

PROC    = _HERE / "processed" / "cimas"
OUT_DIR = PROC / "results_phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARK_PATH = PROC / "cimas_htn_landmark_6m.parquet"

SEEDS    = [7, 11, 19, 23, 37]
CV_FOLDS = 5
N_ROUNDS = 30
FED_LR   = 0.02
N_LOCAL  = 10
FEDPROX_MU = 0.1   # tuned: mu=0.01 invisible at n_epochs=5/lr=0.05;
                    # mu=0.1 produces measurable proximal correction at n_epochs=10/lr=0.02

FEATURE_COLS = [
    "age", "sex_female",
    "scheme_type_ord", "cover_type_bin", "annual_contrib_log",
    "n_refill_months_obs", "refill_recency_days",
    "early_months", "late_months", "has_2m_gap", "first_claim_month",
    "n_claims", "n_claim_dates", "n_products", "n_providers_vis",
    "obs_amount", "obs_units", "amount_per_claim", "units_per_claim",
    "n_networks",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    lm = pd.read_parquet(LANDMARK_PATH)
    avail = [c for c in FEATURE_COLS if c in lm.columns]
    X_df  = lm[avail].copy()
    X_np  = X_df.values.astype(float)
    y_adh = lm["Y_adh"].values.astype(int)
    y_risk = 1 - y_adh
    clients = lm["client"].values
    pids    = lm["pid"].values
    return lm, X_np, y_adh, y_risk, clients, pids, avail


# ─────────────────────────────────────────────────────────────────────────────
# Architecture: centralised logistic regression (matched to FL)
# ─────────────────────────────────────────────────────────────────────────────

def _make_central_pipe(seed: int = 7) -> Pipeline:
    pos_weight_placeholder = 1.0   # recalculated per-fit
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=seed,
        )),
    ])


def _fit_central(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 7) -> Pipeline:
    pipe = _make_central_pipe(seed)
    pipe.fit(X_tr, y_tr)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — Local-only: within-client 5-fold CV × 5 seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_local_only(
    X_np: np.ndarray,
    y_adh: np.ndarray,
    client_labels: np.ndarray,
    primary_clients: list,
) -> list:
    """
    For each primary client: 5-fold CV × 5 seeds.
    Returns list of per-(client, seed, fold) result dicts.
    """
    results = []
    for cid in primary_clients:
        mask = client_labels == cid
        Xc, yc = X_np[mask], y_adh[mask]
        if yc.sum() == 0 or yc.sum() == len(yc) or len(yc) < 20:
            continue
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
            # Accumulate out-of-fold predictions for bootstrapping
            oof_true, oof_prob = [], []
            for tr_idx, te_idx in cv.split(Xc, yc):
                if yc[tr_idx].sum() == 0 or yc[te_idx].sum() == 0:
                    continue
                pipe = _fit_central(Xc[tr_idx], yc[tr_idx], seed=seed)
                yp   = pipe.predict_proba(Xc[te_idx])[:, 1]
                oof_true.extend(yc[te_idx].tolist())
                oof_prob.extend(yp.tolist())
            if not oof_true or sum(oof_true) == 0:
                continue
            yt = np.array(oof_true, dtype=int)
            yp = np.array(oof_prob, dtype=float)
            m = full_metrics_both_directions(yt, yp, label=f"LocalOnly_{cid}_s{seed}")
            m.update({"mode": "local_only", "test_client": cid, "seed": seed})
            results.append(m)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode B — Centralized-LOCO: pool 22 clients, test on k
# ─────────────────────────────────────────────────────────────────────────────

def run_central_loco(
    X_np: np.ndarray,
    y_adh: np.ndarray,
    client_labels: np.ndarray,
    primary_clients: list,
) -> list:
    results = []
    for cid in primary_clients:
        test_mask  = client_labels == cid
        train_mask = (client_labels != cid) & (client_labels != "other")
        Xte, yte   = X_np[test_mask], y_adh[test_mask]
        Xtr, ytr   = X_np[train_mask], y_adh[train_mask]
        if yte.sum() == 0 or yte.sum() == len(yte) or ytr.sum() == 0:
            continue
        for seed in SEEDS:
            try:
                pipe = _fit_central(Xtr, ytr, seed=seed)
                yp   = pipe.predict_proba(Xte)[:, 1]
                m    = full_metrics_both_directions(yte, yp, label=f"CentralLOCO_{cid}_s{seed}")
                m.update({"mode": "central_loco", "test_client": cid, "seed": seed})
                results.append(m)
            except Exception as exc:
                print(f"  CentralLOCO {cid} seed={seed}: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode C — FL-LOCO: federate 22 clients, test on k
# ─────────────────────────────────────────────────────────────────────────────

def _build_client_dict(
    X_np: np.ndarray,
    y: np.ndarray,
    client_labels: np.ndarray,
    exclude_client: str,
    primary_clients: list,
) -> dict:
    cd = {}
    for tc in primary_clients:
        if tc == exclude_client:
            continue
        mask = client_labels == tc
        if mask.sum() < 10 or y[mask].sum() == 0:
            continue
        cd[tc] = (X_np[mask], y[mask])
    return cd


FL_METHODS = {
    "FedAvg":       dict(kind="fedavg",   mu=0.0,           equal_weight=False),
    "FedProx":      dict(kind="fedavg",   mu=FEDPROX_MU,    equal_weight=False),
    "FedAvg-equal": dict(kind="fedavg",   mu=0.0,           equal_weight=True),
    "SCAFFOLD":     dict(kind="scaffold"),
    "FedAdam":      dict(kind="fedadam"),
}


def _run_one_fl(
    method: str,
    cfg: dict,
    client_data: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    test_client: str,
) -> dict:
    kind = cfg["kind"]
    if kind == "fedavg":
        clients, _ = run_federated(
            client_data, n_rounds=N_ROUNDS,
            mu=cfg.get("mu", 0.0),
            equal_weight=cfg.get("equal_weight", False),
            lr=FED_LR, n_local_epochs=N_LOCAL, seed=seed,
        )
    elif kind == "scaffold":
        clients = run_scaffold(
            client_data, n_rounds=N_ROUNDS,
            lr=FED_LR, n_local_steps=N_LOCAL, seed=seed,
        )
    elif kind == "fedadam":
        clients, _ = run_fedadam(
            client_data, n_rounds=N_ROUNDS,
            lr=FED_LR, n_local_epochs=N_LOCAL, seed=seed,
        )
    else:
        raise ValueError(kind)

    # All clients share the same global weights — use first
    yp = clients[0].predict_proba(X_test)[:, 1]
    m  = full_metrics_both_directions(y_test, yp, label=f"{method}_{test_client}_s{seed}")
    m.update({"mode": f"fl_loco_{method}", "test_client": test_client,
               "seed": seed, "fl_method": method})
    return m


def run_fl_loco(
    X_np: np.ndarray,
    y_adh: np.ndarray,
    client_labels: np.ndarray,
    primary_clients: list,
) -> list:
    results = []
    n_total = len(primary_clients) * len(SEEDS) * len(FL_METHODS)
    done = 0
    for cid in primary_clients:
        test_mask = client_labels == cid
        Xte, yte  = X_np[test_mask], y_adh[test_mask]
        if yte.sum() == 0 or yte.sum() == len(yte):
            continue
        client_data = _build_client_dict(X_np, y_adh, client_labels, cid, primary_clients)
        if len(client_data) < 3:
            continue
        for seed in SEEDS:
            for method, cfg in FL_METHODS.items():
                try:
                    m = _run_one_fl(method, cfg, client_data, Xte, yte, seed, cid)
                    results.append(m)
                except Exception as exc:
                    print(f"  {method} {cid} seed={seed}: {exc}")
                done += 1
                if done % 50 == 0:
                    print(f"  FL-LOCO progress: {done}/{n_total}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode D — Long-tail external: train on 23 primary, test on "other"
# ─────────────────────────────────────────────────────────────────────────────

def run_long_tail(
    X_np: np.ndarray,
    y_adh: np.ndarray,
    client_labels: np.ndarray,
    primary_clients: list,
) -> list:
    """
    Train on all 23 primary clients; test on the "other" cohort (22.7% of data).
    The long-tail "other" cohort is NEVER merged into a named client.
    """
    train_mask = np.isin(client_labels, primary_clients)
    other_mask = client_labels == "other"
    Xtr, ytr   = X_np[train_mask], y_adh[train_mask]
    Xte, yte   = X_np[other_mask], y_adh[other_mask]

    if yte.sum() == 0 or yte.sum() == len(yte) or ytr.sum() == 0:
        print("  Long-tail: insufficient outcome variation, skipping.")
        return []

    results = []

    # Centralized logistic (matched architecture)
    for seed in SEEDS:
        try:
            pipe = _fit_central(Xtr, ytr, seed=seed)
            yp   = pipe.predict_proba(Xte)[:, 1]
            m    = full_metrics_both_directions(yte, yp, label=f"Central_LongTail_s{seed}")
            m.update({"mode": "long_tail_central", "test_client": "other", "seed": seed})
            results.append(m)
        except Exception as exc:
            print(f"  LongTail Central seed={seed}: {exc}")

    # FedAvg (best FL model on full 23 clients)
    client_data = {}
    for tc in primary_clients:
        mask = client_labels == tc
        if mask.sum() >= 10 and y_adh[mask].sum() > 0:
            client_data[tc] = (X_np[mask], y_adh[mask])

    for seed in SEEDS:
        try:
            clients, _ = run_federated(
                client_data, n_rounds=N_ROUNDS,
                mu=0.0, lr=FED_LR, n_local_epochs=N_LOCAL, seed=seed,
            )
            yp = clients[0].predict_proba(Xte)[:, 1]
            m  = full_metrics_both_directions(yte, yp, label=f"FedAvg_LongTail_s{seed}")
            m.update({"mode": "long_tail_fedavg", "test_client": "other", "seed": seed})
            results.append(m)
        except Exception as exc:
            print(f"  LongTail FedAvg seed={seed}: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FedProx diagnostic — verify proximal term is active at mu=0.1
# ─────────────────────────────────────────────────────────────────────────────

def run_fedprox_diagnostic(
    X_np: np.ndarray,
    y_adh: np.ndarray,
    client_labels: np.ndarray,
    primary_clients: list,
    test_client: str = None,
) -> None:
    """
    Run 3 rounds of FedAvg vs FedProx with verbose diagnostics on a single fold.
    Prints proximal_penalty, weight_distance, grad_norm per round for mu in grid.
    """
    if test_client is None:
        test_client = primary_clients[0]

    client_data = _build_client_dict(
        X_np, y_adh, client_labels, test_client, primary_clients[:8]
    )
    if len(client_data) < 2:
        print("  Diagnostic: not enough clients, skipping.")
        return

    print(f"\n--- FedProx Diagnostic (test_client={test_client}, 5 rounds, seed=7) ---")
    for mu in [0.0, 0.01, 0.1, 1.0]:
        label = "FedAvg" if mu == 0 else f"FedProx(mu={mu})"
        print(f"\n  {label}:")
        clients, rdiags = run_federated(
            client_data, n_rounds=5, mu=mu,
            lr=FED_LR, n_local_epochs=N_LOCAL, seed=7,
            verbose_diag=True,
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and Phase 3 gate table
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_mode(records: list, mode_filter: str, metric: str = "auroc") -> dict:
    """Pool all records matching mode_filter, return mean [95% CI]."""
    recs = [r for r in records if r.get("mode", "").startswith(mode_filter)]
    if not recs:
        return {"n": 0, "mean": np.nan, "ci": "N/A"}
    raw  = [r.get(metric, None) for r in recs]
    vals = []
    for v in raw:
        try:
            fv = float(v)
            if not np.isnan(fv):
                vals.append(fv)
        except (TypeError, ValueError):
            pass
    if not vals:
        return {"n": 0, "mean": np.nan, "ci": "N/A"}
    arr  = np.array(vals)
    rng  = np.random.default_rng(0)
    boot = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(2000)])
    lo   = float(np.percentile(boot, 2.5))
    hi   = float(np.percentile(boot, 97.5))
    return {"n": len(vals), "mean": arr.mean(), "ci": format_ci(arr.mean(), lo, hi)}


def build_phase3_table(all_results: list, lgbm_row: dict = None) -> pd.DataFrame:
    """
    Build the Phase 3 completion gate table.
    Metrics reported for primary endpoint Y_risk (nonadherence).
    """
    def _get(mode_str, metric):
        return aggregate_mode(all_results, mode_str, metric)

    rows = []

    # Local-only
    r = {}
    for met, col in [("risk_pr_auc","PR-AUC"), ("risk_pr_lift","PR-lift"),
                     ("auroc","AUROC"), ("risk_mcc","MCC"),
                     ("risk_brier","Brier"), ("risk_ece","ECE")]:
        agg = _get("local_only", met)
        r[col] = agg["ci"] if agg["n"] > 0 else "—"
    rows.append({"Method": "Local-only (logistic)", "Evaluation": "Within-client CV", **r})

    # Central-LOCO
    r = {}
    for met, col in [("risk_pr_auc","PR-AUC"), ("risk_pr_lift","PR-lift"),
                     ("auroc","AUROC"), ("risk_mcc","MCC"),
                     ("risk_brier","Brier"), ("risk_ece","ECE")]:
        agg = _get("central_loco", met)
        r[col] = agg["ci"] if agg["n"] > 0 else "—"
    rows.append({"Method": "Centralized logistic", "Evaluation": "LOCO", **r})

    # FL methods
    for method in FL_METHODS:
        r = {}
        for met, col in [("risk_pr_auc","PR-AUC"), ("risk_pr_lift","PR-lift"),
                         ("auroc","AUROC"), ("risk_mcc","MCC"),
                         ("risk_brier","Brier"), ("risk_ece","ECE")]:
            agg = _get(f"fl_loco_{method}", met)
            r[col] = agg["ci"] if agg["n"] > 0 else "—"
        rows.append({"Method": method, "Evaluation": "LOCO", **r})

    # LightGBM (already run in Phase 2 — patient-level CV)
    if lgbm_row:
        rows.append(lgbm_row)
    else:
        rows.append({
            "Method": "LightGBM", "Evaluation": "Patient-level CV",
            "PR-AUC": "see run_phase2_cimas.py",
            "PR-lift": "—", "AUROC": "0.811 [95% CI TBD]",
            "MCC": "—", "Brier": "—", "ECE": "—",
        })

    # Long-tail external
    r = {}
    for met, col in [("risk_pr_auc","PR-AUC"), ("risk_pr_lift","PR-lift"),
                     ("auroc","AUROC"), ("risk_mcc","MCC"),
                     ("risk_brier","Brier"), ("risk_ece","ECE")]:
        agg = _get("long_tail_central", met)
        r[col] = agg["ci"] if agg["n"] > 0 else "—"
    rows.append({"Method": "Best global (logistic)", "Evaluation": "Long-tail external", **r})

    return pd.DataFrame(rows)


def paired_ni_test(central_records: list, fl_records: list, fl_method: str) -> dict:
    """
    D_{k,s} = AUROC_FL_{k,s} - AUROC_CentralLOCO_{k,s}  matched by (test_client, seed).
    """
    central_map = {
        (r["test_client"], r["seed"]): r["auroc"]
        for r in central_records if "test_client" in r and "seed" in r
    }
    fl_map = {
        (r["test_client"], r["seed"]): r["auroc"]
        for r in fl_records
        if r.get("fl_method") == fl_method and "test_client" in r
    }
    pairs = []
    for key, auroc_fl in fl_map.items():
        if key in central_map:
            pairs.append(auroc_fl - central_map[key])
    if not pairs:
        return {"noninferior": None, "n_pairs": 0, "D_observed": np.nan}
    return noninferiority_test(np.array(pairs), margin=-0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== TRUST-CT Phase 3 Complete ===\n")

    if not LANDMARK_PATH.exists():
        raise FileNotFoundError(f"Run cohort/cimas_landmark.py first.")

    lm, X_np, y_adh, y_risk, clients, pids, feat_cols = load_data()
    primary_clients = sorted([c for c in np.unique(clients) if c != "other"])
    other_n = (clients == "other").sum()

    print(f"N={len(y_adh):,}  Y_adh={y_adh.mean():.1%}  Y_risk={y_risk.mean():.1%}")
    print(f"Primary clients: {len(primary_clients)}  Long-tail (other): {other_n:,} ({other_n/len(y_adh):.1%})")
    print(f"Features: {len(feat_cols)}\n")

    # FedProx diagnostic (5 rounds on small subset)
    run_fedprox_diagnostic(X_np, y_adh, clients, primary_clients)

    all_results = []

    # ── Mode A: Local-only ───────────────────────────────────────────────────
    print("Mode A: Local-only (within-client 5-fold CV x 5 seeds)...")
    local_res = run_local_only(X_np, y_adh, clients, primary_clients)
    all_results.extend(local_res)
    print(f"  Records: {len(local_res)}")

    # ── Mode B: Central-LOCO ─────────────────────────────────────────────────
    print("Mode B: Centralized-LOCO (22 clients -> test on k)...")
    central_res = run_central_loco(X_np, y_adh, clients, primary_clients)
    all_results.extend(central_res)
    print(f"  Records: {len(central_res)}")

    # ── Mode C: FL-LOCO ──────────────────────────────────────────────────────
    print(f"Mode C: FL-LOCO ({len(FL_METHODS)} methods x {len(primary_clients)} clients x {len(SEEDS)} seeds)...")
    fl_res = run_fl_loco(X_np, y_adh, clients, primary_clients)
    all_results.extend(fl_res)
    fl_df  = pd.DataFrame(fl_res) if fl_res else pd.DataFrame()
    print(f"  Records: {len(fl_res)}")

    # ── Mode D: Long-tail external ───────────────────────────────────────────
    print("Mode D: Long-tail external (train on 23 primary, test on 'other')...")
    lt_res = run_long_tail(X_np, y_adh, clients, primary_clients)
    all_results.extend(lt_res)
    print(f"  Records: {len(lt_res)}")

    # ── Save raw results ─────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_results)
    all_df.to_csv(OUT_DIR / "phase3_all_results.csv", index=False)

    # ── Noninferiority tests ─────────────────────────────────────────────────
    print("\n--- Noninferiority Tests (NI margin = -0.02 AUROC) ---")
    ni_rows = []
    fl_recs = [r for r in all_results if r.get("mode", "").startswith("fl_loco")]
    cent_recs = [r for r in all_results if r.get("mode") == "central_loco"]
    for method in FL_METHODS:
        ni = paired_ni_test(cent_recs, fl_recs, method)
        status = "NONINFERIOR" if ni.get("noninferior") else "INFERIOR"
        print(f"  {method:14s}: D_obs={ni.get('D_observed', np.nan):.4f}  "
              f"LCB={ni.get('ci_low_95pct', np.nan):.4f}  "
              f"n_pairs={ni.get('n_pairs',0)}  -> {status}")
        ni["method"] = method
        ni_rows.append(ni)

    pd.DataFrame(ni_rows).to_csv(OUT_DIR / "ni_test_results.csv", index=False)

    # ── Phase 3 gate table (primary endpoint: Y_risk) ─────────────────────────
    print("\n--- Phase 3 Completion Gate Table (Y_risk = nonadherence) ---")
    gate_df = build_phase3_table(all_results)
    print(gate_df.to_string(index=False))
    gate_df.to_csv(OUT_DIR / "phase3_gate_table.csv", index=False)

    # ── Site-level breakdown (FedAvg LOCO) ────────────────────────────────────
    print("\n--- Site-level Summary (FedAvg LOCO, pooled across seeds) ---")
    fedavg_recs = [r for r in all_results if r.get("fl_method") == "FedAvg"]
    if fedavg_recs:
        site_summary = (
            pd.DataFrame(fedavg_recs)
              .groupby("test_client")[["auroc", "risk_pr_auc"]]
              .agg(["mean", "std"])
              .round(4)
        )
        print(site_summary.to_string())
        site_summary.to_csv(OUT_DIR / "fedavg_site_breakdown.csv")

    # ── Additional: FedProx vs FedAvg difference check ────────────────────────
    print("\n--- FedProx vs FedAvg AUROC difference ---")
    for method in ["FedAvg", "FedProx"]:
        recs = [r for r in all_results if r.get("fl_method") == method]
        if recs:
            vals = [r["auroc"] for r in recs if not np.isnan(r.get("auroc", np.nan))]
            if vals:
                print(f"  {method:14s}: mean AUROC={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    print(f"\n=== Phase 3 complete. Results: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
