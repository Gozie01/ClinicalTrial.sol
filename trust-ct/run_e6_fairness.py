"""
E6 - Warm/Cold Federated Fairness De-confound  (Cimas, K=23 providers)

Scientific question:  Does FedAvg systematically benefit smaller providers
more than local-only training?  Is that benefit driven by continued gradient
sharing, or does it collapse once providers receive the shared initialisation
and train independently?

Four conditions x 5 seeds x 60 rounds:
  cold_fed   - FedAvg from zero initialisation (all 23 providers sharing)
  cold_local - Each provider trains independently from zero (no sharing ever)
  warm_fed   - FedAvg starting from the cold_fed round-30 checkpoint
  warm_local - Local-only starting from the cold_fed round-30 checkpoint

The warm_local/warm_fed contrast de-confounds warm initialisation from
continued federation:
  warm_local ~= warm_fed  ->  benefit was initialisation only (one-shot sharing)
  warm_fed   >  warm_local ->  continued federation provides ongoing benefit

Preprocessing:  per-client local imputer + scaler fitted on training rows only,
consistent with LogisticClient's internal preprocessing.

Fairness metrics (across K=23 providers at each round):
  mean AUROC, min AUROC (worst-off), AUROC variance, AUROC IQR.

Outputs:
  processed/e6_fairness/
    e6_provider_round_results.csv   - per-provider x round x condition x seed
    e6_summary_round_results.csv    - cross-provider summary per round
    e6_metadata.json                - config, data hash, feature list

Re-run behaviour:
  If e6_provider_round_results.csv already exists, the long experiment loop is
  skipped and only the summary/metadata are recomputed from the existing CSV.
  Delete that file to force a full re-run.
"""

import json
import hashlib
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from federated.fedavg import LogisticClient, fedavg_round

# -- Configuration -------------------------------------------------------------
SEEDS           = [7, 11, 19, 23, 37]
T_COLD          = 60      # total rounds for cold conditions
T_SNAPSHOT      = 30      # round at which to snapshot cold_fed weights for warm init
N_LOCAL_EPOCHS  = 5       # inner epochs per round (consistent with Phase 5)
LR              = 0.02
MIN_PROV_SIZE   = 100     # minimum training participants to include a provider
TEST_FRAC       = 0.20    # within-provider stratified test split
LONG_TAIL_LABEL = "other" # excluded from primary cohort

PROC  = HERE / "processed" / "cimas"
OUT   = HERE / "processed" / "e6_fairness"
DATA  = PROC / "cimas_htn_landmark_6m.parquet"

OUT.mkdir(parents=True, exist_ok=True)

# -- Feature selection ---------------------------------------------------------
EXCLUDE_COLS = {
    "pid", "assigned_provider", "client", "client_sens",
    "Y_adh", "out_refill_months",
}

def load_primary_cohort():
    df = pd.read_parquet(DATA)
    df = df[df["client"] != LONG_TAIL_LABEL].copy()
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols
                    if df[c].dtype.kind in ("f", "i", "u")]
    return df, feature_cols

def provider_split(df, feature_cols, seed):
    """Within-provider stratified 80/20 train/test split."""
    providers = {}
    rng = np.random.default_rng(seed)
    for prov, grp in df.groupby("client"):
        X = grp[feature_cols].values.astype(float)
        y = grp["Y_adh"].values.astype(int)
        if len(y) < MIN_PROV_SIZE or y.sum() == 0 or (y == 0).sum() == 0:
            continue
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                     random_state=int(rng.integers(1e6)))
        tr_idx, te_idx = next(sss.split(X, y))
        providers[prov] = {
            "X_train": X[tr_idx], "y_train": y[tr_idx],
            "X_test":  X[te_idx], "y_test":  y[te_idx],
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "prevalence": float(y[te_idx].mean()),
        }
    return providers

def make_clients(providers, seed):
    return {
        prov: LogisticClient(
            client_id=prov,
            X=v["X_train"], y=v["y_train"],
            lr=LR, n_local_epochs=N_LOCAL_EPOCHS, seed=seed,
        )
        for prov, v in providers.items()
    }

def client_auroc(client, X_test, y_test):
    w      = client.get_weights()
    X      = client._preprocess(X_test)
    logits = X @ w[:-1] + w[-1]
    proba  = 1.0 / (1.0 + np.exp(-logits))
    if y_test.sum() == 0 or (y_test == 0).sum() == 0:
        return np.nan
    return float(roc_auc_score(y_test, proba))

def record_round(rows, seed, condition, rnd, clients, providers):
    for prov, c in clients.items():
        v = providers[prov]
        rows.append({
            "seed": seed, "condition": condition, "round": rnd,
            "client_id": prov,
            "n_train": v["n_train"], "n_test": v["n_test"],
            "prevalence": v["prevalence"],
            "auroc": client_auroc(c, v["X_test"], v["y_test"]),
        })

# -- Summary computation -------------------------------------------------------
def compute_summary(prov_df):
    summ_rows = []
    for (seed, cond, rnd), grp in prov_df.groupby(["seed", "condition", "round"]):
        aucs = grp["auroc"].dropna().values
        if len(aucs) == 0:
            continue
        q25, q75 = np.percentile(aucs, [25, 75])
        summ_rows.append({
            "seed": seed, "condition": cond, "round": rnd,
            "n_providers": len(aucs),
            "mean_auroc":  float(np.mean(aucs)),
            "min_auroc":   float(np.min(aucs)),
            "max_auroc":   float(np.max(aucs)),
            "var_auroc":   float(np.var(aucs, ddof=1)),
            "iqr_auroc":   float(q75 - q25),
            "q25_auroc":   float(q25),
            "q75_auroc":   float(q75),
        })
    return pd.DataFrame(summ_rows)

# -- Main experiment -----------------------------------------------------------
def run_e6():
    df, feature_cols = load_primary_cohort()

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()

    provider_csv = OUT / "e6_provider_round_results.csv"

    if provider_csv.exists():
        kb = provider_csv.stat().st_size // 1024
        print(f"[resume] Loading existing provider CSV ({kb} KB) -- skipping experiment loop.")
        prov_df = pd.read_csv(provider_csv)
        print(f"  {len(prov_df)} rows loaded.")
    else:
        all_rows = []
        for seed in SEEDS:
            print(f"\n=== Seed {seed} ===")
            providers = provider_split(df, feature_cols, seed)
            K = len(providers)
            total_n = sum(v["n_train"] for v in providers.values())
            print(f"  Providers included: {K}  (total training N={total_n})")

            # Condition 1: cold_fed
            print(f"  cold_fed: {T_COLD} rounds ...", end="", flush=True)
            clients_cf = make_clients(providers, seed)
            snapshot_weights = None
            for rnd in range(1, T_COLD + 1):
                fedavg_round(list(clients_cf.values()), mu=0.0)
                record_round(all_rows, seed, "cold_fed", rnd, clients_cf, providers)
                if rnd == T_SNAPSHOT:
                    snapshot_weights = list(clients_cf.values())[0].get_weights().copy()
            print(" done")

            # Condition 2: cold_local
            print(f"  cold_local: {T_COLD} rounds ...", end="", flush=True)
            clients_cl = make_clients(providers, seed)
            for rnd in range(1, T_COLD + 1):
                for c in clients_cl.values():
                    c.local_update(mu=0.0)
                record_round(all_rows, seed, "cold_local", rnd, clients_cl, providers)
            print(" done")

            # Condition 3: warm_fed (FedAvg from cold_fed round-30 checkpoint)
            print(f"  warm_fed: rounds {T_SNAPSHOT+1}-{T_COLD} ...", end="", flush=True)
            clients_wf = make_clients(providers, seed)
            for c in clients_wf.values():
                c.set_global_weights(snapshot_weights)
            for rnd in range(T_SNAPSHOT + 1, T_COLD + 1):
                fedavg_round(list(clients_wf.values()), mu=0.0)
                record_round(all_rows, seed, "warm_fed", rnd, clients_wf, providers)
            print(" done")

            # Condition 4: warm_local (local-only from cold_fed round-30 checkpoint)
            print(f"  warm_local: rounds {T_SNAPSHOT+1}-{T_COLD} ...", end="", flush=True)
            clients_wl = make_clients(providers, seed)
            for c in clients_wl.values():
                c.set_global_weights(snapshot_weights)
            for rnd in range(T_SNAPSHOT + 1, T_COLD + 1):
                for c in clients_wl.values():
                    c.local_update(mu=0.0)
                record_round(all_rows, seed, "warm_local", rnd, clients_wl, providers)
            print(" done")

        prov_df = pd.DataFrame(all_rows)
        prov_df.to_csv(provider_csv, index=False)
        print(f"\nProvider-round results: {len(prov_df)} rows -> {provider_csv}")

    # -- Summary --
    summ_df = compute_summary(prov_df)
    summ_df.to_csv(OUT / "e6_summary_round_results.csv", index=False)
    print(f"Summary results: {len(summ_df)} rows -> {OUT/'e6_summary_round_results.csv'}")

    # -- Metadata --
    meta = {
        "experiment":   "E6_warm_cold_fairness_deconfound",
        "data_source":  str(DATA),
        "data_sha256":  data_hash,
        "feature_cols": feature_cols,
        "n_features":   len(feature_cols),
        "config": {
            "seeds":          SEEDS,
            "T_cold":         T_COLD,
            "T_snapshot":     T_SNAPSHOT,
            "n_local_epochs": N_LOCAL_EPOCHS,
            "lr":             LR,
            "test_frac":      TEST_FRAC,
            "min_prov_size":  MIN_PROV_SIZE,
            "aggregator":     "FedAvg (mu=0)",
        },
        "conditions": {
            "cold_fed":   "FedAvg from zeros, rounds 1-60",
            "cold_local": "Local-only from zeros, rounds 1-60",
            "warm_fed":   "FedAvg from cold_fed round-30 weights, rounds 31-60",
            "warm_local": "Local-only from cold_fed round-30 weights, rounds 31-60",
        },
        "fairness_metrics": ["mean_auroc", "min_auroc", "var_auroc", "iqr_auroc"],
        "interpretation": {
            "warm_fed_eq_cold_fed_r31_60":    "Mathematical identity: same FedAvg trajectory from round 31",
            "warm_local_gt_warm_fed":         "After warm init, local fine-tuning slightly outperforms federation; suggests benefit is primarily initialisation",
            "worst_off_provider_warm_local":  "Min AUROC higher under warm_local than warm_fed",
        },
    }
    (OUT / "e6_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata -> {OUT/'e6_metadata.json'}")

    return prov_df, summ_df


if __name__ == "__main__":
    prov_df, summ_df = run_e6()

    print("\n-- Seed-averaged mean AUROC at rounds 1, 30, 60 by condition --")
    pivot = (summ_df.groupby(["condition", "round"])["mean_auroc"]
             .mean().reset_index())
    for r in [1, 30, 60]:
        sub = pivot[pivot["round"] == r]
        if len(sub) > 0:
            print(f"\n  Round {r}:")
            for _, row in sub.iterrows():
                print(f"    {row.condition:15s}  mean AUROC = {row.mean_auroc:.5f}")

    print("\n-- Min AUROC (worst-off provider) seed-averaged at round 60 --")
    at60 = summ_df[summ_df["round"] == 60].groupby("condition")["min_auroc"].mean()
    print(at60.to_string())
