"""
W09 Targeted Rerun — Attack RNG Reproducibility Fix
=====================================================
Reruns only the configurations affected by the three W09 RNG bugs:

  W09a  atk_random_gaussian: was unseeded (default_rng() with no seed).
        Fix: seed from (outer_seed, round_t, client_i).

  W09b  6C-i Clipped Gaussian noise sweep: same noise vector for all clients
        per round (seed = seed*1000+t). Fix: per-client seed.

  W09c  atk_backdoor: fixed seed=42 independent of outer run seed.
        Fix: seed from (outer_seed, round_t, client_i).

Outputs (saved alongside originals; originals are NOT overwritten):
  trust-ct/processed/phase6/phase6b_adversarial_w09.csv
      — random_gauss and backdoor rows only (eICU + Cimas, all f/defense/seed)
  trust-ct/processed/phase6/phase6c_leakage_fl_w09.csv
      — corrected 6C-i noise sweep rows (all alpha/seed, eICU)

Usage:
  python run_phase6_w09_rerun.py

From: trust-ct/remediation/patches/w09_rng_patch.py (patches written earlier).
"""

import hashlib, json, pathlib, time, warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm as sp_norm
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                              matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT    = pathlib.Path(__file__).resolve().parent
FEDLEARN = ROOT.parent
OUT_DIR  = ROOT / "processed" / "phase6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EICU_CSV = FEDLEARN / "evaluation" / "prepared_datasets" / "eicu_demo_prepared.csv"
CIMAS_PQ = ROOT / "processed" / "cimas" / "cimas_htn_landmark_6m.parquet"

# ── Parameters (must match run_phase6.py exactly) ──────────────────────────────
SEEDS        = [7, 11, 19, 23, 37]
N_ROUNDS     = 15
N_LOCAL      = 5
LR           = 0.02
N_TOP_FEAT   = 60
TEST_FRAC    = 0.20
MIN_CLI_REC  = 5

ATTACK_SCALE  = 3.0
BDOOR_FRAC    = 0.30
TRIGGER_FEAT  = 0
TRIGGER_VAL   = 3.0
BDOOR_TARGET  = 0

K_EICU   = 8
F_EICU   = [1, 2, 3]

K_CIMAS  = 23
F_CIMAS  = [2, 5, 7]

ALPHA_VALS   = [0.0, 0.01, 0.05, 0.10, 0.20]
CLIP_NORM    = 1.0

KRUM_OK = lambda K, f: K > 2 * f + 2

RUN_ID = "w09"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES (copied verbatim from run_phase6.py)
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def balanced_weights(y):
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    return np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

def local_sgd(w0: np.ndarray, X: np.ndarray, y: np.ndarray,
              n_epochs: int = N_LOCAL, lr: float = LR) -> np.ndarray:
    w = w0.copy()
    sw = balanced_weights(y)
    for _ in range(n_epochs):
        err = sigmoid(X @ w[:-1] + w[-1]) - y
        w[:-1] -= lr * (sw * err) @ X / len(y)
        w[-1]  -= lr * (sw * err).mean()
    return w

def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w[:-1] + w[-1])

def eval_metrics(w, X_test, y_test):
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return {"auroc": np.nan, "pr_auc": np.nan, "bal_acc": np.nan, "mcc": np.nan}
    p    = predict_proba(w, X_test)
    pred = (p >= 0.5).astype(int)
    return {
        "auroc":   round(float(roc_auc_score(y_test, p)), 5),
        "pr_auc":  round(float(average_precision_score(y_test, p)), 5),
        "bal_acc": round(float(balanced_accuracy_score(y_test, pred)), 5),
        "mcc":     round(float(matthews_corrcoef(y_test, pred)), 5),
    }

def auc_lc(auroc_series):
    av = np.asarray(auroc_series, dtype=float)
    av = av[~np.isnan(av)]
    return float(np.trapz(av) / max(len(av) - 1, 1)) if len(av) > 1 else np.nan

def bootstrap_ci(values, n_boot=500, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    v   = np.asarray(values, dtype=float)
    v   = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return (float(np.mean(v)),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))

def preprocess(X_train_raw, X_test_raw):
    imp = SimpleImputer(strategy="median").fit(X_train_raw)
    scl = StandardScaler().fit(imp.transform(X_train_raw))
    Xtr = scl.transform(imp.transform(X_train_raw))
    Xte = scl.transform(imp.transform(X_test_raw))
    return Xtr, Xte, imp, scl


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION DEFENSES (verbatim from run_phase6.py)
# ═══════════════════════════════════════════════════════════════════════════════

def agg_fedavg(deltas, sizes):
    sizes = np.array(sizes, dtype=float)
    w     = sizes / sizes.sum()
    return sum(wi * d for wi, d in zip(w, deltas))

def agg_coord_median(deltas, sizes=None):
    return np.median(np.array(deltas), axis=0)

def agg_trimmed_mean(deltas, sizes=None, f=1):
    D = np.array(deltas)
    K = len(D)
    t = max(1, f)
    if 2 * t >= K:
        t = max(0, K // 4)
    D_sort = np.sort(D, axis=0)
    return D_sort[t: K - t, :].mean(axis=0) if K - 2 * t > 0 else D.mean(axis=0)

def agg_krum(deltas, sizes=None, f=1):
    D = np.array(deltas)
    K = len(D)
    n_sel = K - f - 2
    if n_sel < 1:
        return agg_fedavg(deltas, sizes or [1] * K)
    dists = np.array([[np.linalg.norm(D[i] - D[j]) ** 2
                       for j in range(K)] for i in range(K)])
    scores = np.array([np.sort(dists[i])[1: n_sel + 1].sum() for i in range(K)])
    return D[np.argmin(scores)]

def agg_clipping(deltas, sizes, clip_norm=CLIP_NORM):
    clipped = []
    for d in deltas:
        norm = np.linalg.norm(d)
        clipped.append(d * clip_norm / norm if norm > clip_norm else d.copy())
    return agg_fedavg(clipped, sizes)

AGGREGATORS = {
    "fedavg":       lambda d, s, f: agg_fedavg(d, s),
    "coord_median": lambda d, s, f: agg_coord_median(d),
    "trimmed_mean": lambda d, s, f: agg_trimmed_mean(d, s, f),
    "krum":         lambda d, s, f: agg_krum(d, s, f),
    "clipping":     lambda d, s, f: agg_clipping(d, s),
}


# ═══════════════════════════════════════════════════════════════════════════════
# W09 PATCHED ATTACK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_GAUSS_RNG = np.random.default_rng(0)
_BDOOR_RNG = np.random.default_rng(0)


def seed_attack_rngs(outer_seed: int, round_t: int, client_i: int) -> None:
    global _GAUSS_RNG, _BDOOR_RNG
    base = outer_seed * 10_000_000 + round_t * 10_000 + client_i * 10
    _GAUSS_RNG = np.random.default_rng(base + 1)
    _BDOOR_RNG = np.random.default_rng(base + 2)


def atk_random_gaussian_w09(delta, w_global, X, y):
    """W09a fix: uses module-level _GAUSS_RNG seeded via seed_attack_rngs()."""
    scale = ATTACK_SCALE * np.linalg.norm(delta) + 1e-8
    return _GAUSS_RNG.normal(0, scale, size=delta.shape)


def atk_backdoor_w09(delta, w_global, X, y):
    """W09c fix: uses module-level _BDOOR_RNG seeded via seed_attack_rngs()."""
    n_poison   = max(1, int(BDOOR_FRAC * len(y)))
    idx_poison = _BDOOR_RNG.choice(len(y), size=n_poison, replace=False)
    X_p, y_p   = X.copy(), y.copy()
    X_p[idx_poison, TRIGGER_FEAT] = TRIGGER_VAL
    y_p[idx_poison]               = BDOOR_TARGET
    w_local = local_sgd(w_global.copy(), X_p, y_p)
    return w_local - w_global


# Attacks NOT affected by W09 — kept verbatim to confirm unchanged
def atk_none(delta, w_global, X, y):     return delta
def atk_sign_flip(delta, w_global, X, y): return -delta
def atk_scaled_poison(delta, w_global, X, y): return ATTACK_SCALE * delta
def atk_model_replace(delta, w_global, X, y, K=8, f=1):
    scale = K / max(f, 1)
    return scale * (np.zeros_like(w_global) - w_global)
def atk_label_flip(delta, w_global, X, y):
    w_local = local_sgd(w_global.copy(), X, 1 - y)
    return w_local - w_global
def atk_alie(delta, w_global, X, y, honest_deltas=None):
    if honest_deltas is None or len(honest_deltas) == 0:
        return delta
    H = np.array(honest_deltas)
    mu = H.mean(axis=0); sigma = H.std(axis=0) + 1e-8
    n = len(H); f_est = max(1, int(round(n * 0.25)))
    z_max = float(sp_norm.ppf(max((n - f_est) / (n + 1e-6), 0.51)))
    return mu + min(z_max, 5.0) * sigma


W09_ATTACK_FNS = {
    "random_gauss":  atk_random_gaussian_w09,
    "backdoor":      atk_backdoor_w09,
}

ATTACK_NEEDS_HONEST = {"alie"}
ATTACK_HAS_BACKDOOR = {"backdoor"}


def backdoor_success_rate(w, X_test,
                           trigger_feat=TRIGGER_FEAT,
                           trigger_val=TRIGGER_VAL,
                           target_cls=BDOOR_TARGET):
    X_trig = X_test.copy()
    X_trig[:, trigger_feat] = trigger_val
    pred = (predict_proba(w, X_trig) >= 0.5).astype(int)
    return float((pred == target_cls).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# W09-PATCHED run_fl
# ═══════════════════════════════════════════════════════════════════════════════

def run_fl_w09(client_data, X_test, y_test, attack_name, f, defense_name,
               seed=7, n_rounds=N_ROUNDS):
    """
    run_fl with W09 fix: calls seed_attack_rngs(seed, t, i) before each
    malicious-client attack dispatch, ensuring reproducible per-(seed,round,client)
    RNG state for random_gauss and backdoor.
    """
    rng       = np.random.default_rng(seed)
    K         = len(client_data)
    d         = client_data[0][0].shape[1] + 1
    w_global  = rng.normal(0.0, 0.01, d)
    mal_idx   = set(rng.choice(K, size=f, replace=False).tolist())
    agg_fn    = AGGREGATORS[defense_name]
    atk_fn    = W09_ATTACK_FNS[attack_name]   # only called for W09-affected attacks
    is_alie   = False                           # not rerunning ALIE
    is_bdoor  = attack_name in ATTACK_HAS_BACKDOOR

    round_rows = []
    for t in range(n_rounds):
        deltas     = []
        sizes      = []
        hon_deltas = []

        for i, (X_c, y_c) in enumerate(client_data):
            if len(y_c) < MIN_CLI_REC or len(np.unique(y_c)) < 2:
                continue
            w_local = local_sgd(w_global.copy(), X_c, y_c)
            d_hon   = w_local - w_global
            if i not in mal_idx:
                hon_deltas.append(d_hon)

        for i, (X_c, y_c) in enumerate(client_data):
            if len(y_c) < MIN_CLI_REC or len(np.unique(y_c)) < 2:
                continue
            w_local = local_sgd(w_global.copy(), X_c, y_c)
            delta   = w_local - w_global
            if i in mal_idx:
                seed_attack_rngs(seed, t, i)   # W09 fix
                delta = atk_fn(delta, w_global, X_c, y_c)
            deltas.append(delta)
            sizes.append(len(y_c))

        if not deltas:
            break

        if defense_name == "krum" and not KRUM_OK(K, f):
            agg_delta  = agg_fedavg(deltas, sizes)
            krum_valid = False
        else:
            agg_delta  = agg_fn(deltas, sizes, f)
            krum_valid = True

        w_global = w_global + agg_delta
        m = eval_metrics(w_global, X_test, y_test)
        m["round"] = t + 1
        round_rows.append(m)

    return round_rows, w_global, krum_valid if defense_name == "krum" else True


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (identical to run_phase6.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== W09 Rerun: Loading data ===")

eicu_raw = pd.read_csv(EICU_CSV)
drop_eicu = {"patientunitstayid", "hospitalid", "site_group", "Outcome",
             "hospitaldischargestatus", "unitdischargestatus"}
feat_cols_eicu = [c for c in eicu_raw.columns if c not in drop_eicu]

X_eicu_all = eicu_raw[feat_cols_eicu].values.astype(float)
y_eicu_all = eicu_raw["Outcome"].values.astype(int)
sg_eicu    = eicu_raw["site_group"].values.astype(int)

tr_idx, te_idx = train_test_split(
    np.arange(len(y_eicu_all)), test_size=TEST_FRAC,
    stratify=y_eicu_all, random_state=7)

_imp60   = SimpleImputer(strategy="median").fit(X_eicu_all[tr_idx])
_X60_tr  = _imp60.transform(X_eicu_all[tr_idx])
_mi      = mutual_info_classif(_X60_tr, y_eicu_all[tr_idx], random_state=7)
TOP60    = np.argsort(_mi)[::-1][:N_TOP_FEAT]
X_eicu_all = X_eicu_all[:, TOP60]
feat_cols_eicu = [feat_cols_eicu[i] for i in TOP60]

X_eicu_tr_raw, X_eicu_te_raw = X_eicu_all[tr_idx], X_eicu_all[te_idx]
y_eicu_tr, y_eicu_te         = y_eicu_all[tr_idx], y_eicu_all[te_idx]
sg_tr_eicu                   = sg_eicu[tr_idx]

X_eicu_tr, X_eicu_te, _imp_e, _scl_e = preprocess(X_eicu_tr_raw, X_eicu_te_raw)

eicu_clients = []
for g in range(K_EICU):
    mask = sg_tr_eicu == g
    eicu_clients.append((X_eicu_tr[mask], y_eicu_tr[mask]))
eicu_sizes = [len(c[1]) for c in eicu_clients]
print(f"  eICU: N={len(y_eicu_all)}  K={K_EICU}  d=61  "
      f"pos_rate={y_eicu_all.mean():.3f}")

cimas_raw = pd.read_parquet(CIMAS_PQ)
cimas_pri = cimas_raw[cimas_raw["client"] != "other"].copy().reset_index(drop=True)
FEAT_CIMAS = [
    "age", "sex_female", "scheme_type_ord", "cover_type_bin",
    "annual_contrib_log", "n_refill_months_obs", "refill_recency_days",
    "early_months", "late_months", "has_2m_gap", "first_claim_month",
    "n_claims", "n_claim_dates", "n_products", "n_providers_vis",
    "obs_amount", "obs_units", "amount_per_claim", "units_per_claim",
    "n_networks",
]
avail_c = [c for c in FEAT_CIMAS if c in cimas_pri.columns]
X_cim_all = cimas_pri[avail_c].values.astype(float)
y_cim_all = cimas_pri["Y_adh"].values.astype(int)
cl_cim    = cimas_pri["client"].values
provs     = np.unique(cl_cim)

tr_c, te_c = train_test_split(
    np.arange(len(y_cim_all)), test_size=TEST_FRAC,
    stratify=y_cim_all, random_state=7)
X_cim_tr, X_cim_te, _imp_c, _scl_c = preprocess(
    X_cim_all[tr_c], X_cim_all[te_c])
y_cim_tr, y_cim_te = y_cim_all[tr_c], y_cim_all[te_c]
cl_cim_tr = cl_cim[tr_c]

cimas_clients = []
for p in provs:
    mask = cl_cim_tr == p
    if mask.sum() >= MIN_CLI_REC:
        cimas_clients.append((X_cim_tr[mask], y_cim_tr[mask]))
    else:
        cimas_clients.append((X_cim_tr[mask[:0]], y_cim_tr[mask[:0]]))
K_CIMAS_ACT = len(cimas_clients)
cimas_sizes = [len(c[1]) for c in cimas_clients]
print(f"  Cimas: N={len(y_cim_all)}  K={K_CIMAS_ACT}  d={len(avail_c)+1}  "
      f"pos_rate={y_cim_all.mean():.3f}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# W09a + W09c RERUN — random_gauss and backdoor only
# ═══════════════════════════════════════════════════════════════════════════════
print("=== W09 Rerun: Phase 6B (random_gauss + backdoor) ===")

RERUN_ATTACKS = ["random_gauss", "backdoor"]
DEFENSES      = list(AGGREGATORS.keys())


def run_w09_benchmark(dataset_name, clients, X_test, y_test, K, F_vals,
                      seeds=SEEDS):
    rows = []
    done = 0
    for atk in RERUN_ATTACKS:
        for f in F_vals:
            for defense in DEFENSES:
                krum_applicable = (defense != "krum") or KRUM_OK(K, f)
                for seed in seeds:
                    t0 = time.time()
                    round_rows, w_final, krum_v = run_fl_w09(
                        clients, X_test, y_test,
                        attack_name=atk, f=f, defense_name=defense, seed=seed)
                    elapsed = time.time() - t0

                    auroc_series = [r["auroc"] for r in round_rows
                                    if not np.isnan(r.get("auroc", np.nan))]
                    final = round_rows[-1] if round_rows else {}
                    bsr   = (backdoor_success_rate(w_final, X_test)
                             if atk == "backdoor" else np.nan)
                    conv_fail = int(len(auroc_series) < N_ROUNDS // 2)

                    row = {
                        "dataset": dataset_name, "attack": atk, "f": f,
                        "f_pct": round(100 * f / K, 1) if K > 0 else 0.0,
                        "defense": defense,
                        "krum_applicable": krum_applicable if defense == "krum" else True,
                        "seed": seed,
                        "auroc":    final.get("auroc", np.nan),
                        "pr_auc":   final.get("pr_auc", np.nan),
                        "bal_acc":  final.get("bal_acc", np.nan),
                        "mcc":      final.get("mcc", np.nan),
                        "auc_lc":   auc_lc(auroc_series),
                        "bsr":      round(bsr, 5) if not np.isnan(bsr) else np.nan,
                        "convergence_fail": conv_fail,
                        "runtime_s": round(elapsed, 3),
                        "comm_params_per_round": 2 * K * (X_test.shape[1] + 1),
                        "run_id": RUN_ID,
                    }
                    rows.append(row)
                    done += 1

    print(f"  {dataset_name}: {done} configs done.")
    return pd.DataFrame(rows)


print(f"  Running eICU (2 attacks × {len(F_EICU)} f × "
      f"{len(DEFENSES)} defenses × {len(SEEDS)} seeds)...")
eicu_w09 = run_w09_benchmark(
    "eicu", eicu_clients, X_eicu_te, y_eicu_te, K_EICU, F_EICU)

print(f"  Running Cimas (2 attacks × {len(F_CIMAS)} f × "
      f"{len(DEFENSES)} defenses × {len(SEEDS)} seeds)...")
cimas_w09 = run_w09_benchmark(
    "cimas", cimas_clients, X_cim_te, y_cim_te, K_CIMAS_ACT, F_CIMAS)

bench_w09 = pd.concat([eicu_w09, cimas_w09], ignore_index=True)
bench_w09.to_csv(OUT_DIR / "phase6b_adversarial_w09.csv", index=False)
print(f"  Written: phase6b_adversarial_w09.csv ({len(bench_w09)} rows)")
print()

# Summary for validation
print("  W09 eICU random_gauss (FedAvg, f=3) AUROC by seed:")
rg_f3 = bench_w09[(bench_w09.dataset=="eicu") & (bench_w09.attack=="random_gauss") &
                   (bench_w09.f==3) & (bench_w09.defense=="fedavg")]
for _, r in rg_f3.iterrows():
    print(f"    seed={r['seed']}  auroc={r['auroc']:.4f}")

print()
print("  W09 eICU backdoor (FedAvg, f=3) BSR by seed:")
bd_f3 = bench_w09[(bench_w09.dataset=="eicu") & (bench_w09.attack=="backdoor") &
                   (bench_w09.f==3) & (bench_w09.defense=="fedavg")]
for _, r in bd_f3.iterrows():
    print(f"    seed={r['seed']}  bsr={r['bsr']:.4f}  auroc={r['auroc']:.4f}")

bsr_vals = bd_f3["bsr"].dropna()
if len(bsr_vals):
    m, lo, hi = bootstrap_ci(bsr_vals)
    print(f"  BSR (W09-corrected): {m:.4f} [{lo:.4f}, {hi:.4f}]")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# W09b RERUN — 6C-i Clipped Gaussian noise sweep (per-client seeds)
# ═══════════════════════════════════════════════════════════════════════════════
print("=== W09 Rerun: Phase 6C-i (clipped Gaussian noise sweep, per-client RNG) ===")

leakage_fl_rows = []
for alpha in ALPHA_VALS:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d   = X_eicu_te.shape[1] + 1
        w   = rng.normal(0.0, 0.01, d)
        auroc_series = []
        for t in range(N_ROUNDS):
            deltas, sizes = [], []
            for ci_idx, (X_c, y_c) in enumerate(eicu_clients):
                if len(y_c) < MIN_CLI_REC or len(np.unique(y_c)) < 2:
                    continue
                w_loc = local_sgd(w.copy(), X_c, y_c)
                delta = w_loc - w
                # Clip
                norm = np.linalg.norm(delta)
                if norm > CLIP_NORM:
                    delta = delta * CLIP_NORM / norm
                # W09b fix: unique per-client noise
                if alpha > 0:
                    noise_rng = np.random.default_rng(
                        seed * 1_000_000 + t * 1_000 + ci_idx)
                    noise = noise_rng.normal(0, alpha * CLIP_NORM, size=delta.shape)
                    delta = delta + noise
                deltas.append(delta)
                sizes.append(len(y_c))
            if deltas:
                w = w + agg_fedavg(deltas, sizes)
            m = eval_metrics(w, X_eicu_te, y_eicu_te)
            auroc_series.append(m.get("auroc", np.nan))
        final = eval_metrics(w, X_eicu_te, y_eicu_te)
        leakage_fl_rows.append({
            "alpha":   alpha, "seed": seed,
            "auroc":   final.get("auroc", np.nan),
            "pr_auc":  final.get("pr_auc", np.nan),
            "mcc":     final.get("mcc", np.nan),
            "auc_lc":  auc_lc(auroc_series),
            "run_id":  RUN_ID,
        })

leakage_fl_w09 = pd.DataFrame(leakage_fl_rows)
leakage_fl_w09.to_csv(OUT_DIR / "phase6c_leakage_fl_w09.csv", index=False)
print(f"  Written: phase6c_leakage_fl_w09.csv ({len(leakage_fl_w09)} rows)")
print()

print("  6C-i W09 utility by alpha (AUROC mean across 5 seeds):")
for alpha in ALPHA_VALS:
    sub = leakage_fl_w09[leakage_fl_w09.alpha == alpha]["auroc"]
    print(f"    alpha={alpha:.2f}  auroc={sub.mean():.4f}  std={sub.std():.4f}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE RECORD
# ═══════════════════════════════════════════════════════════════════════════════
provenance = {
    "run_id": RUN_ID,
    "script": "trust-ct/run_phase6_w09_rerun.py",
    "remediation_issues": ["W09a", "W09b", "W09c"],
    "attacks_rerun": RERUN_ATTACKS,
    "datasets": ["eicu", "cimas"],
    "seeds": SEEDS,
    "n_rows_w09_adversarial": len(bench_w09),
    "n_rows_w09_leakage_fl":  len(leakage_fl_w09),
    "outputs": {
        "adversarial": "trust-ct/processed/phase6/phase6b_adversarial_w09.csv",
        "leakage_fl":  "trust-ct/processed/phase6/phase6c_leakage_fl_w09.csv",
    },
    "originals_preserved": [
        "trust-ct/processed/phase6/phase6b_adversarial.csv",
        "trust-ct/processed/phase6/phase6c_leakage_fl.csv",
    ],
}
with open(OUT_DIR / "phase6_w09_provenance.json", "w") as fh:
    json.dump(provenance, fh, indent=2)
print(f"  Provenance: phase6_w09_provenance.json")
print()
print("=== W09 Rerun complete ===")
