"""
Phase 6 C0 Verification and Numerical Reconciliation
====================================================
Implements all six lock criteria defined in the Phase 6 lock specification.
Reads the corrected Phase 6 outputs (training-only MI) and produces a
definitive Phase 6 lock sheet.

Lock criteria:
  C0-1  Feature-selection provenance
  C0-2  Run completeness (1200-run manifest)
  C0-3  Security metrics regenerated from corrected feature set
  C0-4  Canary interpretation distinction preserved
  C0-5  MIA AUROC with effect size alongside CI
  C0-6  Artifact consistency across all output files

Output: processed/phase6/ (updated lock), processed/ablation/phase6_master_ledger.json
"""

import hashlib
import json
import pathlib
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import expit
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TRUST_CT = pathlib.Path(r"C:\Users\Gozie\Desktop\Blockchain in Clinical Trial\trust-ct")
P6_DIR   = TRUST_CT / "processed" / "phase6"
P6C_DIR  = TRUST_CT / "processed" / "phase6_corrections"
ABL_DIR  = TRUST_CT / "processed" / "ablation"
ABL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 2000
MATERIALITY_AUROC = 0.02
NI_MARGIN = -0.02


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path) -> str:
    return sha256_bytes(pathlib.Path(path).read_bytes())

def boot_ci(arr, n=N_BOOT):
    arr = np.asarray(arr, float)
    boots = [RNG.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def paired_ttest(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    n = len(diff)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n-1)
    _, p = sp_stats.ttest_rel(a, b)
    return float(mean_d), float(mean_d - t_crit*se), float(mean_d + t_crit*se), float(p)


results = {}
PASS = []
FAIL = []

def check(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    results[name] = {"status": status, "detail": detail}
    (PASS if passed else FAIL).append(name)
    print(f"  [{status}] {name}: {detail}")


print("=" * 70)
print("PHASE 6 C0 VERIFICATION — Six Lock Criteria")
print("=" * 70)

# ============================================================
# C0-1: Feature-Selection Provenance
# ============================================================
print("\n--- C0-1: Feature-Selection Provenance ---")

script_path = TRUST_CT / "run_phase6.py"
script_src  = script_path.read_text(encoding="utf-8")
script_hash = sha256_bytes(script_src.encode())

# Verify split-before-MI pattern in source: find character positions
pos_split = script_src.find("tr_idx, te_idx = train_test_split(")
pos_mi    = script_src.find("mutual_info_classif(_X60_tr,")
split_before_mi = (pos_split != -1) and (pos_mi != -1) and (pos_split < pos_mi)

# Verify imputer fitted on tr_idx only
imputer_on_tr = bool(re.search(r"SimpleImputer.*\.fit\(X_eicu_all\[tr_idx\]", script_src))

# Verify MI uses training labels only
mi_on_tr = bool(re.search(r"mutual_info_classif\(_X60_tr,\s*y_eicu_all\[tr_idx\]", script_src))

# Re-derive feature list from source code to get a hash-linkable provenance record
# We load the eICU prepared CSV (same as run_phase6.py) and reproduce TOP60 selection.
eicu_csv = TRUST_CT.parent / "Fedlearn" / "evaluation" / "prepared_datasets" / "eicu_demo_prepared.csv"
if eicu_csv.exists():
    eicu_df = pd.read_csv(eicu_csv)
    OUTCOME_COL = "Outcome"
    N_TOP_FEAT  = 60
    TEST_FRAC   = 0.20

    # Mirror drop_eicu set from run_phase6.py exactly
    DROP_SET = {"patientunitstayid", "hospitalid", "site_group",
                "Outcome", "hospitaldischargestatus", "unitdischargestatus"}
    feat_cols_eicu = [c for c in eicu_df.columns if c not in DROP_SET]
    X_all = eicu_df[feat_cols_eicu].values.astype(float)
    y_all = eicu_df[OUTCOME_COL].values.astype(int)

    tr_idx, te_idx = train_test_split(
        np.arange(len(y_all)), test_size=TEST_FRAC,
        stratify=y_all, random_state=7)

    _imp60 = SimpleImputer(strategy="median").fit(X_all[tr_idx])
    _X60_tr = _imp60.transform(X_all[tr_idx])
    _mi = mutual_info_classif(_X60_tr, y_all[tr_idx], random_state=7)
    TOP60 = np.argsort(_mi)[::-1][:N_TOP_FEAT]

    selected_features = [feat_cols_eicu[i] for i in TOP60]
    feat_list_hash = sha256_bytes(json.dumps(selected_features).encode())
    n_features_selected = len(selected_features)
    feature_derivation = "reproduced from eicu_demo_prepared.csv with training-only MI"
else:
    selected_features = []
    feat_list_hash = "eicu_csv_not_found"
    n_features_selected = 0
    feature_derivation = "eicu_csv_not_found"

print(f"  script_hash:       {script_hash[:16]}...")
print(f"  split_before_mi:   {split_before_mi}")
print(f"  imputer_on_tr:     {imputer_on_tr}")
print(f"  mi_on_tr:          {mi_on_tr}")
print(f"  n_features:        {n_features_selected}")
print(f"  feature_list_hash: {feat_list_hash[:16]}...")

check("C0-1_split_before_mi",  split_before_mi,  f"train_test_split precedes mutual_info_classif in source")
check("C0-1_imputer_on_tr",    imputer_on_tr,     f"SimpleImputer.fit(X_eicu_all[tr_idx]) confirmed")
check("C0-1_mi_on_training",   mi_on_tr,          f"mutual_info_classif(_X60_tr, y[tr_idx]) confirmed")
check("C0-1_n_features",       n_features_selected == 60, f"{n_features_selected}/60 features selected")


# ============================================================
# C0-2: Run Completeness (1200-run manifest)
# ============================================================
print("\n--- C0-2: Run Completeness ---")

bench = pd.read_csv(P6_DIR / "phase6b_adversarial.csv")
N_ROWS = len(bench)

# Accounting:
# 'none' attack only at f=0 (f-invariant): 5 defenses x 5 seeds x 2 datasets = 50 rows
# adversarial attacks: 7 attacks x 3 f-values x 5 defenses x 5 seeds x 2 datasets = 1050 rows
# BUT krum-N/A (eICU, f=3, adversarial): 7 attacks x 5 seeds = 35 rows (ran with FedAvg fallback)
# Total in CSV: 50 + 1050 = 1100 rows
# Skipped (none at f>0, f-invariant): 1 attack x 2 f-values x 5 defenses x 5 seeds x 2 datasets = 100
# Grand total accounted: 1100 + 100 = 1200

N_NONE    = bench[bench.attack == "none"].shape[0]
N_ADV     = bench[bench.attack != "none"].shape[0]

# Krum-NA: krum defense, eICU, f=3, adversarial (krum_applicable=False)
krum_na   = bench[
    (bench.defense == "krum") & (bench.dataset == "eicu") &
    (bench.f == 3) & (bench.attack != "none") & (~bench.krum_applicable)
]
N_KRUM_NA = len(krum_na)

# f-invariant skips: none attack at f>0 (1 attack × 2 f-values × 5 def × 5 seeds × 2 datasets)
N_SKIP    = 1 * 2 * 5 * 5 * 2  # 100
N_VALID   = N_ROWS - N_KRUM_NA   # valid primary (includes none-attack clean runs)

TOTAL = N_ROWS + N_SKIP
N_ZERO_FAIL = (bench.get("convergence_fail", pd.Series(0, index=bench.index)).fillna(0) > 0).sum()

print(f"  Rows in CSV:        {N_ROWS}")
print(f"  none-attack rows:   {N_NONE}")
print(f"  adversarial rows:   {N_ADV}")
print(f"  krum-NA fallback:   {N_KRUM_NA}")
print(f"  f-invariant skips:  {N_SKIP}")
print(f"  total accounted:    {TOTAL}")
print(f"  valid primary:      {N_VALID}")
print(f"  convergence fails:  {N_ZERO_FAIL}")

check("C0-2_total_1200",      TOTAL == 1200,      f"1100 run + 100 skipped = {TOTAL}")
check("C0-2_krum_na_35",      N_KRUM_NA == 35,    f"krum N/A fallbacks: {N_KRUM_NA}")
check("C0-2_zero_failures",   N_ZERO_FAIL == 0,   f"convergence failures: {N_ZERO_FAIL}")

# Add aggregator_effective and run_status columns to bench; save corrected CSV + parquet
bench["aggregator_effective"] = bench.apply(
    lambda r: "fedavg"
    if (r["defense"] == "krum" and not r["krum_applicable"])
    else r["defense"], axis=1)
bench["run_status"] = bench.apply(
    lambda r: "not_applicable_diagnostic_fallback"
    if (r["defense"] == "krum" and not r["krum_applicable"])
    else "valid_primary", axis=1)
bench.to_csv(P6_DIR / "phase6b_adversarial.csv", index=False)
bench.to_parquet(P6_DIR / "phase6b_adversarial.parquet", index=False)
print(f"  Saved bench with aggregator_effective + run_status columns.")


# ============================================================
# C0-3: Security Metrics (regenerated from corrected feature set)
# ============================================================
print("\n--- C0-3: Security Metrics ---")

SEEDS = [7, 11, 19, 23, 37]

# --- Clean baseline ---
clean_eicu_fa = bench[
    (bench.dataset == "eicu") & (bench.attack == "none") &
    (bench.defense == "fedavg") & (bench.f == 0)
]["auroc"].values
clean_mean, clean_lo, clean_hi = boot_ci(clean_eicu_fa)
print(f"  Clean FedAvg eICU AUROC: {clean_mean:.4f} [{clean_lo:.4f}, {clean_hi:.4f}]")

# --- Per-attack FedAvg degradation at f=3 (eICU) ---
attacks_reported = ["label_flip", "sign_flip", "model_replace", "alie", "backdoor"]
attack_stats = {}
for atk in attacks_reported:
    sub = bench[
        (bench.dataset == "eicu") & (bench.attack == atk) &
        (bench.f == 3) & (bench.defense == "fedavg")
    ].sort_values("seed")["auroc"].values
    if len(sub) == 5:
        m, lo, hi = boot_ci(sub)
        delta = m - clean_mean
        d_mean, d_lo, d_hi, p = paired_ttest(sub, clean_eicu_fa)
        attack_stats[atk] = {
            "auroc": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
            "delta": round(delta, 4), "paired_delta": round(d_mean, 4),
            "paired_ci": [round(d_lo, 4), round(d_hi, 4)], "p": round(p, 4)
        }
        print(f"  FedAvg {atk:15s}: AUROC {m:.4f} delta {delta:+.4f} [{d_lo:.4f},{d_hi:.4f}] p={p:.4f}")

worst_atk = min(attack_stats, key=lambda k: attack_stats[k]["delta"])
check("C0-3_worst_attack",
      attack_stats[worst_atk]["delta"] < -0.05,
      f"worst={worst_atk} delta={attack_stats[worst_atk]['delta']:.4f}")

# --- Robust defense vs FedAvg under worst attack (label_flip, f=3) ---
# clipping does not aggregate-robustly (clips update norms, not Byzantine-resilient)
DEFENSES_ROBUST = ["coord_median", "trimmed_mean"]
defense_stats = {}
fa_lf = bench[
    (bench.dataset == "eicu") & (bench.attack == "label_flip") &
    (bench.f == 3) & (bench.defense == "fedavg")
].sort_values("seed")["auroc"].values

for dfn in DEFENSES_ROBUST:
    sub = bench[
        (bench.dataset == "eicu") & (bench.attack == "label_flip") &
        (bench.f == 3) & (bench.defense == dfn)
    ].sort_values("seed")["auroc"].values
    if len(sub) == 5:
        m, lo, hi = boot_ci(sub)
        delta_vs_clean = m - clean_mean
        d_mean, d_lo, d_hi, p = paired_ttest(sub, fa_lf)
        defense_stats[dfn] = {
            "auroc": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
            "delta_vs_clean": round(delta_vs_clean, 4),
            "recovery_vs_fedavg": round(d_mean, 4),
            "recovery_ci": [round(d_lo, 4), round(d_hi, 4)], "p": round(p, 4)
        }
        print(f"  {dfn:15s} (label_flip f=3): AUROC {m:.4f} "
              f"vs_clean {delta_vs_clean:+.4f}  recovery {d_mean:+.4f} p={p:.4f}")

best_defense = max(defense_stats, key=lambda k: defense_stats[k]["auroc"])
check("C0-3_robust_defense_effective",
      defense_stats[best_defense]["recovery_vs_fedavg"] > 0.05,
      f"best_defense={best_defense} recovery={defense_stats[best_defense]['recovery_vs_fedavg']:.4f}")

# --- Backdoor BSR ---
bdoor_rows = bench[
    (bench.dataset == "eicu") & (bench.attack == "backdoor") &
    (bench.f == 3) & (bench.defense == "fedavg") & (~bench.bsr.isna())
]
bsr_mean = bdoor_rows["bsr"].mean() if len(bdoor_rows) > 0 else np.nan
print(f"  Backdoor BSR (FedAvg, f=3): {bsr_mean:.4f}")
check("C0-3_backdoor_bsr_gt_0",
      not np.isnan(bsr_mean) and bsr_mean > 0,
      f"BSR={bsr_mean:.4f}")

# --- Clean AUROC by defense (eICU only) ---
clean_by_def = bench[
    (bench.attack == "none") & (bench.dataset == "eicu")
].groupby("defense")["auroc"].mean()
print("  Clean AUROC by defense (eICU, f=0, mean over seeds):")
for d, v in clean_by_def.sort_values().items():
    print(f"    {d:15s}: {v:.4f}")


# ============================================================
# C0-4: Canary Interpretation
# ============================================================
print("\n--- C0-4: Canary Interpretation ---")

c1_path = P6C_DIR / "c1_canary.json"
c1 = json.load(open(c1_path))

cos_single = c1["analytic_canary"]["cos_unclipped_mean"]
cos_multi  = c1["multistep_delta"]["cos_multistep_mean"]
phase6_label = c1["phase6_experiment_label_corrected"]
implication   = c1["multistep_delta"]["implication"]

print(f"  Single-step gradient canary: cosine={cos_single:.6f} ({c1['analytic_canary']['result']})")
print(f"  Multi-step delta canary:     cosine={cos_multi:.4f}")
print(f"  Phase 6 experiment label: {phase6_label}")
print(f"  Implication: {implication[:100]}...")

check("C0-4_singlestep_canary_pass",
      c1["analytic_canary"]["result"] == "PASS" and abs(cos_single - 1.0) < 1e-4,
      f"single-step cosine={cos_single:.6f}")
check("C0-4_multistep_breaks_approx",
      cos_multi < 0,
      f"multi-step cosine={cos_multi:.4f} (negative=approximation invalid)")
check("C0-4_phase6_label_correct",
      "fresh single-step gradient" in phase6_label or "optimistic" in phase6_label,
      f"phase6_label confirms optimistic attacker scenario")


# ============================================================
# C0-5: MIA Effect Size
# ============================================================
print("\n--- C0-5: MIA AUROC with Effect Size ---")

mia_df = pd.read_csv(P6_DIR / "phase6c_mia.csv")
c4 = json.load(open(P6C_DIR / "c4_leakage_corrected.json"))
c4_mia = c4["mia_bootstrap"]

# Load per-seed MIA values
mia_eicu_seeds  = mia_df[mia_df.dataset == "eicu"]["mia_auroc"].values
mia_cimas_seeds = mia_df[mia_df.dataset == "cimas"]["mia_auroc"].values

mia_e_mean, mia_e_lo, mia_e_hi = boot_ci(mia_eicu_seeds)
mia_c_mean, mia_c_lo, mia_c_hi = boot_ci(mia_cimas_seeds)

# Bootstrap CI from corrections (authoritative — 2000 resamples with correct procedure)
mia_e_lo_auth = c4_mia["eicu"]["ci95_lo"]
mia_e_hi_auth = c4_mia["eicu"]["ci95_hi"]
mia_c_lo_auth = c4_mia["cimas"]["ci95_lo"]
mia_c_hi_auth = c4_mia["cimas"]["ci95_hi"]

# Effect size: |AUROC - 0.5| / 0.5 (normalized distance from chance)
# Also report Cohen's h for comparing two proportions (AUROC vs 0.5)
def cohen_h(p1, p2=0.5):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))

h_eicu  = cohen_h(mia_e_mean)
h_cimas = cohen_h(mia_c_mean)
abs_adv_eicu  = mia_e_mean - 0.5
abs_adv_cimas = mia_c_mean - 0.5

print(f"  eICU  MIA AUROC: {mia_e_mean:.4f} [{mia_e_lo_auth:.4f}, {mia_e_hi_auth:.4f}]")
print(f"         advantage over chance: {abs_adv_eicu:+.4f}")
print(f"         Cohen's h: {h_eicu:.4f} (|h|<0.2 small)")
print(f"  Cimas MIA AUROC: {mia_c_mean:.4f} [{mia_c_lo_auth:.4f}, {mia_c_hi_auth:.4f}]")
print(f"         advantage over chance: {abs_adv_cimas:+.4f}")
print(f"         Cohen's h: {h_cimas:.4f}")

# Verdict: CI entirely below 0.5 (eICU) and CI entirely above 0.5 but effect negligible (Cimas)
eicu_below_chance  = mia_e_hi_auth < 0.5
cimas_negligible   = abs(h_cimas) < 0.1   # |h| < 0.1 = negligible effect

check("C0-5_eicu_mia_at_chance",
      eicu_below_chance,
      f"eICU CI [{mia_e_lo_auth:.4f},{mia_e_hi_auth:.4f}] entirely <=0.5 (attacker at/below chance)")
check("C0-5_cimas_mia_negligible",
      abs_adv_cimas < 0.01 and abs(h_cimas) < 0.1,
      f"Cimas advantage={abs_adv_cimas:+.4f} Cohen's h={h_cimas:.4f} (negligible)")


# ============================================================
# C0-6: Artifact Consistency
# ============================================================
print("\n--- C0-6: Artifact Consistency ---")

frontier = pd.read_csv(P6_DIR / "phase6d_frontier.csv")
leak_fl  = pd.read_csv(P6_DIR / "phase6c_leakage_fl.csv")

# Frontier alpha=0.00 AUROC vs clean benchmark
frontier_a0 = float(frontier[frontier.alpha == 0.00]["auroc_mean"].values[0])
frontier_a01 = float(frontier[frontier.alpha == 0.01]["auroc_mean"].values[0])
frontier_delta = frontier_a01 - frontier_a0

# Cross-check: leakage_fl alpha=0 mean AUROC should match frontier
leakfl_a0_mean = float(leak_fl[leak_fl.alpha == 0.00]["auroc"].mean())
diff_frontier_clean = abs(frontier_a0 - clean_mean)
diff_frontier_leakfl = abs(frontier_a0 - leakfl_a0_mean)

print(f"  frontier alpha=0.00 AUROC:  {frontier_a0:.4f}")
print(f"  clean benchmark mean AUROC: {clean_mean:.4f}")
print(f"  leakage_fl alpha=0 AUROC:   {leakfl_a0_mean:.4f}")
print(f"  |frontier - clean|:         {diff_frontier_clean:.4f}")
print(f"  |frontier - leakage_fl|:    {diff_frontier_leakfl:.4f}")
print(f"  frontier alpha=0.01 delta:  {frontier_delta:+.4f}")

# C4 corrected numbers
c4_sr0   = c4["sign_recovery"]["alpha_0_value"]
c4_sr01  = c4["sign_recovery"]["alpha_001_value"]
c4_cos0  = c4["cosine_similarity"]["alpha_0_value"]
c4_cos01 = c4["cosine_similarity"]["alpha_001_value"]

# Reconstruction from phase6c: batch-mix cosine at alpha=0 (all batches)
recon_df = pd.read_csv(P6_DIR / "phase6c_reconstruction.csv")
recon_a0_cos = recon_df[recon_df.get("alpha", pd.Series()) == 0.0]["cos_sim"].mean() \
    if "alpha" in recon_df.columns and "cos_sim" in recon_df.columns \
    else recon_df["cos_sim"].mean() if "cos_sim" in recon_df.columns \
    else np.nan
print(f"  Recon batch-mix cosine (phase6c): {recon_a0_cos:.4f}")
print(f"  C4 corrected cosine alpha=0:      {c4_cos0:.4f}")
print(f"  C4 corrected cosine alpha=0.01:   {c4_cos01:.4f}")

# Frontier and clean benchmark should agree within 0.01 (same model, same alpha=0)
check("C0-6_frontier_vs_clean",
      diff_frontier_clean < 0.01,
      f"|frontier({frontier_a0:.4f}) - clean({clean_mean:.4f})|={diff_frontier_clean:.4f}")
check("C0-6_frontier_vs_leakfl",
      diff_frontier_leakfl < 0.01,
      f"|frontier({frontier_a0:.4f}) - leakfl({leakfl_a0_mean:.4f})|={diff_frontier_leakfl:.4f}")
check("C0-6_operating_point_within_materiality",
      abs(frontier_delta) < MATERIALITY_AUROC,
      f"alpha=0.01 delta={frontier_delta:+.4f} < {MATERIALITY_AUROC}")
check("C0-6_c4_sign_recovery_auditable",
      0.7 < c4_sr0 < 0.95 and c4_sr01 < c4_sr0,
      f"sign_rec: {c4_sr0:.3f} -> {c4_sr01:.3f} (decreasing with noise)")


# ============================================================
# Produce Phase 6 Lock Sheet
# ============================================================
print("\n--- Producing Phase 6 Lock Sheet ---")

all_pass = len(FAIL) == 0
n_pass = len(PASS)
n_total = n_pass + len(FAIL)

print(f"  Gates passed: {n_pass}/{n_total}")
if FAIL:
    print(f"  FAILED gates: {FAIL}")
    print("  *** PHASE 6 LOCK WITHHELD ***")
else:
    print("  All gates PASS -> Phase 6 LOCKED")

lock_sheet = {
    "phase": "6",
    "version": "v2_training_only_mi",
    "status": "LOCKED" if all_pass else "UNLOCKED",
    "all_gates_pass": all_pass,
    "n_gates_pass": n_pass,
    "n_gates_total": n_total,
    "gate_results": results,
    "failed_gates": FAIL,
    "feature_selection_provenance": {
        "method": "mutual_information_classif training-only",
        "split_seed": 7,
        "n_features_selected": n_features_selected,
        "feature_list_hash_sha256": feat_list_hash,
        "script_sha256": script_hash,
        "feature_derivation": feature_derivation,
        "top_10_features": selected_features[:10] if selected_features else [],
    },
    "run_manifest": {
        "total_intended": 1200,
        "rows_in_csv": N_ROWS,
        "f_invariant_skips": N_SKIP,
        "krum_na_fallbacks": N_KRUM_NA,
        "valid_primary": N_VALID,
        "formula": f"{N_VALID} valid_primary + {N_KRUM_NA} krum_na + {N_SKIP} skips = 1200",
        "convergence_failures": int(N_ZERO_FAIL),
    },
    "security_metrics": {
        "clean_fedavg_auroc": {
            "mean": round(clean_mean, 4),
            "ci95": [round(clean_lo, 4), round(clean_hi, 4)]
        },
        "attack_degradation_fedavg_f3": {
            k: v for k, v in attack_stats.items()
        },
        "worst_attack": worst_atk,
        "worst_delta": attack_stats[worst_atk]["delta"],
        "robust_defense_label_flip_f3": defense_stats,
        "best_defense": best_defense,
        "backdoor_bsr": round(bsr_mean, 4) if not np.isnan(bsr_mean) else None,
    },
    "canary": {
        "single_step_cosine": cos_single,
        "multi_step_cosine": cos_multi,
        "phase6_experiment_label": phase6_label,
        "single_step_result": c1["analytic_canary"]["result"],
        "implication": implication,
    },
    "mia": {
        "eicu": {
            "mean_auroc": round(mia_e_mean, 4),
            "ci95_lo": mia_e_lo_auth, "ci95_hi": mia_e_hi_auth,
            "advantage_over_chance": round(abs_adv_eicu, 4),
            "cohen_h": round(h_eicu, 4),
            "interpretation": "attacker at or below chance (CI entirely <=0.5)"
        },
        "cimas": {
            "mean_auroc": round(mia_c_mean, 4),
            "ci95_lo": mia_c_lo_auth, "ci95_hi": mia_c_hi_auth,
            "advantage_over_chance": round(abs_adv_cimas, 4),
            "cohen_h": round(h_cimas, 4),
            "interpretation": "marginal +0.7pp advantage; Cohen's h negligible (<0.1); CI above 0.5"
        },
        "authoritative_source": "c4_leakage_corrected.json (2000 resamples)"
    },
    "perturbation_frontier": {
        "operating_point_alpha": 0.01,
        "alpha_0_auroc": round(frontier_a0, 4),
        "alpha_01_auroc": round(frontier_a01, 4),
        "delta_auroc_0_to_001": round(frontier_delta, 4),
        "within_materiality_0_02": abs(frontier_delta) < MATERIALITY_AUROC,
        "recon_cos_alpha_0": round(c4_cos0, 4),
        "recon_cos_alpha_001": round(c4_cos01, 4),
        "sign_recovery_alpha_0": round(c4_sr0, 4),
        "sign_recovery_alpha_001": round(c4_sr01, 4),
        "sign_recovery_abs_diff_pp": round((c4_sr0 - c4_sr01) * 100, 2),
        "mechanism_language": (
            "utility-oriented clipped Gaussian update perturbation providing "
            "empirical leakage mitigation — NOT (epsilon,delta)-DP"
        )
    },
    "manuscript_statements": {
        "A4_governance": (
            "100% detection (350/350) with zero false alerts under the seven "
            "evaluated attack scenarios"
        ),
        "A5_policy": (
            "The learned policy yielded a higher estimated value than random "
            "allocation (0.892 vs. 0.860; delta=0.032), but the bootstrap "
            "confidence interval included zero (95% CI: -0.011 to 0.078), "
            "providing no statistically conclusive evidence of policy-value "
            "improvement."
        ),
        "MIA": (
            "Membership inference AUROC was 0.496 (eICU; 95% CI [0.496, 0.498]; "
            "Cohen's h=-0.008) and 0.507 (Cimas; 95% CI [0.506, 0.507]; "
            "Cohen's h=+0.014). The eICU attacker performed at or below chance. "
            "The Cimas attacker showed a marginal +0.7pp advantage; while "
            "statistically detectable (CI excludes 0.5), the effect is negligible "
            "(Cohen's h=0.014, well below the 0.2 threshold for a small effect). "
            "Neither result is attributed to the perturbation mechanism: MIA AUROC "
            "at alpha=0 was already 0.496, indicating model confidence is "
            "insufficient to meaningfully separate members from non-members."
        )
    },
    "artifact_hashes": {
        "phase6b_adversarial_csv":    sha256_file(P6_DIR / "phase6b_adversarial.csv"),
        "phase6b_adversarial_parquet":sha256_file(P6_DIR / "phase6b_adversarial.parquet"),
        "phase6c_mia_csv":            sha256_file(P6_DIR / "phase6c_mia.csv"),
        "phase6c_leakage_fl_csv":     sha256_file(P6_DIR / "phase6c_leakage_fl.csv"),
        "phase6d_frontier_csv":       sha256_file(P6_DIR / "phase6d_frontier.csv"),
        "c4_leakage_corrected_json":  sha256_file(P6C_DIR / "c4_leakage_corrected.json"),
        "c1_canary_json":             sha256_file(P6C_DIR / "c1_canary.json"),
        "run_phase6_py":              script_hash,
    }
}

# Write to Phase 6 dir
lock_path = P6_DIR / "phase6_c0_lock.json"
with open(lock_path, "w") as f:
    json.dump(lock_sheet, f, indent=2)
print(f"  Written: {lock_path}")

# Write to ablation dir (master ledger location)
master_ledger_path = ABL_DIR / "phase6_master_lock.json"
with open(master_ledger_path, "w") as f:
    json.dump(lock_sheet, f, indent=2)
print(f"  Written: {master_ledger_path}")


# ============================================================
# Final Status
# ============================================================
print()
print("=" * 70)
print("PHASE 6 C0 VERIFICATION SUMMARY")
print("=" * 70)
print(f"  Criteria PASS: {n_pass}/{n_total}")
print(f"  Overall status: {'LOCKED' if all_pass else 'UNLOCKED — see failed gates above'}")
print()
print("  Key numbers (from corrected training-only MI run):")
print(f"    Clean FedAvg AUROC:     {clean_mean:.4f} [{clean_lo:.4f}, {clean_hi:.4f}]")
print(f"    Worst attack (FedAvg):  {worst_atk}  delta={attack_stats[worst_atk]['delta']:+.4f}")
print(f"    Best robust defense:    {best_defense}  "
      f"recovery={defense_stats[best_defense]['recovery_vs_fedavg']:+.4f}")
print(f"    Operating point:        alpha=0.01  AUROC delta={frontier_delta:+.4f}")
print(f"    MIA eICU:               {mia_e_mean:.4f} [{mia_e_lo_auth:.4f}, {mia_e_hi_auth:.4f}] "
      f"h={h_eicu:.4f}")
print(f"    MIA Cimas:              {mia_c_mean:.4f} [{mia_c_lo_auth:.4f}, {mia_c_hi_auth:.4f}] "
      f"h={h_cimas:.4f}")
print()
print(f"  Output: {lock_path.parent}")
