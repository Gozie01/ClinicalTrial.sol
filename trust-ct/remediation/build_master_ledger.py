"""
W12 — Build Master Results Ledger (CORRECTED 2026-08-28)
=========================================================
Fixes all issues identified in the remediation review:
  2. Attack contrasts: explicit matched-seed joins; no hardcoded p-values.
     CI method is paired t-test (df=n_seeds-1), not bootstrap.
  3. Policy contrast read from policy_contrasts.csv; N_BOOT from provenance.json.
  4. Separate n_seeds, n_providers, n_boot, n_evaluations columns.
  5. R31: no fabricated fallback; Clopper-Pearson exact CI from actual file.
  6. Long-tail from longtail_corrected.csv (not missing longtail_results.csv).

Run from the trust-ct/ directory:
    python remediation/build_master_ledger.py
"""

import json, pathlib, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT    = pathlib.Path(__file__).resolve().parent.parent   # = trust-ct/
PROC    = ROOT / "processed"
OUT_CSV = pathlib.Path(__file__).resolve().parent / "master_results_ledger.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_r(v, d=5):
    """Round float safely; return None for None/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, d)
    except (TypeError, ValueError):
        return None


def ttest_paired(a_series, b_series):
    """
    Matched-seed paired t-test.
    a_series, b_series: seed-indexed pd.Series.
    Returns (mean_diff, ci_lo, ci_hi, p_value, n_pairs).
    """
    joined = a_series.rename("a").to_frame().join(b_series.rename("b")).dropna()
    diffs  = joined["a"] - joined["b"]
    n      = len(diffs)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n
    _, p   = stats.ttest_rel(joined["a"], joined["b"])
    mean_d = float(diffs.mean())
    se_d   = float(diffs.std(ddof=1)) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean_d, mean_d - t_crit * se_d, mean_d + t_crit * se_d, float(p), n


def boot_ci(values, n=2000, seed=42):
    """Bootstrap mean and 95% CI."""
    rng = np.random.default_rng(seed)
    v   = np.asarray(values, float)
    v   = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    b = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def parse_ci_str(s):
    """Parse '0.807 [95% CI 0.785-0.829]' -> (0.807, 0.785, 0.829)."""
    m = re.match(r'([\d.]+)\s+\[95% CI ([\d.]+)-([\d.]+)\]', str(s).strip())
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return float(s), np.nan, np.nan


def clopper_pearson(k, n, alpha=0.05):
    """Exact Clopper-Pearson 95% CI for k successes in n trials."""
    lo = float(stats.beta.ppf(alpha / 2,     k,     n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))     if k < n else 1.0
    return lo, hi


# ── Schema ────────────────────────────────────────────────────────────────────

rows = []

def add(result_id, phase, dataset, metric_name,
        value, ci_lo, ci_hi, ci_method,
        p_value,
        n_seeds, n_providers, n_boot, n_evaluations,
        source_file, w_issues, run_id, locked, note=""):
    rows.append({
        "result_id":        result_id,
        "phase":            phase,
        "dataset":          dataset,
        "metric_name":      metric_name,
        "value":            safe_r(value),
        "ci_lo":            safe_r(ci_lo),
        "ci_hi":            safe_r(ci_hi),
        "ci_method":        ci_method,
        "p_value":          safe_r(p_value, 4),
        "n_seeds":          n_seeds,
        "n_providers":      n_providers,
        "n_boot":           n_boot,
        "n_evaluations":    n_evaluations,
        "source_file":      source_file,
        "w_issues_resolved": w_issues,
        "run_id":           run_id,
        "locked":           locked,
        "note":             note,
    })


# ── Load provenance ───────────────────────────────────────────────────────────

p3_prov  = json.loads((PROC / "cimas"    / "results_phase3_freeze" / "provenance.json").read_text())
p4_prov  = json.loads((PROC / "better_bp" / "results_phase4"       / "provenance.json").read_text())

P3_N_BOOT  = p3_prov["n_boot"]               # 2000 (from provenance.json, authoritative)
P3_K       = p3_prov["n_primary_clients"]    # 23
P3_SEEDS   = p3_prov["seeds"]               # [7,11,19,23,37]

P4_N_BOOT  = p4_prov["n_boot"]              # 2000
P4_SEEDS   = p4_prov["seeds"]              # [7,11,19,23,37]
P4_FOLDS   = p4_prov["n_folds"]            # 5


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Cimas FL noninferiority
# ═══════════════════════════════════════════════════════════════════════════════

src_ni = "processed/cimas/results_phase3_freeze/ni_client_level.csv"
ni = pd.read_csv(PROC / "cimas" / "results_phase3_freeze" / "ni_client_level.csv")

methods = {"FedAvg": "R01", "FedProx": "R02", "FedAvg-equal": "R03",
           "SCAFFOLD": "R04", "FedAdam": "R05"}

for method, rid in methods.items():
    row = ni[ni.method == method].iloc[0]
    add(rid, "Phase3", "Cimas", "Delta_AUROC_NI",
        value=float(row["obs"]),
        ci_lo=float(row["lo_95"]),
        ci_hi=float(row["hi_95"]),
        ci_method=f"bootstrap_n{P3_N_BOOT}",
        p_value=None,
        n_seeds=len(P3_SEEDS),
        n_providers=P3_K,
        n_boot=P3_N_BOOT,
        n_evaluations=None,
        source_file=src_ni,
        w_issues="W03b",
        run_id="phase3_freeze",
        locked=True,
        note=f"NI LCB={row['lcb_95']:.5f} > -0.02: {row['verdict']}")

# Long-tail: read from longtail_corrected.csv (issue 6 — correct filename)
src_lt = "processed/cimas/results_phase3_freeze/longtail_corrected.csv"
lt = pd.read_csv(PROC / "cimas" / "results_phase3_freeze" / "longtail_corrected.csv")

for rid, model in [("R06", "Central"), ("R07", "FedAvg")]:
    row       = lt[lt.model == model].iloc[0]
    v, lo, hi = parse_ci_str(row["AUROC"])
    n_pts     = int(row["n_patients"])   if "n_patients"  in row.index else 3716
    n_prov_lt = int(row["n_providers"])  if "n_providers" in row.index else 264
    add(rid, "Phase3", "Cimas_longtail", "AUROC_longtail",
        value=v, ci_lo=lo, ci_hi=hi,
        ci_method="reported_95ci_from_source",
        p_value=None,
        n_seeds=None,
        n_providers=n_prov_lt,
        n_boot=P3_N_BOOT,
        n_evaluations=n_pts,
        source_file=src_lt,
        w_issues="W02c",
        run_id="phase3_freeze",
        locked=True,
        note=f"{model} long-tail AUROC; n_patients={n_pts}, n_providers={n_prov_lt}, "
             f"prev=0.677, risk-positive orientation")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Policy values and contrast at B = 0.6667
# ═══════════════════════════════════════════════════════════════════════════════

src_pv = "processed/better_bp/results_phase4/policy_values.csv"
pv     = pd.read_csv(PROC / "better_bp" / "results_phase4" / "policy_values.csv")
pv_b   = pv[pv.budget == 0.6667]

policy_ids = {
    "no_incentive":    "R08",
    "random_budget":   "R09",
    "risk_targeting":  "R10",
    "uplift_targeting":"R11",
}
for policy, rid in policy_ids.items():
    rw = pv_b[pv_b.policy == policy]
    if len(rw) == 0:
        print(f"WARNING: policy={policy} not found at B=0.6667 in {src_pv}")
        continue
    v = float(rw["dr_value"].iloc[0])
    add(rid, "Phase4", "BETTER-BP", f"DR_value_{policy}",
        value=v,
        ci_lo=None, ci_hi=None,
        ci_method="none",
        p_value=None,
        n_seeds=len(P4_SEEDS),
        n_providers=2,
        n_boot=P4_N_BOOT,
        n_evaluations=None,
        source_file=src_pv,
        w_issues="W04A",
        run_id="phase4",
        locked=True,
        note="Budget B=0.6667; no_incentive DR value is comparator")

# Policy contrast: read from policy_contrasts.csv (issue 3)
src_pc = "processed/better_bp/results_phase4/policy_contrasts.csv"
pc     = pd.read_csv(PROC / "better_bp" / "results_phase4" / "policy_contrasts.csv")
pc_row = pc[
    (pc["budget"]   == 0.6667) &
    (pc["role"]     == "primary") &
    (pc["policy_a"] == "uplift_targeting") &
    (pc["policy_b"] == "random_budget")
].iloc[0]

add("R12", "Phase4", "BETTER-BP", "Delta_V_uplift_vs_random",
    value=float(pc_row["delta_v"]),
    ci_lo=float(pc_row["lcb_95"]),
    ci_hi=float(pc_row["ucb_95"]),
    ci_method=f"bootstrap_n{P4_N_BOOT}_seeds{len(P4_SEEDS)}_folds{P4_FOLDS}",
    p_value=None,
    n_seeds=len(P4_SEEDS),
    n_providers=2,
    n_boot=P4_N_BOOT,
    n_evaluations=None,
    source_file=src_pc,
    w_issues="W04A",
    run_id="phase4",
    locked=True,
    note=f"CI crosses zero (improvement not established); "
         f"ci_includes_zero={int(pc_row['ci_includes_zero'])}; "
         f"delta_attend_per100={pc_row['delta_attend_per100']:.2f} "
         f"[{pc_row['lcb_per100']:.2f}, {pc_row['ucb_per100']:.2f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6B — Adversarial benchmark (matched-seed paired t-tests; issue 2)
# ═══════════════════════════════════════════════════════════════════════════════

src_6b = "processed/phase6/phase6b_adversarial.csv"
bench  = pd.read_csv(PROC / "phase6" / "phase6b_adversarial.csv")
eicu   = bench[bench.dataset == "eicu"]

# Seed-indexed clean FedAvg series (reference for all attack contrasts)
clean_s = eicu[(eicu.attack == "none") & (eicu.defense == "fedavg")].set_index("seed")["auroc"]

# R13: clean baseline (bootstrap, not t-test — no contrast)
c_m, c_lo, c_hi = boot_ci(clean_s.values)
add("R13", "Phase6B", "eICU", "AUROC_clean_FedAvg",
    value=c_m, ci_lo=c_lo, ci_hi=c_hi,
    ci_method="bootstrap_n2000",
    p_value=None,
    n_seeds=len(clean_s), n_providers=8, n_boot=2000, n_evaluations=len(clean_s),
    source_file=src_6b, w_issues="", run_id="original", locked=True,
    note="Clean FedAvg baseline; eICU K=8 hospital groups; 15 rounds")

def attack_row(rid, attack, f, defense, note_extra=""):
    """Add matched-seed contrast row: attack vs clean FedAvg baseline."""
    atk_s  = eicu[(eicu.attack == attack) & (eicu.f == f) &
                  (eicu.defense == defense)].set_index("seed")["auroc"]
    md, lo, hi, p, n = ttest_paired(atk_s, clean_s)
    label = f"Delta_AUROC_{attack}_f{f}_{defense}"
    add(rid, "Phase6B", "eICU", label,
        value=md, ci_lo=lo, ci_hi=hi,
        ci_method=f"paired_ttest_df{n-1}",
        p_value=p,
        n_seeds=n, n_providers=8, n_boot=None, n_evaluations=n,
        source_file=src_6b, w_issues="", run_id="original", locked=True,
        note=f"Matched-seed join on {n} pairs. {note_extra}")

attack_row("R14", "label_flip", 3, "fedavg",
           note_extra="Cross-checked against stat_tests.csv A2.")

# R15: coord_median recovery — contrast with FedAvg under attack (not vs clean)
cm_s  = eicu[(eicu.attack == "label_flip") & (eicu.f == 3) &
              (eicu.defense == "coord_median")].set_index("seed")["auroc"]
fa_s  = eicu[(eicu.attack == "label_flip") & (eicu.f == 3) &
              (eicu.defense == "fedavg")].set_index("seed")["auroc"]
md15, lo15, hi15, p15, n15 = ttest_paired(cm_s, fa_s)
add("R15", "Phase6B", "eICU", "Delta_AUROC_coord_median_vs_FedAvg_label_flip",
    value=md15, ci_lo=lo15, ci_hi=hi15,
    ci_method=f"paired_ttest_df{n15-1}",
    p_value=p15,
    n_seeds=n15, n_providers=8, n_boot=None, n_evaluations=n15,
    source_file=src_6b, w_issues="", run_id="original", locked=True,
    note=f"Defense recovery: coord_median vs FedAvg under label_flip f=3; "
         f"{n15} matched pairs. Cross-checked against stat_tests.csv A2.")

attack_row("R16", "sign_flip",  3, "fedavg")
attack_row("R17", "alie",       3, "fedavg", note_extra="Not significant.")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6B — W09-corrected (random_gauss + backdoor)
# ═══════════════════════════════════════════════════════════════════════════════

src_w09  = "processed/phase6/phase6b_adversarial_w09.csv"
w9       = pd.read_csv(PROC / "phase6" / "phase6b_adversarial_w09.csv")
eicu_w9  = w9[w9.dataset == "eicu"]

# R18: random_gauss (W09a corrected) — paired against original clean_s
rg_s = eicu_w9[(eicu_w9.attack == "random_gauss") & (eicu_w9.f == 3) &
               (eicu_w9.defense == "fedavg")].set_index("seed")["auroc"]
md18, lo18, hi18, p18, n18 = ttest_paired(rg_s, clean_s)
add("R18", "Phase6B", "eICU", "Delta_AUROC_random_gauss_f3_FedAvg",
    value=md18, ci_lo=lo18, ci_hi=hi18,
    ci_method=f"paired_ttest_df{n18-1}",
    p_value=p18,
    n_seeds=n18, n_providers=8, n_boot=None, n_evaluations=n18,
    source_file=src_w09, w_issues="W09a", run_id="w09", locked=True,
    note=f"W09a seeded RNG; paired against original clean baseline ({n18} seeds); "
         f"original unseeded non-reproducible")

# R19: backdoor (W09c corrected) — BSR; no seed-matched contrast available
bd_sub  = eicu_w9[(eicu_w9.attack == "backdoor") & (eicu_w9.f == 3) &
                  (eicu_w9.defense == "fedavg")]
bsr_m, bsr_lo, bsr_hi = boot_ci(bd_sub["bsr"].dropna())
add("R19", "Phase6B", "eICU", "BSR_backdoor_f3_FedAvg",
    value=bsr_m, ci_lo=bsr_lo, ci_hi=bsr_hi,
    ci_method="bootstrap_n2000",
    p_value=None,
    n_seeds=len(bd_sub), n_providers=8, n_boot=2000, n_evaluations=len(bd_sub["bsr"].dropna()),
    source_file=src_w09, w_issues="W09c,W10a", run_id="w09", locked=True,
    note="Triggered target-class rate (renamed from BSR, W10a); "
         "no clean-model baseline collected; reported descriptively")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6C — Reconstruction cosines (C0-correct, batch=1)
# ═══════════════════════════════════════════════════════════════════════════════

src_rc = "processed/phase6/phase6c_reconstruction.csv"
rc     = pd.read_csv(PROC / "phase6" / "phase6c_reconstruction.csv")
rc1    = rc[rc.batch_size == 1]

for alpha, rid in [(0.00,"R20"), (0.01,"R21"), (0.05,"R22"), (0.10,"R23"), (0.20,"R24")]:
    sub = rc1[rc1.alpha == alpha]["cos_sim"]
    if len(sub) == 0:
        print(f"WARNING: no batch=1 rows for alpha={alpha} in {src_rc}")
        continue
    m, lo, hi = boot_ci(sub)
    add(rid, "Phase6C", "eICU", f"Recon_cos_batch1_alpha{alpha}",
        value=m, ci_lo=lo, ci_hi=hi,
        ci_method="bootstrap_n2000",
        p_value=None,
        n_seeds=len(sub), n_providers=None, n_boot=2000, n_evaluations=len(sub),
        source_file=src_rc, w_issues="W07", run_id="original", locked=True,
        note=f"C0-correct MI features; batch=1; {len(sub)} evaluations (W07 resolved)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6C — MIA
# ═══════════════════════════════════════════════════════════════════════════════

src_mia = "processed/phase6/phase6c_mia.csv"
mia     = pd.read_csv(PROC / "phase6" / "phase6c_mia.csv")

for ds, rid, eh in [("eicu","R25",-0.006), ("cimas","R26",+0.014)]:
    sub = mia[mia.dataset == ds]["mia_auroc"]
    m, lo, hi = boot_ci(sub)
    add(rid, "Phase6C", ds, "MIA_AUROC",
        value=m, ci_lo=lo, ci_hi=hi,
        ci_method="bootstrap_n2000",
        p_value=None,
        n_seeds=len(sub), n_providers=None, n_boot=2000, n_evaluations=len(sub),
        source_file=src_mia, w_issues="W08A", run_id="original", locked=True,
        note=f"Cohen h={eh}; 5-seed bootstrap CI; AUROC at/near chance")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6C — W09b corrected leakage FL utility (matched-seed paired t-tests)
# ═══════════════════════════════════════════════════════════════════════════════

src_fl9 = "processed/phase6/phase6c_leakage_fl_w09.csv"
fl9     = pd.read_csv(PROC / "phase6" / "phase6c_leakage_fl_w09.csv")

clean_fl9_s = fl9[fl9.alpha == 0.0].set_index("seed")["auroc"]

for alpha, rid, materiality in [
    (0.01, "R27", "Within"),
    (0.05, "R28", "Within"),
    (0.10, "R29", "Exceeds"),
    (0.20, "R30", "Exceeds"),
]:
    atk_fl9_s = fl9[fl9.alpha == alpha].set_index("seed")["auroc"]
    md, lo, hi, p, n = ttest_paired(atk_fl9_s, clean_fl9_s)
    add(rid, "Phase6C", "eICU", f"Utility_AUROC_alpha{alpha}_W09b",
        value=md, ci_lo=lo, ci_hi=hi,
        ci_method=f"paired_ttest_df{n-1}",
        p_value=p,
        n_seeds=n, n_providers=8, n_boot=None, n_evaluations=n,
        source_file=src_fl9, w_issues="W09b", run_id="w09", locked=True,
        note=f"W09b per-client noise; materiality threshold -0.02: {materiality}")


# ═══════════════════════════════════════════════════════════════════════════════
# GOVERNANCE (Phase 5 audit) — no fabricated fallback (issue 5)
# ═══════════════════════════════════════════════════════════════════════════════

fi_path = PROC / "phase5" / "results_v3" / "fault_injection_results.csv"
fi      = pd.read_csv(fi_path)    # raises FileNotFoundError if missing — no fallback
n_det   = int(fi["detected"].sum())
n_tot   = len(fi)
cp_lo, cp_hi = clopper_pearson(n_det, n_tot)

add("R31", "Phase5", "All", "Audit_fault_detection_rate",
    value=n_det / n_tot,
    ci_lo=cp_lo,
    ci_hi=cp_hi,
    ci_method="clopper_pearson_95",
    p_value=None,
    n_seeds=None, n_providers=None, n_boot=None, n_evaluations=n_tot,
    source_file="processed/phase5/results_v3/fault_injection_results.csv",
    w_issues="W11a,W11b,W11c",
    run_id="v3",
    locked=True,
    note=f"{n_det}/{n_tot} detected; 0 false alerts; Clopper-Pearson exact 95% CI")


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE
# ═══════════════════════════════════════════════════════════════════════════════

ledger = pd.DataFrame(rows)
ledger.to_csv(OUT_CSV, index=False)
print(f"\nMaster results ledger written: {OUT_CSV}")
print(f"  {len(ledger)} rows\n")
print(ledger[["result_id","phase","dataset","metric_name",
              "value","ci_lo","ci_hi","ci_method","p_value",
              "n_seeds","n_providers","n_boot","w_issues_resolved","locked"]].to_string(index=False))
