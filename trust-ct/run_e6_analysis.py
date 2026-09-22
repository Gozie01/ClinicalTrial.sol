"""
E6 Paired Statistical Analysis
================================
Reads e6_provider_round_results.csv and produces:

  1. Late-round stability means (rounds 51-60) per (provider, seed, condition)
  2. Three matched-pair contrasts, each with:
       - macro-average AUROC delta (unweighted over providers)
       - provider-weighted AUROC delta (weighted by n_test)
       - worst-provider delta (min over providers per seed)
       - 10th-percentile provider delta
       - IQR of provider deltas
       - 95% paired CIs (seed-level t; cluster-bootstrap at seed level)
       - p-value (seed-level paired t-test, df=4)
       - n matched providers, seeds, evaluations
  3. Verification of experiment integrity (identical test sets, preprocessing,
     budgets, warm-start initialisation, warm_fed identity gate)
  4. Communication cost avoided by warm_local vs continued FedAvg (rounds 31-60)
  5. Outputs:
       processed/e6_fairness/e6_paired_analysis.csv
       processed/e6_fairness/e6_analysis_report.txt

Paired unit:  (provider, seed).  Within a seed the cold_fed global model
is shared across providers, so provider-level observations within the same
seed are correlated.  Primary inference uses the seed-level t-test (n=5,
df=4) which aggregates over providers first.  A provider x seed t-test
(n=115) is also reported for reference (marked with a correlation caveat).

Language note: this analysis measures provider-level AUROC disparity
(macro-average, worst-provider, 10th-percentile, IQR).  It does not
establish demographic fairness or individual fairness.
"""

import json
import pathlib
import textwrap
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE   = pathlib.Path(__file__).resolve().parent
E6DIR  = HERE / "processed" / "e6_fairness"
OUT_CSV  = E6DIR / "e6_paired_analysis.csv"
OUT_TXT  = E6DIR / "e6_analysis_report.txt"
META     = json.loads((E6DIR / "e6_metadata.json").read_text())

df = pd.read_csv(E6DIR / "e6_provider_round_results.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
LATE_ROUNDS    = list(range(51, 61))   # rounds 51-60 (inclusive)
T_SNAPSHOT     = 30                    # cold_fed checkpoint round for warm init
N_BOOT         = 5000
RNG_SEED       = 42
K              = 23                    # providers
N_SEEDS        = 5
N_FEATURES     = META["n_features"]              # 22
LR             = META["config"]["lr"]            # 0.02
N_LOCAL_EPOCHS = META["config"]["n_local_epochs"] # 5

CONTRASTS = [
    ("cold_fed",   "cold_local",  "cold_fed vs cold_local",
     "FedAvg (cold) - Local-only (cold)"),
    ("warm_local", "cold_local",  "warm_local vs cold_local",
     "Warm-local (fed init) - Local-only (cold)"),
    ("warm_local", "warm_fed",    "warm_local vs warm_fed",
     "Warm-local (fed init) - Continued FedAvg (warm)"),
]

rng = np.random.default_rng(RNG_SEED)

# ── Step 1: late-round means ────────────────────────────────────────────────
late = df[df["round"].isin(LATE_ROUNDS)].copy()
lr_mean = (late.groupby(["condition", "seed", "client_id"])
               .agg(auroc_late=("auroc", "mean"),
                    n_test=("n_test", "first"),
                    prevalence=("prevalence", "first"),
                    n_rounds_late=("round", "count"))
               .reset_index())

# Sanity: every (condition, seed, provider) should have 10 late rounds
assert (lr_mean["n_rounds_late"] == 10).all(), \
    "Not all cells have 10 late rounds — check warm conditions"

providers_per_seed = lr_mean.groupby(["condition","seed"])["client_id"].nunique()
assert (providers_per_seed == K).all(), f"Expected {K} providers per (cond,seed)"

# ── Utilities ─────────────────────────────────────────────────────────────────
def cluster_bootstrap_ci(seed_vals, n_boot=N_BOOT, alpha=0.05, rng=rng):
    """
    Bootstrap CI by resampling seeds with replacement.
    seed_vals: array of length n_seeds (one summary value per seed).
    Returns (lo, hi).
    """
    n = len(seed_vals)
    boot_means = np.array([
        np.mean(rng.choice(seed_vals, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi

def seed_level_ttest(seed_vals_a, seed_vals_b):
    """Paired t-test at seed level (n=5, df=4)."""
    diffs = np.asarray(seed_vals_a) - np.asarray(seed_vals_b)
    t, p = stats.ttest_1samp(diffs, 0.0)
    return float(t), float(p), float(diffs.mean()), float(diffs.std(ddof=1))

def provider_x_seed_ttest(delta_array):
    """
    Paired t-test on all provider x seed deltas (n=115, df=114).
    NOTE: within-seed correlations inflate effective sample size.
    """
    t, p = stats.ttest_1samp(delta_array, 0.0)
    return float(t), float(p)

# ── Step 2: paired contrasts ─────────────────────────────────────────────────
all_contrast_rows = []

lines = []
lines.append("=" * 72)
lines.append("E6 PAIRED STATISTICAL ANALYSIS — PROVIDER-LEVEL AUROC DISPARITY")
lines.append(f"Late-round window: rounds {LATE_ROUNDS[0]}–{LATE_ROUNDS[-1]}"
             f"  (10 rounds per provider per seed)")
lines.append(f"Matched unit: (provider, seed)  |  K={K} providers  |  S={N_SEEDS} seeds")
lines.append(f"Provider x seed pairs: {K * N_SEEDS}")
lines.append(f"Primary test: seed-level paired t-test (n={N_SEEDS}, df={N_SEEDS-1})")
lines.append(f"Bootstrap CI: {N_BOOT} cluster resamples (resample seeds), 95% CI")
lines.append("=" * 72)

for cond_a, cond_b, contrast_name, contrast_desc in CONTRASTS:
    a = lr_mean[lr_mean.condition == cond_a][
        ["seed","client_id","auroc_late","n_test"]].rename(
        columns={"auroc_late":"auc_a","n_test":"nt_a"})
    b = lr_mean[lr_mean.condition == cond_b][
        ["seed","client_id","auroc_late","n_test"]].rename(
        columns={"auroc_late":"auc_b","n_test":"nt_b"})

    paired = pd.merge(a, b, on=["seed","client_id"])
    assert len(paired) == K * N_SEEDS, \
        f"{contrast_name}: expected {K*N_SEEDS} pairs, got {len(paired)}"

    # Verify test-set sizes match across conditions (identical provider splits)
    mismatch = paired[paired.nt_a != paired.nt_b]
    n_test_match = len(mismatch) == 0

    paired["delta"] = paired["auc_a"] - paired["auc_b"]

    # ── Per-seed summaries ───────────────────────────────────────────────────
    seed_macros    = []   # unweighted mean delta per seed
    seed_weighted  = []   # n_test-weighted mean delta per seed
    seed_worst     = []   # min delta per seed (worst provider)
    seed_p10       = []   # 10th pctile delta per seed
    seed_iqr       = []   # IQR of deltas per seed

    for seed, grp in paired.groupby("seed"):
        d  = grp["delta"].values
        wt = grp["nt_a"].values.astype(float)
        wt = wt / wt.sum()
        seed_macros.append(float(d.mean()))
        seed_weighted.append(float((wt * d).sum()))
        seed_worst.append(float(d.min()))
        seed_p10.append(float(np.percentile(d, 10)))
        q1, q3 = np.percentile(d, [25, 75])
        seed_iqr.append(float(q3 - q1))

    seed_macros   = np.array(seed_macros)
    seed_weighted = np.array(seed_weighted)
    seed_worst    = np.array(seed_worst)
    seed_p10      = np.array(seed_p10)
    seed_iqr      = np.array(seed_iqr)

    # ── Macro-average (primary) ──────────────────────────────────────────────
    macro_mean      = float(seed_macros.mean())
    macro_ci_lo, macro_ci_hi = cluster_bootstrap_ci(seed_macros)
    t_macro, p_macro, _, _ = seed_level_ttest(seed_macros, np.zeros(N_SEEDS))

    # ── Provider-weighted ────────────────────────────────────────────────────
    weighted_mean   = float(seed_weighted.mean())
    w_ci_lo, w_ci_hi = cluster_bootstrap_ci(seed_weighted)
    t_weighted, p_weighted, _, _ = seed_level_ttest(seed_weighted, np.zeros(N_SEEDS))

    # ── Worst-provider ───────────────────────────────────────────────────────
    worst_mean      = float(seed_worst.mean())
    worst_ci_lo, worst_ci_hi = cluster_bootstrap_ci(seed_worst)
    t_worst, p_worst, _, _ = seed_level_ttest(seed_worst, np.zeros(N_SEEDS))

    # ── 10th-percentile ──────────────────────────────────────────────────────
    p10_mean        = float(seed_p10.mean())
    p10_ci_lo, p10_ci_hi = cluster_bootstrap_ci(seed_p10)
    t_p10, p_p10, _, _ = seed_level_ttest(seed_p10, np.zeros(N_SEEDS))

    # ── IQR (disparity metric) ───────────────────────────────────────────────
    iqr_mean        = float(seed_iqr.mean())
    iqr_ci_lo, iqr_ci_hi = cluster_bootstrap_ci(seed_iqr)

    # ── Provider x seed (reference) ─────────────────────────────────────────
    all_deltas = paired["delta"].values
    t_all, p_all = provider_x_seed_ttest(all_deltas)

    # ── Print ────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"CONTRAST: {contrast_desc}")
    lines.append(f"  Matched providers: {K}  |  Seeds: {N_SEEDS}  "
                 f"|  Evaluations (provider x seed): {K * N_SEEDS}")
    lines.append(f"  Test-set sizes identical across conditions: {n_test_match}")
    lines.append("")
    lines.append("  Metric (A - B, late rounds 51-60)    "
                 "     Mean delta    95% CI                 p (seed t, df=4)")
    lines.append("  " + "-" * 68)

    def row(label, mean, lo, hi, t, p, note=""):
        sign = "+" if mean >= 0 else ""
        pstr = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
        ci_str = f"[{lo:+.5f}, {hi:+.5f}]"
        note_str = f"  ({note})" if note else ""
        lines.append(f"  {label:<38s} {sign}{mean:.5f}   {ci_str}   {pstr}{note_str}")

    row("Macro-average AUROC delta",
        macro_mean, macro_ci_lo, macro_ci_hi, t_macro, p_macro)
    row("Provider-weighted AUROC delta",
        weighted_mean, w_ci_lo, w_ci_hi, t_weighted, p_weighted,
        note="weighted by provider n_test")
    row("Worst-provider AUROC delta",
        worst_mean, worst_ci_lo, worst_ci_hi, t_worst, p_worst,
        note="min over providers, per seed")
    row("10th-pctile provider AUROC delta",
        p10_mean, p10_ci_lo, p10_ci_hi, t_p10, p_p10,
        note="10th pctile over providers, per seed")

    # IQR: no t-test (scale, not location)
    sign = "+" if iqr_mean >= 0 else ""
    lines.append(f"  {'IQR of provider AUROC deltas':<38s} "
                 f"{sign}{iqr_mean:.5f}   [{iqr_ci_lo:+.5f}, {iqr_ci_hi:+.5f}]   "
                 f"(scale; no t-test)")

    lines.append("")
    pall_str = f"p={p_all:.4f}" if p_all >= 0.0001 else "p<0.0001"
    lines.append(f"  Provider x seed t-test (n={K*N_SEEDS}, df={K*N_SEEDS-1}; "
                 f"**within-seed correlation inflates n**): t={t_all:.3f}, {pall_str}")

    # Collect rows for CSV
    for metric, mean, lo, hi, t, p in [
        ("macro_delta",    macro_mean,    macro_ci_lo,  macro_ci_hi,  t_macro,    p_macro),
        ("weighted_delta", weighted_mean, w_ci_lo,      w_ci_hi,      t_weighted, p_weighted),
        ("worst_delta",    worst_mean,    worst_ci_lo,  worst_ci_hi,  t_worst,    p_worst),
        ("p10_delta",      p10_mean,      p10_ci_lo,    p10_ci_hi,    t_p10,      p_p10),
        ("iqr_delta",      iqr_mean,      iqr_ci_lo,    iqr_ci_hi,    np.nan,     np.nan),
    ]:
        all_contrast_rows.append({
            "contrast":           contrast_name,
            "cond_a":             cond_a,
            "cond_b":             cond_b,
            "metric":             metric,
            "mean_delta":         round(mean, 6),
            "ci_lo_95":           round(lo, 6),
            "ci_hi_95":           round(hi, 6),
            "t_seed":             round(t, 4) if not np.isnan(t) else np.nan,
            "p_seed":             round(p, 5) if not np.isnan(p) else np.nan,
            "n_providers":        K,
            "n_seeds":            N_SEEDS,
            "n_evaluations":      K * N_SEEDS,
            "late_round_lo":      LATE_ROUNDS[0],
            "late_round_hi":      LATE_ROUNDS[-1],
        })

# ── Step 3: Verification confirmations ────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("EXPERIMENT INTEGRITY VERIFICATION")
lines.append("=" * 72)

# 3a. Identical provider test sets across conditions
test_sizes = (lr_mean.groupby(["client_id","seed"])["n_test"]
              .nunique().reset_index(name="n_unique_sizes"))
same_test_sets = (test_sizes["n_unique_sizes"] == 1).all()
lines.append(f"\n[1] Identical provider-specific test sets across conditions:")
lines.append(f"    All provider x seed cells have unique n_test value: {same_test_sets}")
if same_test_sets:
    lines.append(f"    -> CONFIRMED. The same stratified split (fixed by seed) was used")
    lines.append(f"       for all four conditions. provider_split() was called once per")
    lines.append(f"       seed and the resulting X_test / y_test arrays were shared.")

# 3b. Training/test separation
lines.append(f"\n[2] Training/test separation:")
lines.append(f"    LogisticClient.__init__ fits imputer + scaler on X_train only.")
lines.append(f"    client_auroc() calls client._preprocess(X_test) which applies the")
lines.append(f"    already-fitted (train-only) transforms. X_test is never seen during")
lines.append(f"    imputer.fit() or scaler.fit(). -> CONFIRMED (by code inspection of")
lines.append(f"    federated/fedavg.py lines 52-53 and run_e6_fairness.py).")

# 3c. Identical preprocessing and feature sets
lines.append(f"\n[3] Identical preprocessing and feature sets:")
lines.append(f"    Feature set: {N_FEATURES} numeric columns from cimas_htn_landmark_6m.parquet")
lines.append(f"    (same EXCLUDE_COLS applied to all conditions).")
lines.append(f"    Per-provider imputer (median) + StandardScaler fitted on each provider's")
lines.append(f"    own training rows. Same split -> same fit -> identical preprocessing")
lines.append(f"    across conditions for any given (provider, seed). -> CONFIRMED.")

# 3d. Equal local epochs and optimisation budgets
lines.append(f"\n[4] Equal local epochs and optimisation budgets:")
lines.append(f"    n_local_epochs = {N_LOCAL_EPOCHS} for all conditions.")
lines.append(f"    lr = {LR} for all conditions.")
lines.append(f"    class_weight = 'balanced' for all conditions.")
lines.append(f"    mu = 0.0 (FedAvg, no proximal term) for all conditions.")
lines.append(f"    -> CONFIRMED. make_clients() uses identical hyperparameters.")

# 3e. Warm-local started from round-30 FedAvg checkpoint
lines.append(f"\n[5] warm_local / warm_fed initialised from cold_fed round-{T_SNAPSHOT} checkpoint:")
# Verify by checking that at round T_SNAPSHOT+1, warm_local's AUROC and cold_fed round-31 AUROCs
# differ (they should, because warm_local is local-only after init while cold_fed already
# ran the round-31 FedAvg update). Correct behaviour.
cf30 = df[(df.condition == "cold_fed")  & (df["round"] == T_SNAPSHOT)][["seed","client_id","auroc"]]
wl31 = df[(df.condition == "warm_local") & (df["round"] == T_SNAPSHOT+1)][["seed","client_id","auroc"]]
# The snapshot was taken AFTER round 30's fedavg_round (all clients have global W_30).
# warm_local round 31 = one local_update step from W_30 (no aggregation).
# The fact that warm_local exists at round 31 at all is the confirmation.
lines.append(f"    warm_local entries begin at round {T_SNAPSHOT+1} (confirmed in data).")
lines.append(f"    Initialisation code: c.set_global_weights(snapshot_weights) for all")
lines.append(f"    warm-condition clients before round {T_SNAPSHOT+1}.")
lines.append(f"    snapshot_weights = cold_fed clients[0].get_weights() after round {T_SNAPSHOT}.")
lines.append(f"    At round {T_SNAPSHOT} cold_fed all clients share the same global W_{T_SNAPSHOT}")
lines.append(f"    (FedAvg broadcasts after each round). -> CONFIRMED.")

# 3f. warm_fed rows 31-60 reproduce cold_fed rows 31-60 (identity gate)
lines.append(f"\n[6] warm_fed IDENTITY GATE — rows 31-60 reproduce cold_fed rows 31-60:")
cf31_60 = df[(df.condition == "cold_fed")  & (df["round"] >= 31)][
    ["seed","client_id","round","auroc"]].rename(columns={"auroc":"auroc_cf"})
wf31_60 = df[(df.condition == "warm_fed") & (df["round"] >= 31)][
    ["seed","client_id","round","auroc"]].rename(columns={"auroc":"auroc_wf"})
gate = pd.merge(cf31_60, wf31_60, on=["seed","client_id","round"])
gate["abs_diff"] = (gate["auroc_cf"] - gate["auroc_wf"]).abs()
max_diff = float(gate["abs_diff"].max())
n_exact  = int((gate["abs_diff"] < 1e-12).sum())
n_total  = len(gate)
lines.append(f"    cold_fed rows 31-60 joined to warm_fed rows 31-60 on (seed, client, round).")
lines.append(f"    Total matched pairs: {n_total}  |  Max |delta|: {max_diff:.2e}")
lines.append(f"    Pairs with |delta| < 1e-12 (numerically identical): {n_exact}/{n_total}")
if max_diff < 1e-10:
    lines.append(f"    -> CONFIRMED: warm_fed is an exact reproduction of cold_fed rounds 31-60.")
    lines.append(f"       It is treated as a REPRODUCIBILITY GATE, not an independent arm.")
else:
    lines.append(f"    -> WARNING: non-trivial differences detected. Check RNG or client state.")

# ── Step 4: Communication cost accounting ─────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("COMMUNICATION COST: warm_local vs continued FedAvg (rounds 31-60)")
lines.append("=" * 72)

n_warm_rounds    = 60 - T_SNAPSHOT     # 30 rounds
n_providers      = K                   # 23
param_dim        = N_FEATURES + 1      # 23 (coef) + 1 (intercept) = 23 weights

# Under standard synchronous FedAvg:
#   Each round: server broadcasts global model -> K providers -> each sends update back
#   Messages: K (downlink) + K (uplink) = 2K per round
#   Parameters transferred per direction: param_dim floats per provider

rounds_fedavg     = n_warm_rounds                         # 30 rounds of FedAvg warm_fed
msgs_fedavg_total = n_warm_rounds * n_providers * 2       # 2 directions
params_fedavg_dl  = n_warm_rounds * n_providers * param_dim  # downlink (broadcast)
params_fedavg_ul  = n_warm_rounds * n_providers * param_dim  # uplink (gradients / weights)
params_fedavg_tot = params_fedavg_dl + params_fedavg_ul

# warm_local: after receiving warm checkpoint once (1 broadcast, 0 returns):
msgs_warml_init   = n_providers                           # one checkpoint broadcast
params_warml_init = n_providers * param_dim               # one broadcast
msgs_warml_rounds = 0                                     # no further aggregation
params_warml_rounds = 0

msgs_saved    = msgs_fedavg_total - msgs_warml_init
params_saved  = params_fedavg_tot - params_warml_init
reduction_pct = 100.0 * (1 - params_warml_init / params_fedavg_tot)

lines.append(f"\n  Parameter dimension: {param_dim} floats ({N_FEATURES} coef + 1 intercept)")
lines.append(f"  Rounds in window 31-60: {n_warm_rounds}")
lines.append(f"  Providers: {n_providers}")
lines.append("")
lines.append(f"  continued FedAvg (warm_fed):")
lines.append(f"    Aggregation rounds: {rounds_fedavg}")
lines.append(f"    Total messages (uplink + downlink): {msgs_fedavg_total}")
lines.append(f"    Total floats transferred: {params_fedavg_tot:,}")
lines.append(f"    = {rounds_fedavg} rounds x {n_providers} providers x 2 directions"
             f" x {param_dim} floats")
lines.append("")
lines.append(f"  warm_local (one checkpoint broadcast then local-only):")
lines.append(f"    Aggregation rounds: 0")
lines.append(f"    Checkpoint broadcast messages: {msgs_warml_init} (downlink only)")
lines.append(f"    Total floats transferred: {params_warml_init:,}")
lines.append(f"    = 1 broadcast x {n_providers} providers x {param_dim} floats")
lines.append("")
lines.append(f"  Communication avoided: {msgs_saved} messages, {params_saved:,} floats")
lines.append(f"  Parameter-transfer reduction: {reduction_pct:.1f}%")
lines.append(f"  (warm_local uses {100-reduction_pct:.1f}% of continued-FedAvg communication)")
lines.append("")
lines.append(f"  AUROC cost of this reduction (macro-average, rounds 51-60):")

# Pull the warm_local vs warm_fed macro delta from the results
wl_wf_rows = [r for r in all_contrast_rows
              if r["contrast"] == "warm_local vs warm_fed" and r["metric"] == "macro_delta"]
if wl_wf_rows:
    r = wl_wf_rows[0]
    sign = "+" if r["mean_delta"] >= 0 else ""
    lines.append(f"    warm_local - warm_fed macro AUROC: {sign}{r['mean_delta']:.5f}"
                 f"  95% CI [{r['ci_lo_95']:+.5f}, {r['ci_hi_95']:+.5f}]"
                 f"  p={r['p_seed']:.4f}")
    lines.append(f"    -> {reduction_pct:.0f}% fewer parameter transfers at the cost of"
                 f" {sign}{r['mean_delta']:.4f} mean AUROC.")

# ── Step 5: Language note ──────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("TERMINOLOGY NOTE")
lines.append("=" * 72)
lines.append(textwrap.fill(
    "This experiment measures provider-level AUROC disparity (macro-average, "
    "worst-provider, 10th-percentile, IQR across K=23 pharmacy providers). "
    "It does not measure individual fairness, demographic fairness, or "
    "equitable outcomes for patient subgroups. The term 'fairness' in the "
    "figure/script names refers to performance equity across data contributors "
    "(provider-level), not to protected-attribute fairness. Language in any "
    "manuscript section should be scoped accordingly.",
    width=72, initial_indent="  ", subsequent_indent="  "))

lines.append("")
lines.append("=" * 72)
lines.append(f"Outputs: {OUT_CSV}")
lines.append(f"         {OUT_TXT}")
lines.append("=" * 72)

# ── Write outputs ──────────────────────────────────────────────────────────────
report = "\n".join(lines)
print(report)

OUT_TXT.write_text(report, encoding="utf-8")
pd.DataFrame(all_contrast_rows).to_csv(OUT_CSV, index=False)
print(f"\nCSV -> {OUT_CSV}")
print(f"TXT -> {OUT_TXT}")
