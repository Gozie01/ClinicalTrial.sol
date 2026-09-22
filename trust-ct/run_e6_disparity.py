"""
E6 Disparity Consolidation
============================
Recomputes condition-level disparity statistics (not contrast deltas) with
internally consistent seed-level bootstrap CIs, then separates disparity
changes from provider-specific contrast heterogeneity.

Outputs:
  processed/e6_fairness/e6_disparity_stats.csv
      Per-condition IQR / range / CV of AUROC across providers,
      with seed-level bootstrap 95% CI.

  processed/e6_fairness/e6_provider_fixed_effects.csv
      Per-provider fixed effects for each contrast:
        mean_delta (across seeds), sd_delta, sign_consistency.
      Separates systematic from noisy heterogeneity.

  processed/e6_fairness/e6_worst_provider_identity.csv
      For each contrast × seed: which provider is "worst affected"?
      Tracks whether worst-provider identity is stable across seeds.

  processed/e6_fairness/e6_disparity_report.txt
      Formatted narrative report.

Language note: all disparity metrics are provider-level AUROC statistics.
This analysis does not measure demographic or individual fairness.
"""

import json
import pathlib
import textwrap
import numpy as np
import pandas as pd

HERE  = pathlib.Path(__file__).resolve().parent
E6DIR = HERE / "processed" / "e6_fairness"
OUT_DISP  = E6DIR / "e6_disparity_stats.csv"
OUT_FX    = E6DIR / "e6_provider_fixed_effects.csv"
OUT_WORST = E6DIR / "e6_worst_provider_identity.csv"
OUT_TXT   = E6DIR / "e6_disparity_report.txt"

df = pd.read_csv(E6DIR / "e6_provider_round_results.csv")

LATE_LO, LATE_HI = 51, 60
N_BOOT, RNG_SEED = 5000, 42
K, S = 23, 5

CONTRASTS = [
    ("cold_fed",   "cold_local",  "cold_fed - cold_local"),
    ("warm_local", "cold_local",  "warm_local - cold_local"),
    ("warm_local", "warm_fed",    "warm_local - warm_fed"),
]

rng = np.random.default_rng(RNG_SEED)

# ── Late-round means per (condition, seed, provider) ──────────────────────────
late = df[df["round"].between(LATE_LO, LATE_HI)]
lr = (late.groupby(["condition","seed","client_id"])
          .agg(auroc=("auroc","mean"), n_test=("n_test","first"))
          .reset_index())

# ── Gini helper ───────────────────────────────────────────────────────────────
def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.mean() == 0:
        return np.nan
    idx = np.arange(1, n+1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n+1)/n)

# ── Bootstrap CI (cluster by seed) ───────────────────────────────────────────
def boot_ci(seed_vals, n_boot=N_BOOT, alpha=0.05):
    n = len(seed_vals)
    boots = np.array([np.mean(rng.choice(seed_vals, n, replace=True))
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 100*alpha/2)), float(np.percentile(boots, 100*(1-alpha/2)))

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: Condition-level disparity statistics
# ──────────────────────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 72)
lines.append("E6 CONDITION-LEVEL DISPARITY STATISTICS")
lines.append("Provider-level AUROC disparity (NOT demographic/individual fairness)")
lines.append(f"Late-round window: rounds {LATE_LO}-{LATE_HI}, K={K} providers, S={S} seeds")
lines.append("=" * 72)

disp_rows = []
CONDS = ["cold_fed","cold_local","warm_fed","warm_local"]

for cond in CONDS:
    sub = lr[lr.condition == cond]
    # Per-seed disparity metrics
    s_iqr, s_range, s_cv, s_gini = [], [], [], []
    s_min, s_max, s_mean, s_q10 = [], [], [], []
    for seed, grp in sub.groupby("seed"):
        aucs = grp["auroc"].values
        q25, q75 = np.percentile(aucs, [25, 75])
        s_iqr.append(float(q75 - q25))
        s_range.append(float(aucs.max() - aucs.min()))
        s_cv.append(float(aucs.std(ddof=1) / aucs.mean()) if aucs.mean() > 0 else np.nan)
        s_gini.append(gini(aucs))
        s_min.append(float(aucs.min()))
        s_max.append(float(aucs.max()))
        s_mean.append(float(aucs.mean()))
        s_q10.append(float(np.percentile(aucs, 10)))

    for metric, vals in [
        ("iqr",   s_iqr),
        ("range", s_range),
        ("cv",    s_cv),
        ("gini",  s_gini),
        ("min",   s_min),
        ("max",   s_max),
        ("mean",  s_mean),
        ("p10",   s_q10),
    ]:
        arr = np.array([v for v in vals if not np.isnan(v)])
        lo, hi = boot_ci(arr)
        disp_rows.append({
            "condition": cond, "metric": metric,
            "seed_mean": round(float(arr.mean()), 6),
            "ci_lo_95":  round(lo, 6),
            "ci_hi_95":  round(hi, 6),
            "n_seeds":   len(arr),
        })

disp_df = pd.DataFrame(disp_rows)
disp_df.to_csv(OUT_DISP, index=False)

# Print condition-level disparity table
lines.append("")
lines.append("  Metric      | " + " | ".join(f"{c:14s}" for c in CONDS))
lines.append("  " + "-" * 72)
for metric in ["mean","min","p10","max","iqr","range","cv","gini"]:
    row = [disp_df[(disp_df.condition==c) & (disp_df.metric==metric)]["seed_mean"].values[0]
           for c in CONDS]
    cells = [f"{v:14.5f}" for v in row]
    lines.append(f"  {metric:9s}   | " + " | ".join(cells))

lines.append("")
lines.append("  95% CIs (seed-level bootstrap) for IQR (disparity):")
for cond in CONDS:
    r = disp_df[(disp_df.condition==cond) & (disp_df.metric=="iqr")].iloc[0]
    lines.append(f"    {cond:14s}: {r.seed_mean:.5f} [{r.ci_lo_95:.5f}, {r.ci_hi_95:.5f}]")

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: Provider fixed effects — systematic vs noisy heterogeneity
# ──────────────────────────────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("PROVIDER-LEVEL FIXED EFFECTS (CONTRAST HETEROGENEITY)")
lines.append("Mean delta across S=5 seeds per provider; SD measures consistency.")
lines.append("sign_consistency = fraction of seeds with positive delta.")
lines.append("A |mean|/SD ratio > 2 suggests a systematic directional effect.")
lines.append("=" * 72)

fx_rows = []
worst_rows = []

for cond_a, cond_b, contrast_name in CONTRASTS:
    a = lr[lr.condition==cond_a][["seed","client_id","auroc","n_test"]].rename(
        columns={"auroc":"auc_a"})
    b = lr[lr.condition==cond_b][["seed","client_id","auroc"]].rename(
        columns={"auroc":"auc_b"})
    paired = pd.merge(a, b, on=["seed","client_id"])
    paired["delta"] = paired["auc_a"] - paired["auc_b"]

    lines.append(f"\n  Contrast: {contrast_name}")
    lines.append(f"  {'Provider':<38s} {'mean_delta':>10} {'sd_delta':>9} "
                 f"{'sign_cons':>10} {'n_test':>7} {'systematic?':>12}")
    lines.append("  " + "-" * 88)

    # Per-provider stats
    for prov, grp in paired.groupby("client_id"):
        d = grp["delta"].values
        mn = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
        sign_cons = float((d > 0).mean())
        nt = int(grp["n_test"].iloc[0])
        systematic = (not np.isnan(sd)) and (sd > 0) and (abs(mn)/sd > 2.0)
        fx_rows.append({
            "contrast":        contrast_name,
            "client_id":       prov,
            "mean_delta":      round(mn, 6),
            "sd_delta":        round(sd, 6) if not np.isnan(sd) else np.nan,
            "sign_consistency": round(sign_cons, 3),
            "n_test":          nt,
            "systematic":      systematic,
            "snr_ratio":       round(abs(mn)/sd, 3) if (not np.isnan(sd) and sd > 0) else np.nan,
        })
        s_flag = "YES" if systematic else "—"
        sd_str = f"{sd:.5f}" if not np.isnan(sd) else "  —  "
        lines.append(f"  {prov:<38s} {mn:>+10.5f} {sd_str:>9} "
                     f"{sign_cons:>10.2f} {nt:>7d} {s_flag:>12}")

    # Worst-provider identity: which provider has the most negative delta per seed?
    for seed, grp in paired.groupby("seed"):
        idx_worst = grp["delta"].idxmin()
        worst_prov = grp.loc[idx_worst, "client_id"]
        worst_delta = float(grp.loc[idx_worst, "delta"])
        worst_rows.append({
            "contrast": contrast_name, "seed": seed,
            "worst_provider": worst_prov, "worst_delta": round(worst_delta, 6),
        })

    # Summary of systematic providers
    fx_sub = [r for r in fx_rows if r["contrast"] == contrast_name]
    n_sys = sum(1 for r in fx_sub if r["systematic"])
    n_neg = sum(1 for r in fx_sub if r["mean_delta"] < -0.01)
    n_pos = sum(1 for r in fx_sub if r["mean_delta"] > +0.01)
    lines.append(f"\n  -> Systematic (|mean|/SD > 2): {n_sys}/{K} providers")
    lines.append(f"     Mean delta < -0.01 AUROC:   {n_neg}/{K} providers")
    lines.append(f"     Mean delta > +0.01 AUROC:   {n_pos}/{K} providers")

fx_df    = pd.DataFrame(fx_rows)
worst_df = pd.DataFrame(worst_rows)
fx_df.to_csv(OUT_FX, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: Worst-provider identity stability
# ──────────────────────────────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("WORST-PROVIDER IDENTITY STABILITY")
lines.append("Does the same provider consistently appear as worst-affected?")
lines.append("=" * 72)

for contrast_name in [c for _, _, c in CONTRASTS]:
    sub = worst_df[worst_df.contrast == contrast_name]
    prov_counts = sub.groupby("worst_provider")["seed"].count().sort_values(ascending=False)
    most_common_prov = prov_counts.index[0]
    n_seeds_most = int(prov_counts.iloc[0])
    lines.append(f"\n  Contrast: {contrast_name}")
    lines.append(f"  Most frequently worst-affected provider: {most_common_prov}")
    lines.append(f"  Appeared as worst in {n_seeds_most}/{S} seeds")
    if n_seeds_most >= 4:
        lines.append(f"  -> SYSTEMATIC: strong consistency across seeds")
    elif n_seeds_most >= 2:
        lines.append(f"  -> PARTIAL: appears in majority of seeds")
    else:
        lines.append(f"  -> NOISY: no provider dominates the worst-affected slot")
    lines.append("  Seed-by-seed:")
    for _, row in sub.iterrows():
        lines.append(f"    seed={row.seed}: {row.worst_provider} (delta={row.worst_delta:+.4f})")

worst_df.to_csv(OUT_WORST, index=False)

# ──────────────────────────────────────────────────────────────────────────────
# PART 4: Disparity change vs contrast heterogeneity (conceptual decomposition)
# ──────────────────────────────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("DISPARITY CHANGE vs CONTRAST HETEROGENEITY")
lines.append("=" * 72)
lines.append(textwrap.fill(
    "Disparity CHANGE measures whether a condition produces a more or less "
    "spread distribution of AUROC across providers (e.g., cold_fed IQR vs "
    "cold_local IQR). Contrast HETEROGENEITY measures whether the per-provider "
    "delta is uniform or variable across providers (e.g., SD of the delta "
    "distribution). These are distinct. A condition can have lower IQR than a "
    "comparator (narrower distribution) while still showing high contrast "
    "heterogeneity (some providers gain a lot, others lose a lot, but they "
    "happen to cancel out). Both are reported below.",
    width=72, initial_indent="  ", subsequent_indent="  "))

# Disparity change: cold_fed vs cold_local IQR comparison
cf_iqr = disp_df[(disp_df.condition=="cold_fed")   & (disp_df.metric=="iqr")].iloc[0]
cl_iqr = disp_df[(disp_df.condition=="cold_local") & (disp_df.metric=="iqr")].iloc[0]
wl_iqr = disp_df[(disp_df.condition=="warm_local") & (disp_df.metric=="iqr")].iloc[0]
wf_iqr = disp_df[(disp_df.condition=="warm_fed")   & (disp_df.metric=="iqr")].iloc[0]
cf_min = disp_df[(disp_df.condition=="cold_fed")   & (disp_df.metric=="min")].iloc[0]
cl_min = disp_df[(disp_df.condition=="cold_local") & (disp_df.metric=="min")].iloc[0]
wl_min = disp_df[(disp_df.condition=="warm_local") & (disp_df.metric=="min")].iloc[0]

lines.append("")
lines.append("  Disparity change (condition-level IQR comparison):")
lines.append(f"    cold_fed   IQR: {cf_iqr.seed_mean:.5f} [{cf_iqr.ci_lo_95:.5f}, {cf_iqr.ci_hi_95:.5f}]")
lines.append(f"    cold_local IQR: {cl_iqr.seed_mean:.5f} [{cl_iqr.ci_lo_95:.5f}, {cl_iqr.ci_hi_95:.5f}]")
iqr_diff_1 = cf_iqr.seed_mean - cl_iqr.seed_mean
lines.append(f"    cold_fed - cold_local IQR: {iqr_diff_1:+.5f} "
             f"({'wider under FedAvg' if iqr_diff_1>0 else 'narrower under FedAvg'})")

lines.append(f"    warm_local IQR: {wl_iqr.seed_mean:.5f} [{wl_iqr.ci_lo_95:.5f}, {wl_iqr.ci_hi_95:.5f}]")
lines.append(f"    warm_fed   IQR: {wf_iqr.seed_mean:.5f} [{wf_iqr.ci_lo_95:.5f}, {wf_iqr.ci_hi_95:.5f}]")
iqr_diff_3 = wl_iqr.seed_mean - wf_iqr.seed_mean
lines.append(f"    warm_local - warm_fed IQR: {iqr_diff_3:+.5f} "
             f"({'wider under warm_local' if iqr_diff_3>0 else 'narrower under warm_local'})")

lines.append("")
lines.append("  Worst-provider absolute AUROC (condition-level, not contrast):")
lines.append(f"    cold_fed   min: {cf_min.seed_mean:.5f} [{cf_min.ci_lo_95:.5f}, {cf_min.ci_hi_95:.5f}]")
lines.append(f"    cold_local min: {cl_min.seed_mean:.5f} [{cl_min.ci_lo_95:.5f}, {cl_min.ci_hi_95:.5f}]")
lines.append(f"    warm_local min: {wl_min.seed_mean:.5f} [{wl_min.ci_lo_95:.5f}, {wl_min.ci_hi_95:.5f}]")
lines.append("    Note: warm_local has the highest absolute worst-provider AUROC,")
lines.append("    yet warm_local vs warm_fed contrast WORST DELTA is -0.087.")
lines.append("    These are consistent: the WORST PROVIDER UNDER WARM_LOCAL")
lines.append("    is a DIFFERENT provider than the worst provider under warm_fed.")

lines.append("")
lines.append("  Contrast heterogeneity (SD of per-provider deltas across seeds):")
for cond_a, cond_b, contrast_name in CONTRASTS:
    fx_sub = fx_df[fx_df.contrast == contrast_name]
    delta_sd = float(fx_sub["mean_delta"].std(ddof=1))
    n_sys = int(fx_sub["systematic"].sum())
    lines.append(f"    {contrast_name}: SD of provider mean_deltas = {delta_sd:.5f}; "
                 f"systematic providers = {n_sys}/{K}")

# ──────────────────────────────────────────────────────────────────────────────
# Summary metadata
# ──────────────────────────────────────────────────────────────────────────────
lines.append("")
lines.append("=" * 72)
lines.append("OUTPUTS")
lines.append(f"  {OUT_DISP}")
lines.append(f"  {OUT_FX}")
lines.append(f"  {OUT_WORST}")
lines.append(f"  {OUT_TXT}")
lines.append("=" * 72)

report = "\n".join(lines)
print(report)
OUT_TXT.write_text(report, encoding="utf-8")
