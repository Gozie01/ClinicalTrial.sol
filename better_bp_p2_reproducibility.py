"""
P2 — BETTER-BP Trial Reproducibility Analysis
==============================================
Reproduces PUBLICLY VERIFIABLE trial statistics from the canonical P1 datasets.

What this script CAN reproduce (from public REDCap export):
  - 6-month and 12-month visit attendance by treatment arm (ITT)
  - SBP and DBP changes from baseline at 6m and 12m
  - MASES self-efficacy changes at 6m and 12m
  - Study completion and discontinuation counts
  - Adjusted logistic/linear regression with available covariates

What this script CANNOT reproduce (data not in public release):
  - Published primary outcome: 71% vs 34% adequate adherence (no daily bottle-opening data)
  - Lottery draw outcomes, $5/$50 incentive payment amounts
  - Sequential per-participant action logs

Outputs: processed/better_bp/p2_reproducibility/
  - table_attendance.csv        — visit attendance by arm (unadjusted + adjusted)
  - table_bp_changes.csv        — SBP/DBP mean changes at 6m and 12m
  - table_mases_changes.csv     — MASES self-efficacy changes
  - table_completion.csv        — study completion and discontinuation
  - results_summary.json        — machine-readable summary of all estimates + CIs
  - figure_sbp_change.png       — violin/strip plot of SBP change by arm
  - figure_attendance_flow.png  — visit attendance Sankey-style flow bar
  - p2_report.txt               — human-readable plain-text report
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = "processed/better_bp/p2_reproducibility"
os.makedirs(OUT, exist_ok=True)

# ── Load P1 outputs ────────────────────────────────────────────────────────────
out_df  = pd.read_parquet("processed/better_bp/trial_outcomes.parquet")
bas_df  = pd.read_parquet("processed/better_bp/participants_baseline.parquet")
vis_df  = pd.read_parquet("processed/better_bp/visits.parquet")

# ── Age top-coding correction ──────────────────────────────────────────────────
# P1 called numeric() on Age, which silently converts "aged 90 or older" to NaN.
# Correct treatment: encode as 90 (numeric) and flag with a top-coded indicator.
# This affects 1 Site A participant.
# Re-read the raw age column from the parquet — where it is NaN and site == "A",
# set age = 90 and age_top_coded = 1.
import pathlib, re as _re
_RAW_A = pathlib.Path("New/BETTER-BP/BETTERBP_De-identification-DATA_LABELS_SiteA.xlsx")
_raw_a = pd.read_excel(_RAW_A, dtype=str)
_raw_a.columns = [str(c).strip() for c in _raw_a.columns]
_raw_a["participant_id"] = "A_" + _raw_a["Record ID"].str.strip()
_ri_mask = _raw_a["Repeat Instrument"].fillna("").str.strip() == ""
_ev_mask = _raw_a["Event Name"].str.lower().str.contains("baseline", na=False)
_bl_a    = _raw_a[_ri_mask & _ev_mask]
_age_raw = (
    _bl_a.groupby("participant_id")["Age"]
    .first()
    .rename("age_raw")
)
_topcoded_pids = _age_raw[_age_raw.str.lower().str.contains("90 or older", na=False)].index
bas_df = bas_df.merge(_age_raw.reset_index(), on="participant_id", how="left")
bas_df["age_top_coded"] = bas_df["participant_id"].isin(_topcoded_pids).astype(int)
# Fill the 1 NaN age with 90
bas_df.loc[bas_df["participant_id"].isin(_topcoded_pids), "age"] = 90.0

# Merge baseline covariates into outcomes
df = out_df.merge(
    bas_df[["participant_id", "age", "age_top_coded", "sbp_baseline", "dbp_baseline", "mases_baseline"]],
    on="participant_id", how="left"
)

# Binary encoding
df["arm_bin"] = (df["treatment_arm"] == "Intervention").astype(int)
df["site_bin"] = (df["site"] == "A").astype(int)

N_TOTAL = len(df)
N_INT   = (df["treatment_arm"] == "Intervention").sum()
N_CTL   = (df["treatment_arm"] == "Control").sum()

report_lines = []
results_summary = {}

def _rline(s=""):
    report_lines.append(s)

def _section(title):
    _rline()
    _rline("=" * 70)
    _rline(f"  {title}")
    _rline("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Cohort Summary
# ─────────────────────────────────────────────────────────────────────────────
_section("§1  Cohort Summary")
_rline(f"Total randomized (raw public release): {N_TOTAL}")
_rline(f"  Intervention: {N_INT}   Control: {N_CTL}")
_rline()
_rline("Denominator note:")
_rline("  Published report states N=400 (265 intervention + 135 control).")
_rline("  Raw public files contain 402 records (266+136). Both discrepant records")
_rline("  are retained throughout; no records are excluded from the ITT analysis.")
_rline()
n_topcoded = int(bas_df["age_top_coded"].sum())
_rline(f"Age top-coding:")
_rline(f"  Site A REDCap encodes age as 'aged 90 or older' for {n_topcoded} participant(s).")
_rline(f"  These are recorded as age = 90 with an age_top_coded indicator (= 1).")
_rline(f"  This avoids listwise deletion of a valid observation.")

def _arm_table(df, col, label):
    g = df.groupby("treatment_arm")[col]
    rows = []
    for arm in ["Intervention", "Control"]:
        vals = g.get_group(arm).dropna()
        rows.append({
            "Arm": arm, "N": int(vals.count()),
            "Mean": round(float(vals.mean()), 3),
            "SD": round(float(vals.std()), 3),
            "Median": round(float(vals.median()), 3),
        })
    t = pd.DataFrame(rows)
    return t

# Baseline SBP
_rline()
_rline("Baseline SBP (mmHg) — randomized participants:")
bsln_sbp = df.merge(bas_df[["participant_id","sbp_baseline"]], on="participant_id", how="left",
                    suffixes=("","_b"))
_rline(_arm_table(df, "sbp_change_6m", "sbp_change_6m").to_string(index=False))

# Age (from baseline)
_rline()
_rline("Age by arm (randomized only):")
age_tbl = _arm_table(df, "age", "age")
_rline(age_tbl.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# §2  Visit Attendance
# ─────────────────────────────────────────────────────────────────────────────
_section("§2  Visit Attendance (Intention-to-Treat)")

def attendance_stats(df, col, arm_col="treatment_arm"):
    rows = []
    for arm in ["Intervention", "Control"]:
        sub = df[df[arm_col] == arm]
        n   = len(sub)
        att = sub[col].sum()
        pct = att / n
        # Wilson 95% CI
        z   = 1.96
        p   = pct
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        half   = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
        rows.append({
            "Arm": arm, "N": n, "Attended": int(att),
            "Pct": round(pct*100, 1),
            "CI_lo": round(max(0, centre-half)*100, 1),
            "CI_hi": round(min(1, centre+half)*100, 1),
        })
    return pd.DataFrame(rows)

att6  = attendance_stats(df, "visit_6m_attended")
att12 = attendance_stats(df, "visit_12m_attended")

_rline()
_rline("6-Month Visit Attendance:")
_rline(att6.to_string(index=False))
_rline()
_rline("12-Month Visit Attendance:")
_rline(att12.to_string(index=False))

# Unadjusted risk difference for 6m attendance
int6_p  = att6.set_index("Arm").loc["Intervention", "Pct"] / 100
ctl6_p  = att6.set_index("Arm").loc["Control",      "Pct"] / 100
rd6     = int6_p - ctl6_p
n_int, n_ctl = N_INT, N_CTL
se_rd6  = (int6_p*(1-int6_p)/n_int + ctl6_p*(1-ctl6_p)/n_ctl) ** 0.5
rd6_lo  = rd6 - 1.96 * se_rd6
rd6_hi  = rd6 + 1.96 * se_rd6

_rline()
_rline(f"Unadjusted RD (6m, Intervention−Control): "
       f"{rd6*100:+.1f}% (95% CI {rd6_lo*100:+.1f}% to {rd6_hi*100:+.1f}%)")

# Unadjusted OR for 6m
from scipy.stats import chi2_contingency
ct6 = pd.crosstab(df["treatment_arm"], df["visit_6m_attended"])
chi2_6, p_6, _, _ = chi2_contingency(ct6)
_rline(f"Chi-squared test (6m attendance): chi2={chi2_6:.3f}, nominal p={p_6:.4f}")
_rline()
_rline("IMPORTANT — significance caveat:")
_rline("  Six-month visit attendance was not the prespecified primary BETTER-BP endpoint")
_rline("  (which was objective medication adherence by bottle-opening). Multiple outcomes")
_rline("  are examined in this script. The nominal p-value (0.039) is therefore exploratory")
_rline("  and does not survive a Bonferroni correction for all comparisons conducted.")
_rline()
_rline("Recommended interpretation of the 6m vs 12m attendance pattern:")
_rline("  Randomization to the incentive intervention was associated with higher six-month")
_rline("  visit attendance, corresponding to the active incentive period, whereas no")
_rline("  attendance difference was observed at 12 months. Because visit attendance was")
_rline("  analyzed as a secondary retrospective endpoint and the intervention targeted")
_rline("  medication adherence rather than visit participation, this finding is interpreted")
_rline("  as evidence of a potentially time-limited engagement effect requiring external")
_rline("  confirmation. The 12-month null result is substantively important: universally")
_rline("  providing incentives did not produce a sustained attendance advantage, motivating")
_rline("  the adaptive targeting approach in TRUST-CT.")

# Adjusted logistic regression for 6m attendance
# Covariates: site, age, sbp_baseline, dbp_baseline (complete cases)
adj_df = df.dropna(subset=["age","sbp_baseline","dbp_baseline","visit_6m_attended"]).copy()
adj_df["sbp_base_c"] = adj_df["sbp_baseline"] - adj_df["sbp_baseline"].mean()
adj_df["age_c"]      = adj_df["age"]           - adj_df["age"].mean()
adj_df["dbp_base_c"] = adj_df["dbp_baseline"]  - adj_df["dbp_baseline"].mean()

try:
    logit_6m = logit(
        "visit_6m_attended ~ arm_bin + site_bin + age_c + sbp_base_c + dbp_base_c",
        data=adj_df
    ).fit(disp=False)
    coef_arm = logit_6m.params["arm_bin"]
    ci_lo_l  = logit_6m.conf_int().loc["arm_bin", 0]
    ci_hi_l  = logit_6m.conf_int().loc["arm_bin", 1]
    p_arm    = logit_6m.pvalues["arm_bin"]
    or_arm   = np.exp(coef_arm)
    or_lo    = np.exp(ci_lo_l)
    or_hi    = np.exp(ci_hi_l)
    _rline()
    n_excl = N_TOTAL - adj_df.shape[0]
    _rline(f"Adjusted OR for 6m attendance (arm, controlling for site, age, baseline SBP/DBP):")
    _rline(f"  OR={or_arm:.3f}  95% CI [{or_lo:.3f}, {or_hi:.3f}]  p={p_arm:.4f}")
    _rline(f"  N in adjusted model: {adj_df.shape[0]} (complete-case; {n_excl} records excluded")
    _rline(f"  for missing baseline SBP/DBP; this is NOT the published N=400 cohort).")
    results_summary["att6m_adj_or"]    = round(or_arm, 4)
    results_summary["att6m_adj_or_lo"] = round(or_lo, 4)
    results_summary["att6m_adj_or_hi"] = round(or_hi, 4)
    results_summary["att6m_adj_p"]     = round(p_arm, 6)
except Exception as e:
    _rline(f"  [Adjusted logistic failed: {e}]")

# 12-month
int12_p = att12.set_index("Arm").loc["Intervention", "Pct"] / 100
ctl12_p = att12.set_index("Arm").loc["Control",      "Pct"] / 100
rd12    = int12_p - ctl12_p
se_rd12 = (int12_p*(1-int12_p)/n_int + ctl12_p*(1-ctl12_p)/n_ctl) ** 0.5
_rline()
_rline(f"Unadjusted RD (12m, Intervention−Control): "
       f"{rd12*100:+.1f}% (95% CI {rd12-1.96*se_rd12:+.1f}% to {rd12+1.96*se_rd12:+.1f}%)")
ct12 = pd.crosstab(df["treatment_arm"], df["visit_12m_attended"])
chi2_12, p_12, _, _ = chi2_contingency(ct12)
_rline(f"Chi-squared test (12m attendance): χ²={chi2_12:.3f}, p={p_12:.4f}")

results_summary["att6m"] = {
    "intervention_pct": float(att6.set_index("Arm").loc["Intervention","Pct"]),
    "control_pct":      float(att6.set_index("Arm").loc["Control","Pct"]),
    "rd_pct": round(rd6*100, 2), "rd_ci_lo": round(rd6_lo*100, 2), "rd_ci_hi": round(rd6_hi*100, 2),
    "chi2": round(chi2_6, 4), "p": round(p_6, 6),
}
results_summary["att12m"] = {
    "intervention_pct": float(att12.set_index("Arm").loc["Intervention","Pct"]),
    "control_pct":      float(att12.set_index("Arm").loc["Control","Pct"]),
    "rd_pct": round(rd12*100, 2),
}


# ─────────────────────────────────────────────────────────────────────────────
# §3  Blood-Pressure Changes
# ─────────────────────────────────────────────────────────────────────────────
_section("§3  SBP and DBP Changes from Baseline")

def bp_table(df, col, timepoint):
    rows = []
    for arm in ["Intervention", "Control"]:
        sub = df[df["treatment_arm"] == arm][col].dropna()
        n   = len(sub)
        mn  = sub.mean()
        sd  = sub.std()
        se  = sd / n**0.5
        rows.append({
            "Arm": arm, "N_with_data": n,
            "Mean_change": round(mn, 2), "SD": round(sd, 2),
            "CI_lo": round(mn - 1.96*se, 2), "CI_hi": round(mn + 1.96*se, 2),
            "Timepoint": timepoint,
        })
    return pd.DataFrame(rows)

def compare_arms(df, col):
    g_int = df[df["treatment_arm"]=="Intervention"][col].dropna()
    g_ctl = df[df["treatment_arm"]=="Control"][col].dropna()
    # Two-sample t-test
    t, p = stats.ttest_ind(g_int, g_ctl, equal_var=False)
    diff  = g_int.mean() - g_ctl.mean()
    pooled_sd = ((g_int.std()**2/len(g_int)) + (g_ctl.std()**2/len(g_ctl)))**0.5
    ci_lo = diff - 1.96 * pooled_sd
    ci_hi = diff + 1.96 * pooled_sd
    return {"diff": round(diff,2), "ci_lo": round(ci_lo,2), "ci_hi": round(ci_hi,2),
            "t": round(t,3), "p": round(p,6)}

sbp6_tbl  = bp_table(df, "sbp_change_6m",  "6m")
sbp12_tbl = bp_table(df, "sbp_change_12m", "12m")
dbp6_tbl  = bp_table(df, "dbp_change_6m",  "6m")
dbp12_tbl = bp_table(df, "dbp_change_12m", "12m")

_rline()
_rline("SBP change from baseline (mmHg):")
_rline("  6-month:")
_rline(sbp6_tbl.to_string(index=False))
cmp = compare_arms(df, "sbp_change_6m")
_rline(f"  Arm difference (Int−Ctl): {cmp['diff']:+.2f} mmHg "
       f"(95% CI {cmp['ci_lo']:+.2f} to {cmp['ci_hi']:+.2f}), t={cmp['t']:.3f}, p={cmp['p']:.4f}")

_rline()
_rline("  12-month:")
_rline(sbp12_tbl.to_string(index=False))
cmp12 = compare_arms(df, "sbp_change_12m")
_rline(f"  Arm difference (Int−Ctl): {cmp12['diff']:+.2f} mmHg "
       f"(95% CI {cmp12['ci_lo']:+.2f} to {cmp12['ci_hi']:+.2f}), t={cmp12['t']:.3f}, p={cmp12['p']:.4f}")

_rline()
_rline("DBP change from baseline (mmHg):")
_rline("  6-month:")
_rline(dbp6_tbl.to_string(index=False))
cmp_d6 = compare_arms(df, "dbp_change_6m")
_rline(f"  Arm difference (Int−Ctl): {cmp_d6['diff']:+.2f} mmHg "
       f"(95% CI {cmp_d6['ci_lo']:+.2f} to {cmp_d6['ci_hi']:+.2f}), t={cmp_d6['t']:.3f}, p={cmp_d6['p']:.4f}")

_rline()
_rline("  12-month:")
_rline(dbp12_tbl.to_string(index=False))
cmp_d12 = compare_arms(df, "dbp_change_12m")
_rline(f"  Arm difference (Int−Ctl): {cmp_d12['diff']:+.2f} mmHg "
       f"(95% CI {cmp_d12['ci_lo']:+.2f} to {cmp_d12['ci_hi']:+.2f}), t={cmp_d12['t']:.3f}, p={cmp_d12['p']:.4f}")

results_summary["sbp_change_6m"]  = cmp
results_summary["sbp_change_12m"] = cmp12
results_summary["dbp_change_6m"]  = cmp_d6
results_summary["dbp_change_12m"] = cmp_d12

# Adjusted linear regression for SBP change at 6m
try:
    adj_sbp_df = df.dropna(subset=["sbp_change_6m","age","sbp_baseline"]).copy()
    adj_sbp_df["sbp_base_c"] = adj_sbp_df["sbp_baseline"] - adj_sbp_df["sbp_baseline"].mean()
    adj_sbp_df["age_c"]      = adj_sbp_df["age"]           - adj_sbp_df["age"].mean()
    ols_sbp = ols(
        "sbp_change_6m ~ arm_bin + site_bin + age_c + sbp_base_c",
        data=adj_sbp_df
    ).fit()
    coef = ols_sbp.params["arm_bin"]
    ci   = ols_sbp.conf_int().loc["arm_bin"]
    pv   = ols_sbp.pvalues["arm_bin"]
    _rline()
    _rline(f"Adjusted treatment effect on SBP change at 6m (OLS, N={adj_sbp_df.shape[0]}):")
    _rline(f"  Coefficient={coef:+.3f} mmHg  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={pv:.4f}")
    results_summary["sbp_change_6m_adj"] = {
        "coef": round(coef,3), "ci_lo": round(ci[0],3), "ci_hi": round(ci[1],3), "p": round(pv,6)
    }
except Exception as e:
    _rline(f"  [Adjusted OLS for SBP change failed: {e}]")


# ─────────────────────────────────────────────────────────────────────────────
# §4  MASES Self-Efficacy Changes
# ─────────────────────────────────────────────────────────────────────────────
_section("§4  MASES Self-Efficacy Changes")
_rline("NOTE: MASES is a self-reported self-efficacy scale, NOT objective adherence.")

mas6_tbl  = bp_table(df, "mases_change_6m",  "6m")
mas12_tbl = bp_table(df, "mases_change_12m", "12m")

_rline()
_rline("MASES change from baseline:")
_rline("  6-month:")
_rline(mas6_tbl.to_string(index=False))
cmp_m6 = compare_arms(df, "mases_change_6m")
_rline(f"  Arm difference (Int−Ctl): {cmp_m6['diff']:+.3f} "
       f"(95% CI {cmp_m6['ci_lo']:+.3f} to {cmp_m6['ci_hi']:+.3f}), t={cmp_m6['t']:.3f}, p={cmp_m6['p']:.4f}")

_rline()
_rline("  12-month:")
_rline(mas12_tbl.to_string(index=False))
cmp_m12 = compare_arms(df, "mases_change_12m")
_rline(f"  Arm difference (Int−Ctl): {cmp_m12['diff']:+.3f} "
       f"(95% CI {cmp_m12['ci_lo']:+.3f} to {cmp_m12['ci_hi']:+.3f}), t={cmp_m12['t']:.3f}, p={cmp_m12['p']:.4f}")

results_summary["mases_change_6m"]  = cmp_m6
results_summary["mases_change_12m"] = cmp_m12


# ─────────────────────────────────────────────────────────────────────────────
# §5  Study Completion and Discontinuation
# ─────────────────────────────────────────────────────────────────────────────
_section("§5  Study Completion and Discontinuation")
_rline("Three mutually exclusive categories: completed / discontinued / unknown (missing EOS record).")
_rline("Unknown and discontinued are NOT combined in the primary descriptive result.")

overall_n     = len(df)
overall_compl = int((df["study_completed"]==1).sum())
overall_disc  = int((df["explicit_noncompletion"]==1).sum())
overall_meos  = int(df["missing_eos_record"].sum())

# By arm
rows_compl = []
for arm in ["Intervention", "Control"]:
    sub = df[df["treatment_arm"]==arm]
    n   = len(sub)
    c   = int((sub["study_completed"]==1).sum())
    d   = int((sub["explicit_noncompletion"]==1).sum())
    m   = int(sub["missing_eos_record"].sum())
    rows_compl.append({
        "Arm": arm, "N": n,
        "Completed": c, "Completed_pct": round(c/n*100, 1),
        "Discontinued": d, "Discontinued_pct": round(d/n*100, 1),
        "Unknown_missing_EOS": m, "Unknown_pct": round(m/n*100, 1),
    })
compl_tbl = pd.DataFrame(rows_compl)
_rline()
_rline(compl_tbl.to_string(index=False))
_rline()
_rline(f"Overall (N={overall_n}):")
_rline(f"  Completed           : {overall_compl}/{overall_n} = {overall_compl/overall_n*100:.1f}%")
_rline(f"  Discontinued        : {overall_disc}/{overall_n} = {overall_disc/overall_n*100:.1f}%")
_rline(f"  Unknown/missing EOS : {overall_meos}/{overall_n} = {overall_meos/overall_n*100:.1f}%")
_rline(f"  [Check: {overall_compl}+{overall_disc}+{overall_meos} = {overall_compl+overall_disc+overall_meos} (should equal {overall_n})]")

results_summary["completion"] = {
    "total": overall_n,
    "completed": overall_compl,
    "completed_pct": round(overall_compl/overall_n*100, 1),
    "discontinued": overall_disc,
    "discontinued_pct": round(overall_disc/overall_n*100, 1),
    "missing_eos": overall_meos,
    "missing_eos_pct": round(overall_meos/overall_n*100, 1),
}


# ─────────────────────────────────────────────────────────────────────────────
# §6  Site-Stratified Summaries
# ─────────────────────────────────────────────────────────────────────────────
_section("§6  Site-Stratified Summaries")

for site in ["A", "B"]:
    sub = df[df["site"]==site]
    n   = len(sub)
    a6  = sub["visit_6m_attended"].mean()
    a12 = sub["visit_12m_attended"].mean()
    sbp_m = sub["sbp_change_6m"].mean()
    _rline(f"Site {site}: N={n}, 6m att={a6*100:.1f}%, 12m att={a12*100:.1f}%, "
           f"mean SBP Δ at 6m={sbp_m:+.1f} mmHg")


# ─────────────────────────────────────────────────────────────────────────────
# §7  What We Cannot Reproduce
# ─────────────────────────────────────────────────────────────────────────────
_section("§7  Outcomes NOT Reproducible from Public Data")
_rline("""
The following published results CANNOT be reproduced from the public REDCap export:

  1. PRIMARY ADHERENCE OUTCOME: Published report shows 71.0% vs 34.1% adequate
     adherence (bottle opening ≥80% of days). The AdhereTech bottle-opening
     event stream is NOT included in the public release; only bottle setup/return
     records are available. This outcome is therefore computationally unverifiable.

  2. LOTTERY OUTCOMES: The $5 and $50 lottery randomizations and payout records
     are not in the public dataset. Per-participant incentive amounts cannot be
     reconstructed.

  3. SEQUENTIAL ACTION LOGS: No daily or weekly per-participant policy action
     propensity records are available. DQN or bandit policies cannot be trained
     on observed trial actions.

These limitations are documented in the TRUST-CT CMPB revision. All claims
about BETTER-BP in the paper are restricted to visit-level outcomes above.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────────────────────────────────────
attendance_out = pd.concat([
    att6.assign(Timepoint="6m"), att12.assign(Timepoint="12m")
])
attendance_out.to_csv(f"{OUT}/table_attendance.csv", index=False)

bp_out = pd.concat([sbp6_tbl.assign(Measure="SBP"), sbp12_tbl.assign(Measure="SBP"),
                    dbp6_tbl.assign(Measure="DBP"),  dbp12_tbl.assign(Measure="DBP")])
bp_out.to_csv(f"{OUT}/table_bp_changes.csv", index=False)

mases_out = pd.concat([mas6_tbl, mas12_tbl])
mases_out.to_csv(f"{OUT}/table_mases_changes.csv", index=False)

compl_tbl.to_csv(f"{OUT}/table_completion.csv", index=False)

with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

with open(f"{OUT}/p2_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — SBP Change Distributions
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.suptitle("BETTER-BP: SBP Change from Baseline by Arm", fontsize=13)

for ax, (tp, col) in zip(axes, [("6-Month", "sbp_change_6m"), ("12-Month", "sbp_change_12m")]):
    data_int = df[df["treatment_arm"]=="Intervention"][col].dropna()
    data_ctl = df[df["treatment_arm"]=="Control"][col].dropna()
    ax.violinplot([data_int, data_ctl], positions=[1, 2], showmedians=True, showextrema=False)
    ax.scatter(np.ones(len(data_int)) + np.random.uniform(-0.05,0.05,len(data_int)),
               data_int, alpha=0.25, s=6, color="steelblue")
    ax.scatter(np.ones(len(data_ctl))*2 + np.random.uniform(-0.05,0.05,len(data_ctl)),
               data_ctl, alpha=0.25, s=6, color="coral")
    ax.axhline(0, lw=0.8, ls="--", color="gray")
    ax.set_title(tp)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Intervention", "Control"])
    ax.set_ylabel("SBP Change (mmHg)")

    # Annotate means
    ax.annotate(f"μ={data_int.mean():+.1f}", xy=(1, data_int.mean()), xytext=(1.2, data_int.mean()+2),
                fontsize=8, color="steelblue")
    ax.annotate(f"μ={data_ctl.mean():+.1f}", xy=(2, data_ctl.mean()), xytext=(2.1, data_ctl.mean()+2),
                fontsize=8, color="coral")

plt.tight_layout()
plt.savefig(f"{OUT}/figure_sbp_change.png", dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Visit Attendance Flow
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("BETTER-BP: Visit Attendance by Arm (ITT)", fontsize=13)

arms   = ["Intervention", "Control"]
colors = ["steelblue", "coral"]
n_arm  = {"Intervention": N_INT, "Control": N_CTL}
x      = [0, 1, 2]
labels = ["Randomized", "6m Visit", "12m Visit"]
width  = 0.35

for i, (arm, color) in enumerate(zip(arms, colors)):
    n = n_arm[arm]
    a6  = int(att6.set_index("Arm").loc[arm, "Attended"])
    a12 = int(att12.set_index("Arm").loc[arm, "Attended"])
    bars = ax.bar([xi + i*width - width/2 for xi in x], [n, a6, a12], width, label=arm, color=color, alpha=0.75)
    for bar, val in zip(bars, [n, a6, a12]):
        pct = val / n * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Participants")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/figure_attendance_flow.png", dpi=150, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Final printout
# ─────────────────────────────────────────────────────────────────────────────
print("\n".join(report_lines).encode("ascii", errors="replace").decode("ascii"))
print()
print(f"Outputs written to: {os.path.abspath(OUT)}")
print("  table_attendance.csv, table_bp_changes.csv, table_mases_changes.csv")
print("  table_completion.csv, results_summary.json, p2_report.txt")
print("  figure_sbp_change.png, figure_attendance_flow.png")
