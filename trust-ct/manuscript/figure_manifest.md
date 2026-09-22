# Figure Manifest — TRUST-CT CMPB Submission

All figures must be generated from final locked CSV/Parquet files.
Scripts must use colorblind-accessible palettes and be readable in grayscale.

---

## Figure 1 — TRUST-CT System Workflow

**Caption:** TRUST-CT closed-loop federated learning framework.
Three cohorts provide complementary experimental roles.
Provider-local data remain at each represented provider.
Raw clinical records are not transmitted to the coordinator.
Secure aggregation and distributed consensus were not implemented.

**Required elements:**
- Three cohort blocks: Cimas (K=23 providers), BETTER-BP (2 sites), eICU (8 hospitals, security benchmark)
- Provider-local data → local model training
- Coordinator aggregation (FedAvg / robust rules)
- Engagement-policy evaluation module (DR uplift)
- Trial-calibrated availability replay
- Privacy/security evaluation block (clipped Gaussian perturbation; reconstruction; MIA)
- Signed hash-linked audit log
- Arrows showing closed-loop flow

**Explicitly excluded:**
- Blockchain, PureChain, smart contracts, IPFS
- Secure aggregation
- Token rewards
- Gas cost

**Source:** Conceptual diagram (author-drawn); verify against
`run_phase6.py`, `run_phase5_v3.py`, `run_phase4_betterbp.py`

**Output:** `manuscript/figures/fig1_workflow.pdf`

---

## Figure 2 — Federated Prediction and Noninferiority

**Caption:** Forest plot of FL versus centralized AUROC for five
aggregation methods on the Cimas LOCO evaluation (N=12,649; K=23).
The vertical dashed line marks the prespecified noninferiority
margin of −0.02. All confidence intervals lie entirely above −0.02.

**Required elements:**
- One row per FL method (FedAvg, FedProx, FedAvg-equal, SCAFFOLD, FedAdam)
- Point estimate and 95% CI for ΔAUROC (FL − centralized)
- Vertical dashed line at −0.02
- Centralized reference point at 0

**Data source:** `processed/cimas/results_phase3_freeze/ni_client_level.csv`
Columns: method, obs (delta), lcb_95, lo_95, hi_95

**Values to plot:**
| Method       | ΔAUROC | NI LCB  |
|--------------|--------|---------|
| FedAvg       | −0.002 | −0.0045 |
| FedProx      | −0.002 | −0.0047 |
| FedAvg-equal | −0.001 | −0.0040 |
| SCAFFOLD     | −0.001 | −0.0040 |
| FedAdam      | −0.005 | −0.0095 |

**Script:** `manuscript/figure_scripts/fig2_ni_forest.py`
**Output:** `manuscript/figures/fig2_ni_forest.pdf`

---

## Figure 3 — Policy Value and Closed-Loop Replay

**Caption (two-panel):**
(A) DR policy-value estimates with 95% bootstrap confidence intervals
at budget B=2/3 (BETTER-BP, N=402).
(B) Cimas cold-start learning curves (nAULC) for conditions C0–C3
over 30 rounds with the ±0.001 operational materiality band shown.

**Panel A required elements:**
- Four policy bars/points: π₀ (no incentive), π_rand, π_risk, π_up
- 95% bootstrap CI for each
- Horizontal reference line at V(π_rand) = 0.860
- Note that CI for π_up − π_rand crosses zero

**Panel A data:**
`processed/better_bp/results_phase4/policy_values.csv` (budget=0.6667)
`processed/better_bp/results_phase4/policy_contrasts.csv`

**Panel B required elements:**
- Four learning curves: C0, C1, C2, C3 (mean over 5 seeds × 10 CRN draws)
- Shaded ±SE bands
- Horizontal dotted lines at ±0.001 (materiality band) centered on C1
- x-axis: round (1–30); y-axis: holdout AUROC

**Panel B data:**
`processed/phase5/results_v3/cimas_cold_rounds.parquet`
Columns: scenario, round, auroc_holdout (aggregate over model_seed, crn_draw)

**Script:** `manuscript/figure_scripts/fig3_policy_replay.py`
**Output:** `manuscript/figures/fig3_policy_replay.pdf`

---

## Figure 4 — Attack Degradation and Robust Aggregation Recovery

**Caption:** eICU adversarial FL benchmark (K=8, f=3, 5 seeds, 15 rounds).
(A) Mean AUROC degradation for five attack families under FedAvg
(37.5% malicious clients).
(B) Coordinate-median recovery relative to FedAvg under label flipping.
(C) Targeted backdoor: aggregate AUROC change (≈ 0) versus backdoor
success rate (BSR = 0.922).
Error bars: 95% bootstrap CI.

**Panel A data:** `processed/phase6/phase6b_adversarial.csv`
Filter: dataset=eicu, f=3, defense=fedavg, attack ∈ {label_flip, sign_flip,
model_replace, alie, backdoor}; exclude run_status=not_applicable_diagnostic_fallback

**Panel B data:** same file; compare defense in {fedavg, coord_median}
for label_flip, f=3

**Panel C data:** BSR column for backdoor rows; AUROC degradation for same

**Key values:**
- label_flip FedAvg: ΔAUROC = −0.120 [−0.178, −0.061]
- label_flip coord_median: recovery = +0.111 [+0.058, +0.165]
- backdoor BSR = 0.922; ΔAUROC = +0.001

**Script:** `manuscript/figure_scripts/fig4_attacks.py`
**Output:** `manuscript/figures/fig4_attacks.pdf`

---

## Figure 5 — Privacy–Utility Frontier

**Caption:** (A) Predictive utility (AUROC, mean ± 95% CI) and
optimistic reconstruction cosine similarity across α values.
The selected operating point α=0.01 is marked.
(B) Feature-sign recovery rate across α values.
(C) Membership-inference AUROC for eICU and Cimas cohorts across
α values; dashed line at 0.5 (chance).

**Panel A data:** `processed/phase6/phase6d_frontier.csv`
Columns: alpha, auroc_mean, auroc_lo95, auroc_hi95, recon_cos_sim_mean

Corrected reconstruction cosines (batch=1, from c4_leakage_corrected.json):
α=0.00: 0.534 | α=0.01: 0.464

Panel A note: use frontier CSV for AUROC; use C4 corrected values for
batch=1 cosine; plot both (batch-mix and batch=1) if space permits,
or use batch=1 as primary leakage indicator.

**Panel B data:** `processed/phase6_corrections/c4_leakage_corrected.json`
sign_recovery: α=0.00: 0.803, α=0.01: 0.679 (others from frontier label_recovery_rate)

**Panel C data:** `processed/phase6/phase6c_mia.csv`
aggregate by dataset; per-alpha MIA from c4_leakage_corrected.json per-alpha table

**Key annotations:**
- α=0.01 vertical dashed line (operating point)
- Materiality bands for AUROC (±0.02 from baseline)
- Chance line (0.5) on Panel C

**Script:** `manuscript/figure_scripts/fig5_frontier.py`
**Output:** `manuscript/figures/fig5_frontier.pdf`

---

## Figure 6 — E6 Provider-Level Performance Disparity: Warm/Cold De-confound

**Caption:** Provider-level performance disparity under warm/cold federated conditions
(Cimas, K=23, 5 seeds, 60 rounds).
(A) Mean AUROC learning curves for four conditions; shading = ±1 seed-SD.
(B) Worst-off provider AUROC learning curves; captures tail behaviour.
(C) Warm-FedAvg vs warm-local gap (rounds 31–60); isolates continued federation
from one-shot warm initialisation; error bars = 1 SEM.
Vertical dashed line at round 30 marks the warm-start checkpoint.

**Required elements:**
- Four conditions: cold_fed (FedAvg cold start), cold_local (local-only cold),
  warm_fed (FedAvg warm, reproducibility gate), warm_local (local-only warm)
- Colourblind palette: blue / orange / green / red
- Three panels in 14×5 layout
- Annotation of round-60 AUROC values and warm gap

**Data source:** `trust-ct/processed/e6_fairness/e6_summary_round_results.csv`
Columns: seed, condition, round, mean_auroc, min_auroc, max_auroc, std_auroc

**Key values (round 60, mean across seeds):**
| Condition  | Mean AUROC | Worst-provider AUROC |
|------------|------------|----------------------|
| cold_fed   | 0.761      | 0.518                |
| cold_local | 0.758      | 0.509                |
| warm_local | 0.763      | 0.525                |
warm_local vs warm_fed gap at r=60: +0.002 (warm_local slightly above)

**Script:** `manuscript/figure_scripts/fig6_e6_fairness.py`
**Output:** `manuscript/figures/fig6_e6_fairness.pdf` + `.png`
**Status:** CREATED
