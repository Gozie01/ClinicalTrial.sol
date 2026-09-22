# Table Manifest — TRUST-CT CMPB Submission

All tables use locked values only. No vertical rules or shaded cells.
Every table cited in numerical order before it appears.

---

## Table 1 — Clinical Cohorts, Outcomes and Experimental Roles

**Location in manuscript:** Section 2.2
**Caption:** Clinical cohorts, outcomes and experimental roles.
**Source:** Locked cohort metadata (cimas_cohort_flow.json, phase4_lock.json, phase6_c0_lock.json)

**Columns:** Cohort | N | K | Primary outcome | Role | Evaluation

**Rows:**
| Cimas     | 12,649 | 23 | Refill adherence (≥5 months; 42.4%) | FL prediction; closed loop | LOCO / landmark |
| BETTER-BP | 402    | 2  | 12-month non-attendance (22.1%)     | Policy est.; replay calib. | Trial secondary  |
| eICU      | 2,520  | 8  | In-hospital mortality (5.0%)        | Security / privacy bench.  | Train/test split |

**Source files:**
- Cimas: `processed/cimas/cimas_cohort_flow.json`
- BETTER-BP: `processed/better_bp/results_phase4/cohort_flow.json`
- eICU: Phase 6 lock (run output: N=2520, K=8, pos_rate=0.050)

---

## Table 2 — Federated Prediction and Policy Evaluation

**Location in manuscript:** Section 3.2–3.3
**Caption:** Federated prediction (Cimas LOCO, N=12,649, K=23) and policy evaluation (BETTER-BP, N=402, B=2/3).

**Columns:** Analysis | Comparator | Metric | Estimate [95% CI] | Criterion | Interpretation

**Rows (confirmed from locked artifacts):**

Prediction rows:
| FedAvg vs Central   | Centralized  | ΔAUROC | −0.002 [−0.005, +0.001] | NI LCB > −0.02 | Non-inferior |
| FedAdam vs Central  | Centralized  | ΔAUROC | −0.005 [−0.011, 0.000]  | NI LCB > −0.02 | Non-inferior |
| Central (long-tail) | Prev. 0.677  | PR-AUC | 0.900 [0.882, 0.917]    | PR-lift > 1    | Lift = 1.329 |
| FedAvg (long-tail)  | Prev. 0.677  | PR-AUC | 0.899 [0.880, 0.916]    | PR-lift > 1    | Lift = 1.327 |

Policy rows:
| π_up vs π_rand | Random budget | ΔV̂ | +0.032 [−0.011, +0.078] | CI excludes 0 | Not established |

**Source files:**
- Prediction: `processed/cimas/results_phase3_freeze/ni_client_level.csv`,
  `gate_table_corrected.csv`
- Policy: `processed/better_bp/results_phase4/policy_contrasts.csv`,
  `policy_values.csv`

---

## Table 3 — Adversarial Robustness and Privacy–Utility Evaluation

**Location in manuscript:** Section 3.4–3.5
**Caption:** Adversarial robustness and privacy–utility evaluation (eICU, K=8, f=3, 5 seeds, 15 rounds).

**Columns:** Condition | Aggregator / α | Utility change (ΔAUROC) | Leakage metric | Interpretation

**Rows (confirmed from phase6_c0_lock.json):**
| Clean baseline       | FedAvg        | ---                            | AUROC 0.890 [0.889, 0.891]            | Reference              |
| Label flip (f=3)     | FedAvg        | −0.120 [−0.178, −0.061]*      | BSR n/a; PR-AUC ↓                    | Poisoning effective    |
| Label flip (f=3)     | Coord. median | +0.111 [+0.058, +0.165]*      | Recovery vs FedAvg                    | Defense effective      |
| Sign flip (f=3)      | FedAvg        | −0.059 [−0.103, −0.016]†     | ---                                   | Poisoning effective    |
| Backdoor (f=3)       | FedAvg        | +0.001 [−0.001, +0.003]      | BSR 0.922                             | AUROC insensitive      |
| ALIE (f=3)           | FedAvg        | −0.004 [−0.013, +0.004]      | ---                                   | Not significant        |
| α=0.01               | Clip + noise  | −0.005 [−0.020, +0.010]‡    | Recon cos. 0.534→0.464 (−13.0%)      | Within materiality     |
| α=0.05               | Clip + noise  | −0.074                        | Sign rec. 80.3%→67.9% (−12.4 pp)    | Exceeds materiality    |
| eICU MI attack       | α=0           | ---                           | AUROC 0.497 [0.496, 0.498]; h=−0.006 | At/below chance        |
| Cimas MI attack      | α=0           | ---                           | AUROC 0.507 [0.506, 0.507]; h=+0.014 | Negligible             |

Footnotes: * p=0.005; † p=0.020; ‡ p=0.39; 95% CI from bootstrap (2,000 resamples).

**Source files:**
- `processed/phase6/phase6b_adversarial.csv`
- `processed/phase6/phase6d_frontier.csv`
- `processed/phase6/phase6c_mia.csv`
- `processed/phase6_corrections/c4_leakage_corrected.json`

---

## Table 4 — Component Ablation Summary

**Location in manuscript:** Section 3.7
**Caption:** Component ablation summary (A1–A5).

**Columns:** ID | Component | Locked estimate [95% CI] | Prespecified criterion | Conclusion

**Rows (confirmed from ablation_summary.json):**
| A1 | FL noninferiority (FedAvg)    | NI LCB = −0.0045                       | > −0.02 AUROC          | Supported         |
| A2 | Coord. median under label flip | Recovery +0.111 [0.058, 0.165]         | p < 0.05; Δ > 0        | Supported         |
| A3 | α=0.01 utility cost           | ΔAUROC = −0.005 [−0.020, +0.010]; p=0.39 | |Δ| < 0.02           | Within range      |
| A4 | Audit-log fault detection      | 350/350; 0 false alerts                | All 350 detected        | Supported         |
| A5 | DR-policy uplift               | +0.032 [−0.011, +0.078]               | CI excludes 0           | Not established   |

**Source files:**
- `processed/ablation/ablation_summary.json`
- `processed/ablation/ablation_table.csv`

---

## Table 5 — E6 Provider-Level Performance Disparity (Warm/Cold De-confound)

**Location in manuscript:** Section 3.6 (E6 Results subsection)
**Caption:** Provider-level performance disparity under warm/cold federated learning conditions (Cimas, K=23, N=12,649; 5 seeds; rounds 51–60).

**Columns:** Contrast | Metric | Estimate [95% CI] | p | Interpretation

**Condition-level disparity (absolute IQR of provider AUROCs):**
| cold-FedAvg | IQR [95% CI] | 0.092 [0.073, 0.109] | — | Reference |
| cold-local  | IQR [95% CI] | 0.094 [0.073, 0.116] | — | Wider than FedAvg |
| warm-local  | IQR [95% CI] | 0.091 [0.075, 0.105] | — | Narrowest |

**Paired contrasts (macro-average AUROC):**
| cold-FedAvg vs cold-local | Macro ΔAUROC      | +0.003 [−0.003, +0.008] | 0.39  | Not significant |
| cold-FedAvg vs cold-local | Worst-provider Δ  | −0.114 [−0.141, −0.089] | 0.001 | FedAvg harms tail |
| warm-local vs cold-local  | Macro ΔAUROC      | +0.005 [+0.002, +0.009] | 0.07  | Trend only |
| warm-local vs cold-local  | Worst-provider Δ  | −0.031 [−0.044, −0.022] | 0.01  | Warm init harms tail |
| warm-local vs warm-FedAvg | Macro ΔAUROC      | +0.002 [+0.000, +0.006] | 0.21  | Cost of independence |

**Source files:**
- `processed/e6_fairness/e6_paired_analysis.csv`
- `processed/e6_fairness/e6_disparity_stats.csv`
- `processed/e6_fairness/e6_provider_fixed_effects.csv`

**Notes:**
- warm-FedAvg is a reproducibility gate (mathematically identical to cold-FedAvg rounds 31–60; 3,450/3,450 pairs, max |δ|=0); not reported as independent arm.
- Seed-level t-test (n=5, df=4); 95% CI from 5,000 seed-cluster bootstrap resamples.
- IQR difference between conditions not tested for significance; reported as descriptive disparity.
- 0/23 providers met the systematic-effect threshold (|mean_delta|/SD_delta > 2).
- Communication cost: warm-local saves 98.3% of parameter transfers vs continued FedAvg.
