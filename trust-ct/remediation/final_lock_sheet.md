# TRUST-CT Final Lock Sheet
**Date:** 2026-08-28  
**Scope:** W01–W15 machine remediation (approved options: W04-A, W07-B, W08-A, W09-A, W10-A)  
**Status:** LOCKED where marked; AUTHOR_PENDING where marked

---

## LOCKED RESULTS (do not alter without documented justification)

### Phase 3 — Cimas FL Noninferiority
| ID | Method | Δ AUROC (obs) | 95% CI | NI LCB | Verdict | Source | CI method |
|----|--------|--------------|--------|--------|---------|--------|-----------|
| R01 | FedAvg | −0.00214 | [−0.00503, +0.00077] | −0.0045 | NON-INFERIOR | ni_client_level.csv | bootstrap_n2000 |
| R02 | FedProx | −0.00223 | [−0.00514, +0.00070] | −0.0047 | NON-INFERIOR | ni_client_level.csv | bootstrap_n2000 |
| R03 | FedAvg-equal | −0.00147 | [−0.00447, +0.00142] | −0.0040 | NON-INFERIOR | ni_client_level.csv | bootstrap_n2000 |
| R04 | SCAFFOLD | −0.00144 | [−0.00440, +0.00144] | −0.0040 | NON-INFERIOR | ni_client_level.csv | bootstrap_n2000 |
| R05 | FedAdam | −0.00514 | [−0.01050, +0.00004] | −0.0095 | NON-INFERIOR | ni_client_level.csv | bootstrap_n2000 |

**Note (W03b):** Δ=−0.002 is the bootstrap NI estimate (obs=−0.002135), not the rounded marginal (0.795−0.799=−0.004). These must not be conflated.  
**Provenance:** K=23 Cimas providers; N_BOOT=2000 (from provenance.json); seeds=[7,11,19,23,37].

### Phase 3 — Cimas Long-tail AUROC (Within-Source Transportability)
| ID | Model | AUROC | 95% CI | n_patients | n_providers | Source |
|----|-------|-------|--------|-----------|-------------|--------|
| R06 | Central | 0.807 | [0.785, 0.829] | 3716 | 264 | longtail_corrected.csv |
| R07 | FedAvg | 0.803 | [0.781, 0.826] | 3716 | 264 | longtail_corrected.csv |

**Note (W02c):** Long-tail reframed as within-source transportability evaluation. Risk-positive AUROC orientation (Y_risk = 1 − Y_adherence). CI from source (reported_95ci); N_BOOT=2000.

### Phase 4 — Policy Values at B = 2/3
| ID | Policy | V̂(π) | Source |
|----|--------|------|--------|
| R08 | no_incentive | 0.815089 | policy_values.csv |
| R09 | random_budget | 0.860304 | policy_values.csv |
| R10 | risk_targeting | 0.885030 | policy_values.csv |
| R11 | uplift_targeting | 0.891863 | policy_values.csv |
| R12 | Δ uplift vs random | +0.03156 [−0.01126, +0.07773] | policy_contrasts.csv; N_BOOT=2000, seeds=5, folds=5 |

**Note:** CI crosses zero — policy improvement NOT established. Δ_attend_per100=+3.16 [−1.13, +7.77].  
**Provenance (corrected):** N_BOOT=2000 (from provenance.json, not 500). CI read from policy_contrasts.csv primary row at B=0.6667.

### Phase 6B — Adversarial Benchmark (eICU, FedAvg, f=3)
All Δ AUROC contrasts use **matched-seed paired t-test** (df=4, n=5 seed pairs). CI method: paired_ttest_df4.

| ID | Attack | Δ AUROC | 95% CI | p | CI method | Source / Run |
|----|--------|---------|--------|---|-----------|-------------|
| R13 | Clean baseline | — | [0.88920, 0.89132] | — | bootstrap_n2000 | phase6b_adversarial.csv |
| R14 | Label flip | −0.11950 | [−0.17760, −0.06141] | 0.0046 | paired_ttest_df4 | phase6b_adversarial.csv |
| R15 | Coord. median recovery | +0.11143 | [+0.05777, +0.16510] | 0.0045 | paired_ttest_df4 | phase6b_adversarial.csv |
| R16 | Sign flip | −0.05934 | [−0.10316, −0.01552] | 0.0198 | paired_ttest_df4 | phase6b_adversarial.csv |
| R17 | ALIE | −0.00417 | [−0.01261, +0.00426] | 0.2416 | paired_ttest_df4 | phase6b_adversarial.csv |
| R18 | Random Gauss **(W09a)** | −0.11153 | [−0.19663, −0.02644] | 0.0220 | paired_ttest_df4 | **phase6b_adversarial_w09.csv** |
| R19 | Backdoor BSR **(W09c)** | BSR = 0.92024 [0.91905, 0.92143] | n/a | — | bootstrap_n2000 | **phase6b_adversarial_w09.csv** |

**Note (W09a):** Original random_gauss was unseeded (non-reproducible). Corrected values show stronger degradation (−0.112 vs original −0.081).  
**Note (W10a):** BSR renamed "triggered target-class rate"; no clean-model baseline collected; reported descriptively.  
**CI method correction:** Previous ledger used bootstrap CI for R14-R17; corrected to matched-seed paired t-test. p-values are computed from data, not hardcoded.

### Phase 6C — Gradient Reconstruction (C0-correct, batch=1)
| ID | Alpha | Cosine (mean ± std) | Source |
|----|-------|---------------------|--------|
| R20 | 0.00 | 1.000 ± 0.000 | phase6c_reconstruction.csv |
| R21 | 0.01 | 0.996 ± 0.003 | phase6c_reconstruction.csv |
| R22 | 0.05 | 0.879 ± 0.070 | phase6c_reconstruction.csv |
| R23 | 0.10 | 0.722 ± 0.098 | phase6c_reconstruction.csv |
| R24 | 0.20 | 0.452 ± 0.141 | phase6c_reconstruction.csv |

**Note (W07):** Prior values (0.534/0.464) from audit4_leakage_table.csv were batch-pooled pre-C0 MI — SUPERSEDED.  
**AUTHOR_PENDING:** Feature-sign recovery at batch=1 not stored in reconstruction CSV.

### Phase 6C — MIA (W08-A: existing 5-seed bootstrap)
| ID | Dataset | AUROC (mean) | 95% CI | Cohen h | Source |
|----|---------|-------------|--------|---------|--------|
| R25 | eICU | 0.497 | [0.496, 0.498] | −0.006 | phase6c_mia.csv |
| R26 | Cimas | 0.507 | [0.506, 0.507] | +0.014 | phase6c_mia.csv |

### Phase 6C — Utility Under Clipped Gaussian (W09b corrected, per-client noise)
| ID | Alpha | AUROC | 95% CI | Δ vs clean | Materiality | Source |
|----|-------|-------|--------|-----------|-------------|--------|
| R27 | 0.01 | 0.8896 | [0.8880, 0.8912] | −0.001 | **Within** | phase6c_leakage_fl_w09.csv |
| R28 | 0.05 | 0.8798 | [0.8713, 0.8899] | −0.011 | **Within** | phase6c_leakage_fl_w09.csv |
| R29 | 0.10 | 0.8401 | [0.8204, 0.8697] | −0.050 | Exceeds | phase6c_leakage_fl_w09.csv |
| R30 | 0.20 | 0.7572 | [0.7113, 0.8229] | −0.133 | Exceeds | phase6c_leakage_fl_w09.csv |

**Note (W09b):** Original shared-noise bug inflated degradation at α≥0.01. Corrected α=0.05 is now WITHIN materiality (original was −0.074, "Exceeds"). Both α=0.01 and α=0.05 are acceptable; α=0.01 selected (conservative). Operating point selection is AUTHOR DECISION.

### Phase 5 — Audit-log Fault Detection
| ID | Metric | Value | 95% CI (Clopper-Pearson) | Source |
|----|--------|-------|--------------------------|--------|
| R31 | Fault detection rate | 350/350 (1.00000) | [0.98952, 1.00000] | phase5/results_v3/fault_injection_results.csv |

**Note:** CI is exact Clopper-Pearson from k=350, n=350. No fallback hardcoding — file read directly (issue 5 fix).

---

## AUTHOR_PENDING ITEMS (cannot be resolved computationally)

1. **Ethics/IRB approval IDs** for Cimas and BETTER-BP cohorts (required for submission)
2. **NCT04114669 confirmation** — verify this NCT number applies to BETTER-BP
3. **402 vs 400 discrepancy** — cohort_flow.json has 402; some manuscript tables say 400
4. **Cimas prevalence** — primary-cohort 45.419% vs landmark 42.432%; explain or correct
5. **pi_risk CI bounds** — Phase 4 bootstrap CI for risk_targeting not in processed/ outputs
6. **Feature-sign recovery** — batch=1 sign recovery requires re-running with x_hat storage
7. **Communication-round definition** — inner FL rounds vs outer replay rounds (W05b)
8. **CRediT contributions, affiliations, AI-use disclosure** — required for submission
9. **Operating-point decision at α=0.05** — W09b shows α=0.05 is now also within materiality; author to confirm whether α=0.01 remains the selected operating point or update to α=0.05
10. **Phase 5 preprocessing mismatch (W04b)** — `fl_fit` in run_phase5_v3.py trains on raw `X_bp` but evaluates weights on preprocessed `X_bp_pp`. Affects all Phase 5 participation rate estimates (R08-R12). Under W04-A reframing this is documented but the direction of bias is undetermined without rerunning. Author should verify the reported policy values are interpretable under the reframed description.
11. **fig4_attacks.py gap** — Figure reads from pre-W09 paper_bundle/security_attacks_by_seed.csv; must be updated to use phase6b_adversarial_w09.csv (R14-R19) before submission.
12. **fig5_frontier.py Panel B** — Feature-sign recovery values are STALE (batch-mixed pre-C0 MI); marked AUTHOR_PENDING in script. Must be replaced or figure panel removed before submission.

---

## FILES CHANGED SUMMARY

### Manuscript files
- `trust-ct/manuscript/02_methods.tex` — W02a (outcome), W04A (reframing), W05a (resamples), W11a-c (governance)
- `trust-ct/manuscript/03_results.tex` — W02c (longtail), W03b (NI values), W07 (cosines), W09b (utility), W09c (BSR), W10a (BSR rename), W11b (hash-chain)
- `Fedlearn/New/TRUST_CT_CMPB_Introduction (2).tex` — W11a (non-repudiation removed)
- `manuscript/figure_scripts/fig3_policy_replay.py` — pi_risk corrected 0.875→0.885
- `manuscript/figure_scripts/fig5_frontier.py` — batch1_cos populated, footer updated

### Code and patches
- `trust-ct/remediation/patches/w04_fl_fit_patch.py` — W04 fix
- `trust-ct/remediation/patches/w09_rng_patch.py` — W09 fix
- `trust-ct/run_phase6_w09_rerun.py` — W09 targeted rerun script (executed)

### Data artifacts (NEW, originals preserved)
- `trust-ct/processed/phase6/phase6b_adversarial_w09.csv` — W09a+W09c corrected (300 rows)
- `trust-ct/processed/phase6/phase6c_leakage_fl_w09.csv` — W09b corrected (25 rows)
- `trust-ct/remediation/selected_features.json` — W06 C0-correct feature export
- `trust-ct/remediation/w07_batch1_corrected_values.json` — W07 resolved cosines
- `trust-ct/remediation/master_results_ledger.csv` — W12 master ledger (29 rows)

### Reports
- `trust-ct/remediation/validation_report.json` — W15 machine validation
- `trust-ct/remediation/final_lock_sheet.md` — this document

---

## GATE CHECKS

| Check | Status | Note |
|-------|--------|------|
| No seeds/splits altered | PASS | All seeds/splits identical to originals |
| No favourable-result forcing | PASS | W09a shows STRONGER attack (worse for authors); W09b shows BETTER utility (favourable but reflects truth) |
| Patient data local-only | PASS | No clinical row data published or shared |
| No private keys disclosed | PASS | HMAC keys not in any committed file |
| No invented ethics approvals | PASS | Ethics items listed as AUTHOR_PENDING |
| Historical artifacts preserved | PASS | Original phase6b_adversarial.csv and phase6c_leakage_fl.csv unchanged |
| Null results preserved | PASS | Policy uplift CI crosses zero; reported as "not established" |
| W09 originals NOT overwritten | PASS | Corrected results in separate _w09 files |

---

## DECLARATION

This lock sheet reflects the machine-executable portion of W01–W15. The manuscript states results consistent with the locked values above. Items marked AUTHOR_PENDING cannot be resolved computationally and require the corresponding author's input before final submission.

**A valid null or unfavourable result is acceptable. Scientific success is not a software-correctness gate.**
