# Unresolved Items — Author Verification Required Before Submission
# TRUST-CT | CMPB Manuscript

All items below require author action before final submission.
Do NOT edit numerical values in the manuscript files without also updating `numerical_provenance.csv`.

---

## 1. Ethics and Regulatory

**1.1 BETTER-BP ethics approval reference**
- Required for Methods §2.2 and the journal's human-subjects checklist.
- Placeholder in manuscript: `[TODO: BETTER-BP ethics ref]`
- Action: Insert the exact IRB/REC approval number and institution.

**1.2 BETTER-BP ClinicalTrials.gov registration**
- Required for the BETTER-BP trial description in Methods §2.2.
- Placeholder in manuscript: `[TODO: ClinicalTrials.gov NCT...]`
- Action: Confirm the registration number and insert it.

**1.3 Consent procedure for secondary data analyses**
- eICU is de-identified public data (PhysioNet, credentialed access).
- BETTER-BP secondary analysis of trial data: confirm whether a waiver of consent was obtained or whether original trial consent covered secondary analysis.
- Placeholder in manuscript: `[TODO: consent statement for BETTER-BP secondary analysis]`
- Action: Insert one sentence in Methods §2.2.

**1.4 Cimas ethics approval reference**
- Confirm that the approval number already cited in prior submissions is current and correctly formatted.
- Placeholder in manuscript: `[TODO: Cimas ethics ref]`
- Action: Verify and insert.

---

## 2. Citation Keys (all marked TODO in .tex files)

| Placeholder key | Reference description | Location in manuscript |
|---|---|---|
| `TODO:pollard2018eicu` | Pollard et al. (2018) eICU collaborative research database | Methods §2.2; supplement S1 |
| `TODO:mcmahan2017fedavg` | McMahan et al. (2017) Communication-efficient FL (FedAvg) | Methods §2.4; supplement S4 |
| `TODO:karimireddy2020scaffold` | Karimireddy et al. (2020) SCAFFOLD | Methods §2.4; supplement S4 |
| `TODO:baruch2019alie` | Baruch et al. (2019) A Little Is Enough (ALIE) | Methods §2.7; supplement S2 |
| `TODO:li2020fedprox` | Li et al. (2020) FedProx | Methods §2.4 |
| `TODO:kennedy1983doublyrobust` or appropriate DR citation | Doubly robust / AIPW policy evaluation | Methods §2.5 |
| `TODO:zhao2018mia` or appropriate MIA citation | Membership inference attack | Methods §2.9 |
| `TODO:dimanov2021reconstruction` or appropriate gradient inversion citation | Gradient inversion / reconstruction | Methods §2.8 |

Action: Replace every `TODO:` key with valid BibTeX keys and create/update `manuscript/references.bib`.

---

## 3. Numerical Queries

**3.1 FedAdam NI CI upper bound**
- Table 2 row: FedAdam δAUROC = −0.005 [−0.011, 0.000].
- Query: The upper bound is exactly 0.000 (three decimal places). Confirm whether this should be displayed as `0.000` or `+0.001` (rounded up from a value just below 0).
- Source: `processed/cimas/results_phase3_freeze/ni_client_level.csv` column `hi_95`.
- Action: Read `ni_client_level.csv`, extract the FedAdam hi_95 value to 4+ decimal places, and update the table and provenance CSV if necessary.

**3.2 Long-tail external validation denominator**
- Results §3.2 states N=3,716 for the long-tail external validation.
- Confirm this is the correct denominator (not the full Cimas N=12,649).
- Source: `processed/cimas/results_phase3_freeze/gate_table_corrected.csv` — check `n_test` for long-tail rows.
- Action: Verify and update text if incorrect.

**3.3 FedAvg vs local-only CI**
- Results §3.2 states FedAvg exceeded local-only by +0.007 [−0.003, +0.018].
- Confirm this value is in the locked artifacts.
- Source: `processed/cimas/results_phase3_freeze/ni_client_level.csv` or `gate_table_corrected.csv`.
- Action: Locate the specific column/row containing the FL-vs-local comparison.

**3.4 α=0.05 utility cost point estimate**
- Table 3 row: α=0.05 gives ΔAUROC = −0.074 (no CI reported).
- This value was sourced from `processed/phase6/phase6d_frontier.csv`.
- Confirm whether a per-seed CI is available. If so, report it.
- Action: Read `phase6d_frontier.csv`, extract α=0.05 rows, compute CI if 5 seeds are present.

**3.5 Cimas cold-start nAULC: C2−C1 vs C3−C1**
- Results §3.3 reports both C2−C1 and C3−C1 as approximately 7.8×10⁻⁵.
- Confirm whether C2−C1 and C3−C1 are genuinely identical or just rounded to the same value.
- Source: `processed/phase5/results_v3/phase5_v3_lock.json`.
- Action: Extract both values to full precision; update text if they differ meaningfully.

---

## 4. Figures (not yet generated)

**4.1 Figure 1 — System workflow diagram**
- Author-drawn; no script.
- Must exclude: blockchain, PureChain, smart contracts, IPFS, secure aggregation, token rewards, gas cost.
- Must include: three cohort blocks, signed hash-linked audit log, perturbation block.
- See `figure_manifest.md` §Figure 1 for full element list.

**4.2 Figure 2 — NI forest plot**
- Script: `manuscript/figure_scripts/fig2_ni_forest.py` (NOT YET CREATED).
- Data: `processed/cimas/results_phase3_freeze/ni_client_level.csv`.

**4.3 Figure 3 — Policy value and closed-loop replay (two-panel)**
- Script: `manuscript/figure_scripts/fig3_policy_replay.py` (NOT YET CREATED).
- Panel A data: `processed/better_bp/results_phase4/policy_values.csv`.
- Panel B data: `processed/phase5/results_v3/cimas_cold_rounds.parquet`.

**4.4 Figure 4 — Attack degradation and robust aggregation recovery**
- Script: `manuscript/figure_scripts/fig4_attacks.py` (NOT YET CREATED).
- Data: `processed/phase6/phase6b_adversarial.csv`.
- Exclude Krum-NA rows (`run_status=not_applicable_diagnostic_fallback`).

**4.5 Figure 5 — Privacy–utility frontier (three-panel)**
- Script: `manuscript/figure_scripts/fig5_frontier.py` (NOT YET CREATED).
- Reconstruction cosines: use C4 corrected values (batch=1) from
  `processed/phase6_corrections/c4_leakage_corrected.json`.
- MIA AUROC: `processed/phase6/phase6c_mia.csv`.

---

## 5. Writing and Format Checks

**5.1 Word count validation**
- Methods target: ~1,350–1,450 words (excluding abstract and references).
- Results target: ~700–850 words.
- Action: Run `wc -w` or pandoc on `02_methods.tex` and `03_results.tex` after stripping LaTeX commands; confirm within CMPB budget.

**5.2 Equation numbering consistency**
- Equations are numbered (1)–(7) in the manuscript.
- Supplement algorithms reference `Eq.(ref{eq:dr})`, `Eq.(ref{eq:pert})`, etc.
- Action: Compile the full manuscript once to verify cross-references resolve.

**5.3 Table footnote package**
- Table 3 uses `\begin{tablenotes}` which requires the `threeparttable` package.
- Action: Confirm `\usepackage{threeparttable}` and `\begin{threeparttable}` wrappers are in the main .tex preamble.

**5.4 Algorithm package compatibility**
- Supplement uses `\usepackage{algorithm2e}` and `\usepackage{algorithmicx}`.
- These two packages can conflict. Action: Test compilation; switch to `algorithm2e` only if conflict occurs.

**5.5 Citation order**
- All citations should appear in numerical square-bracket order of first use.
- Action: After filling in all citation keys, verify numbered order is consistent.

---

## 6. Submission Checklist Items

- [ ] All TODO: citation keys resolved and .bib file complete
- [ ] Ethics statements inserted (items 1.1–1.4)
- [ ] ClinicalTrials.gov registration confirmed (item 1.2)
- [ ] All five figures generated from locked data (items 4.1–4.5)
- [ ] Word counts within CMPB budget (item 5.1)
- [ ] Full manuscript compiled without errors
- [ ] Supplementary file separately compiled
- [ ] Numerical values in manuscript cross-checked against `numerical_provenance.csv`
- [ ] No claim of formal DP, secure aggregation, blockchain, or policy-value improvement beyond what locked artifacts support
- [ ] Author list, affiliations, and corresponding author email confirmed
- [ ] Data availability statement drafted (eICU: PhysioNet; Cimas: restricted; BETTER-BP: per trial governance)
- [ ] Code availability statement drafted (link to repository with locked scripts)
