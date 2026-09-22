# TRUST-CT: A Trial-Calibrated Federated Learning Framework for Participation Policy, Security, and Auditable Clinical Workflows

**TRUST-CT** is a closed-loop federated learning framework for clinical trial management. It integrates privacy-preserving federated training, doubly-robust participation-policy evaluation, blockchain-based audit logging, and adversarial robustness benchmarking across three real-world clinical datasets.

This repository accompanies the manuscript submitted to *Computers in Biology and Medicine* (CMPB).

---

## Datasets

| Cohort | Sites | N | Purpose |
|---|---|---|---|
| Cimas | K = 23 providers | 12,649 | Adherence prediction (Phases 2–3, E6) |
| BETTER-BP | K = 2 sites | 402 | Policy evaluation / participation (Phase 4–5) |
| eICU-CRD Demo | K = 8 hospital groups | 2,520 | Security & privacy benchmarks (Phase 6) |

Raw datasets are not included. Processed results (CSVs, JSON lock files) are in `trust-ct/processed/`.

---

## Repository Structure

```
trust-ct/
  federated/          # FL algorithms: FedAvg, FedProx, SCAFFOLD, FedAdam
  security/           # Byzantine attack families (7) and defenses (5)
  policy/             # Doubly-robust uplift estimator
  cohort/             # Cimas landmark cohort builder
  data_adapters/      # BETTER-BP and Cimas data adapters
  models/             # Tabular MLP model
  closed_loop/        # Phase 5 closed-loop replay runner
  governance/         # Blockchain oracle interface
  configs/            # YAML configs for each phase
  processed/          # Locked results (CSVs, JSON, Parquet)
  manuscript/         # LaTeX methods/results and figure scripts
  run_phase*.py       # Phase execution scripts (2–6)

manuscript/
  figure_scripts/     # Figure generation scripts (figs 1–6)
  figures/            # Generated figure PNGs

contracts/            # Solidity smart contracts (CTToken, TrialManager, etc.)
certora/              # Certora formal verification spec
echidna/              # Echidna fuzz test contracts
script/               # Foundry deployment scripts
test/                 # Foundry Solidity test suite
```

---

## Environment

Python 3.11, virtual environment `.venv311`.

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Or use the Conda environment:

```bash
conda env create -f trust-ct/environment.yml
conda activate trust-ct
```

---

## Running the Pipeline

Each phase produces locked results in `trust-ct/processed/`. Run phases in order:

```powershell
# Phase 2 — Cimas cohort FL training
python trust-ct/run_phase2_cimas.py

# Phase 3 — Non-inferiority evaluation (freeze)
python trust-ct/run_phase3_freeze.py

# Phase 4 — BETTER-BP doubly-robust policy evaluation
python trust-ct/run_phase4_betterbp.py

# Phase 5 — Closed-loop replay (v3)
python trust-ct/run_phase5_v3.py

# Phase 6 — Security and privacy benchmarks (W09-corrected)
python trust-ct/run_phase6_w09_rerun.py

# E6 — Fairness experiment (warm/cold federated de-confound)
python trust-ct/run_e6_fairness.py
```

---

## Generating Figures

All six manuscript figures can be regenerated from locked results:

```powershell
# PDF output (figs 1–5 via _pkg_work runner; fig 6 direct)
python trust-ct/_pkg_work/manuscript/figure_scripts/generate_all_figures_v2.py

# PNG output (figs 1–5)
python trust-ct/_pkg_work/manuscript/figure_scripts/generate_png_figures.py

# Fig 6 standalone (saves both PDF and PNG)
python manuscript/figure_scripts/fig6_e6_fairness.py
```

PNGs are written to `manuscript/figures/` and `trust-ct/_pkg_work/manuscript/figures/`.

---

## Blockchain Layer

Smart contracts are written in Solidity and use the Foundry toolchain.

```bash
forge build
forge test

# Formal verification (requires Certora Prover)
certoraRun certora/TrustCTGovernance.spec

# Fuzz testing (requires Echidna)
echidna echidna/TrustCTEchidna.sol --config echidna.yaml
```

---

## Key Locked Results

| Metric | Value | 95% CI |
|---|---|---|
| Cimas FedAvg NI (AUROC diff) | −0.002135 | [−0.005032, +0.000768] |
| eICU clean FedAvg AUROC | 0.8904 | [0.8892, 0.8913] |
| Label-flip degradation | −0.120 | [−0.178, −0.061] |
| Coord-median recovery | +0.111 | [+0.058, +0.165] |
| Policy uplift contrast | +0.032 | [−0.011, +0.078] |
| Fault injection detection | 350/350 | — |

Full provenance in `trust-ct/processed/` and `trust-ct/remediation/final_lock_sheet.md`.

---

## Citation

> Nnadiekwe, C.A. et al. TRUST-CT: A Trial-Calibrated Federated Learning Framework for Participation Policy, Security, and Auditable Clinical Workflows. *Computers in Biology and Medicine* (under review), 2026.
