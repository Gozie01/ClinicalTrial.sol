"""
BETTER-BP data adapter for TRUST-CT.

Loads processed parquet files from Fedlearn/processed/better_bp/
(produced by better_bp_harmonize.py — P1 audit, all 14 checks pass).

Provides:
  - load_betterbp_fl()    — two-client {A, B} feature/outcome dicts for FL
  - load_betterbp_policy() — full cohort for doubly-robust policy estimation

Primary outcome for FL:  nonattendance_12m (Y=1 if no 12m visit)
Treatment indicator:     arm == "Intervention" (incentive arm, e=2/3)
Sites as FL clients:     A (n=312) and B (n=90)

BETTER-BP does NOT contain:
  - Daily bottle-opening timestamps (AdhereTech rows = setup records)
  - Daily/weekly adherence percentages
  - Adequate-adherence outcome (cannot reproduce published 71%/34%)
  - Lottery draws or incentive payment amounts

See P1 audit (better_bp_harmonize.py) for full data inventory.
"""

import pathlib
import numpy as np
import pandas as pd

_PROC = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "Fedlearn"
    / "processed"
    / "better_bp"
)

FEATURE_COLS = [
    "age", "sex_female", "hispanic", "race_black", "race_white",
    "ins_medicaid", "ins_medicare", "ins_private",
    "bmi", "sbp_baseline", "dbp_baseline",
    "mases", "tsrq_autonomous", "tsrq_controlled",
    "charlson", "phq8", "sf12_mcs", "sf12_pcs",
]


def _load_trial_outcomes() -> pd.DataFrame:
    path = _PROC / "trial_outcomes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Run better_bp_harmonize.py first to produce {path}"
        )
    return pd.read_parquet(path)


def _load_baseline() -> pd.DataFrame:
    path = _PROC / "participants_baseline.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)


def load_betterbp_fl(
    outcome: str = "nonattendance_12m",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {site: (X, y)} for sites A and B.
    Site C excluded (0 randomized participants).
    """
    outcomes = _load_trial_outcomes()
    baseline = _load_baseline()

    merged = outcomes.merge(baseline, on="participant_id", how="inner")
    merged = merged[merged["site"].isin(["A", "B"])].copy()

    avail_feats = [c for c in FEATURE_COLS if c in merged.columns]
    client_data = {}
    for site, grp in merged.groupby("site"):
        X = grp[avail_feats].values.astype(float)
        y = grp[outcome].values.astype(int)
        client_data[f"site_{site}"] = (X, y)
    return client_data


def load_betterbp_policy(
    outcome: str = "nonattendance_12m",
    arm_col: str = "arm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y, a) for doubly-robust policy estimation.
    a=1 for Intervention arm, a=0 for Control.
    Known propensity e = P(A=1) = 2/3.
    """
    outcomes = _load_trial_outcomes()
    baseline = _load_baseline()
    merged = outcomes.merge(baseline, on="participant_id", how="inner")
    merged = merged[merged["site"].isin(["A", "B"])].copy()

    avail_feats = [c for c in FEATURE_COLS if c in merged.columns]
    X = merged[avail_feats].values.astype(float)
    y = merged[outcome].values.astype(int)
    a = (merged[arm_col].str.lower() == "intervention").astype(int).values
    return X, y, a
