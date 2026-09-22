"""
TRUST-CT Closed-Loop Simulation — C0–C3 conditions.

This is a SEMI-SYNTHETIC closed-loop replay calibrated using real
randomised-trial data (BETTER-BP) and real longitudinal-claims data (Cimas).
It is NOT prospective clinical validation.

Conditions:
  C0 — No incentive (pure observation baseline)
  C1 — Static universal incentive (treat everyone)
  C2 — Risk-based: treat top-K% by predicted nonattendance risk
  C3 — Budgeted uplift + fairness: cross-fitted DR CATE, budget constraint

Primary comparison: C3 vs C1 (same budget, smarter allocation).
Secondary: C3 vs C2 (uplift vs risk-only targeting).

DQN note:
  DQN attendance 0.4117 < static 0.4448 — DQN does NOT outperform static
  in this simulation. DQN is retained ONLY as a model-based comparator.
  Remove 'reinforcement-driven' from the main contribution statement.
  Primary policy = cross-fitted doubly robust (C3).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from policy.doubly_robust import budget_allocation, evaluate_policy


CONDITIONS = ["C0", "C1", "C2", "C3"]


def simulate_condition(
    condition: str,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    e: float,
    tau: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
    risk_score: Optional[np.ndarray] = None,
    budget_fraction: float = 0.33,
) -> dict:
    """
    Simulate one closed-loop condition and return IPW policy value.

    Args:
        condition: "C0", "C1", "C2", or "C3"
        X: feature matrix (n, p)
        y: outcome vector (n,)
        a: observed treatment (n,) — from RCT
        e: known propensity P(A=1)
        tau: DR uplift estimates (required for C3)
        mu0: baseline risk estimates (required for C2, C3 tie-break)
        risk_score: predicted risk for C2 allocation
        budget_fraction: fraction of patients to incentivise

    Returns:
        dict with condition, ipw_value, treat_rate, n_treated
    """
    n = len(y)
    budget_n = int(n * budget_fraction)

    if condition == "C0":
        alloc = np.zeros(n, dtype=int)

    elif condition == "C1":
        alloc = np.ones(n, dtype=int)

    elif condition == "C2":
        if risk_score is None:
            raise ValueError("C2 requires risk_score (predicted nonattendance)")
        alloc = np.zeros(n, dtype=int)
        top = np.argsort(-risk_score)[:budget_n]
        alloc[top] = 1

    elif condition == "C3":
        if tau is None or mu0 is None:
            raise ValueError("C3 requires tau (uplift) and mu0 (baseline risk)")
        alloc = budget_allocation(tau, mu0, budget_n)

    else:
        raise ValueError(f"Unknown condition: {condition!r}. Must be one of {CONDITIONS}")

    result = evaluate_policy(y, a, alloc, e=e)
    result["condition"] = condition
    result["budget_fraction"] = budget_fraction
    return result


def run_all_conditions(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    e: float,
    tau: np.ndarray,
    mu0: np.ndarray,
    risk_score: np.ndarray,
    budget_fraction: float = 0.33,
    label: str = "",
) -> pd.DataFrame:
    """
    Run all four conditions and return a comparison DataFrame.
    """
    rows = []
    for cond in CONDITIONS:
        row = simulate_condition(
            cond, X, y, a, e,
            tau=tau, mu0=mu0,
            risk_score=risk_score,
            budget_fraction=budget_fraction,
        )
        row["label"] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["condition", "ipw_value", "treat_rate", "n_treated", "n_total",
               "budget_fraction", "label", "note"]]
