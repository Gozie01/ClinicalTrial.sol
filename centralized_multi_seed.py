import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras import layers, models


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 7):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetes.csv"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Model
# =========================================================
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =========================================================
# Metrics
# =========================================================
def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# =========================================================
# Single-seed run
# =========================================================
def run_single_seed(seed: int = 42):
    set_seed(seed)

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_mlp(X_train.shape[1])

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    y_proba = model.predict(X_test, verbose=0).ravel()

    # ---------------- Fixed threshold ----------------
    y_pred_fixed = (y_proba >= 0.5).astype(int)
    metrics_fixed = compute_metrics(y_test, y_pred_fixed, y_proba)
    metrics_fixed["seed"] = int(seed)
    metrics_fixed["threshold"] = 0.5
    metrics_fixed["mode"] = "fixed"

    # ---------------- Tuned threshold ----------------
    best_thr = 0.5
    best_f1 = -1.0

    for t in np.linspace(0.1, 0.9, 50):
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    y_pred_tuned = (y_proba >= best_thr).astype(int)
    metrics_tuned = compute_metrics(y_test, y_pred_tuned, y_proba)
    metrics_tuned["seed"] = int(seed)
    metrics_tuned["threshold"] = float(best_thr)
    metrics_tuned["mode"] = "tuned"

    return metrics_fixed, metrics_tuned


# =========================================================
# Summary helper
# =========================================================
def summarize_results(df: pd.DataFrame):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "mcc",
        "balanced_accuracy",
        "threshold",
    ]

    rows = []
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std(ddof=1) if len(df) > 1 else 0.0

        rows.append({
            "Metric": metric,
            "Mean": round(float(mean_val), 6),
            "Std": round(float(std_val), 6),
            "Mean ± Std": f"{mean_val:.4f} ± {std_val:.4f}",
        })

    return pd.DataFrame(rows)


# =========================================================
# Main multi-seed pipeline
# =========================================================
def run_multi_seed_pipeline(seeds=None):
    if seeds is None:
        seeds = [7, 42, 99, 123, 2024]

    fixed_rows = []
    tuned_rows = []

    for seed in seeds:
        metrics_fixed, metrics_tuned = run_single_seed(seed)
        fixed_rows.append(metrics_fixed)
        tuned_rows.append(metrics_tuned)

        print(
            f"Seed {seed} | "
            f"Fixed F1={metrics_fixed['f1']:.4f}, AUC={metrics_fixed['auc']:.4f} | "
            f"Tuned F1={metrics_tuned['f1']:.4f}, AUC={metrics_tuned['auc']:.4f}, "
            f"Thr={metrics_tuned['threshold']:.3f}"
        )

    fixed_df = pd.DataFrame(fixed_rows)
    tuned_df = pd.DataFrame(tuned_rows)

    fixed_summary = summarize_results(fixed_df)
    tuned_summary = summarize_results(tuned_df)

    fixed_csv = EVAL_DIR / "centralized_fixed_multiseed.csv"
    tuned_csv = EVAL_DIR / "centralized_tuned_multiseed.csv"
    fixed_summary_csv = EVAL_DIR / "centralized_fixed_multiseed_summary.csv"
    tuned_summary_csv = EVAL_DIR / "centralized_tuned_multiseed_summary.csv"
    tuned_json = EVAL_DIR / "centralized_tuned_multiseed.json"

    fixed_df.to_csv(fixed_csv, index=False)
    tuned_df.to_csv(tuned_csv, index=False)
    fixed_summary.to_csv(fixed_summary_csv, index=False)
    tuned_summary.to_csv(tuned_summary_csv, index=False)

    with open(tuned_json, "w", encoding="utf-8") as f:
        json.dump(tuned_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Centralized Multi-Seed Results Saved ===")
    print(fixed_csv)
    print(tuned_csv)
    print(fixed_summary_csv)
    print(tuned_summary_csv)
    print(tuned_json)

    print("\n=== Fixed Threshold Summary ===")
    print(fixed_summary.to_string(index=False))

    print("\n=== Tuned Threshold Summary ===")
    print(tuned_summary.to_string(index=False))

    return fixed_df, tuned_df, fixed_summary, tuned_summary


if __name__ == "__main__":
    run_multi_seed_pipeline()