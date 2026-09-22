import os
import json
import time
import random
import hashlib
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import *

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# ============================
# Reproducibility
# ============================
def set_seed(seed=7):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================
# Paths
# ============================
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetes.csv"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)


# ============================
# Data Loading
# ============================
def load_data():
    df = pd.read_csv(DATA_FILE)
    X = df.drop("Outcome", axis=1).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.int32)
    return X, y


# ============================
# Site Splitting
# ============================
def split_iid(X, y, n_sites):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    return [(X[s], y[s]) for s in np.array_split(idx, n_sites)]


def split_non_iid(X, y, n_sites, alpha=0.85):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    sites = []
    per_site = len(y) // n_sites

    for i in range(n_sites):
        dom = 0 if i % 2 == 0 else 1
        n_dom = int(alpha * per_site)
        n_min = per_site - n_dom

        if dom == 0:
            sel = np.concatenate([idx0[:n_dom], idx1[:n_min]])
            idx0 = idx0[n_dom:]
            idx1 = idx1[n_min:]
        else:
            sel = np.concatenate([idx1[:n_dom], idx0[:n_min]])
            idx1 = idx1[n_dom:]
            idx0 = idx0[n_min:]

        np.random.shuffle(sel)
        sites.append((X[sel], y[sel]))

    return sites


# ============================
# Model
# ============================
def build_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def fedavg(weights):
    return [np.mean(w, axis=0) for w in zip(*weights)]


# ============================
# Metrics
# ============================
def best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    idx = np.argmax(scores)
    return thresholds[idx], scores[idx]


def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "specificity": tn / (tn + fp + 1e-8),
        "sensitivity": tp / (tp + fn + 1e-8),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


# ============================
# Main FL Pipeline
# ============================
def run_fl(n_sites=3, rounds=20, non_iid=False, smote=False):

    set_seed(7)
    X, y = load_data()

    # Split 70/10/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, stratify=y_train)

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    # Split sites
    sites = split_non_iid(X_train, y_train, n_sites) if non_iid else split_iid(X_train, y_train, n_sites)

    model = build_model(X.shape[1])

    history = []

    for r in range(rounds):
        local_weights = []

        for Xs, ys in sites:

            if smote:
                try:
                    Xs, ys = SMOTE().fit_resample(Xs, ys)
                except:
                    pass

            local_model = build_model(X.shape[1])
            local_model.set_weights(model.get_weights())

            local_model.fit(Xs, ys, epochs=10, batch_size=64, verbose=0)

            local_weights.append(local_model.get_weights())

        # FedAvg
        model.set_weights(fedavg(local_weights))

        # Validation threshold tuning
        val_prob = model.predict(X_val, verbose=0).ravel()
        thr_tuned, _ = best_threshold(y_val, val_prob)

        test_prob = model.predict(X_test, verbose=0).ravel()

        # Compute BOTH metrics
        m_tuned = compute_metrics(y_test, test_prob, thr_tuned)
        m_fixed = compute_metrics(y_test, test_prob, 0.5)

        record = {"round": r + 1}

        for k, v in m_tuned.items():
            record[f"{k}_tuned"] = v

        for k, v in m_fixed.items():
            record[f"{k}_fixed"] = v

        record["threshold_tuned"] = thr_tuned

        history.append(record)

        print(f"Round {r+1}: F1_tuned={m_tuned['f1']:.4f} | F1_fixed={m_fixed['f1']:.4f}")

    df = pd.DataFrame(history)

    tag = f"{'non_iid' if non_iid else 'iid'}_{'smote' if smote else 'nosmote'}"
    df.to_csv(EVAL_DIR / f"fl_metrics_{tag}.csv", index=False)

    return df


# ============================
# CLI
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--smote", action="store_true")
    args = parser.parse_args()

    run_fl(non_iid=args.non_iid, smote=args.smote)