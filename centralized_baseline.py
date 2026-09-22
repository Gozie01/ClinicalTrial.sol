import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetes.csv"


def build_mlp(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_mlp(X_train.shape[1])

    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    y_proba = model.predict(X_test).ravel()

    # FIXED threshold baseline
    y_pred_fixed = (y_proba >= 0.5).astype(int)

    metrics_fixed = compute_metrics(y_test, y_pred_fixed, y_proba)

    # TUNED threshold
    best_thr = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 50):
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    y_pred_tuned = (y_proba >= best_thr).astype(int)
    metrics_tuned = compute_metrics(y_test, y_pred_tuned, y_proba)

    print("\n=== Centralized MLP ===")
    print("Fixed threshold (0.5):", metrics_fixed)
    print("Tuned threshold:", metrics_tuned)


if __name__ == "__main__":
    run()