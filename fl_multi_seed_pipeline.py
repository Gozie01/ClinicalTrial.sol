import os
import json
import time
import hashlib
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers


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
# Data
# =========================================================
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if "Outcome" not in df.columns:
        raise ValueError("diabetes.csv must contain an 'Outcome' column.")

    X = df.drop(columns=["Outcome"]).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.int32)
    feature_names = df.drop(columns=["Outcome"]).columns.tolist()
    return X, y, feature_names


def split_sites_iid(X, y, n_sites):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    shards = np.array_split(idx, n_sites)
    return [(X[s], y[s]) for s in shards]


def split_sites_non_iid_by_label(X, y, n_sites, alpha=0.85):
    idx0 = np.where(y == 0)[0].tolist()
    idx1 = np.where(y == 1)[0].tolist()
    random.shuffle(idx0)
    random.shuffle(idx1)

    per_site = len(y) // n_sites
    sites = []

    for s in range(n_sites):
        dominant = 0 if s % 2 == 0 else 1
        n_dom = int(alpha * per_site)
        n_min = per_site - n_dom

        if dominant == 0:
            take0 = min(n_dom, len(idx0))
            take1 = min(n_min, len(idx1))
            chosen = idx0[:take0] + idx1[:take1]
            idx0 = idx0[take0:]
            idx1 = idx1[take1:]
        else:
            take1 = min(n_dom, len(idx1))
            take0 = min(n_min, len(idx0))
            chosen = idx1[:take1] + idx0[:take0]
            idx1 = idx1[take1:]
            idx0 = idx0[take0:]

        random.shuffle(chosen)
        chosen = np.array(chosen, dtype=int)
        sites.append((X[chosen], y[chosen]))

    leftover = np.array(idx0 + idx1, dtype=int)
    if len(leftover) > 0:
        for i, j in enumerate(leftover):
            s = i % n_sites
            Xs, ys = sites[s]
            sites[s] = (
                np.vstack([Xs, X[j:j+1]]),
                np.concatenate([ys, y[j:j+1]])
            )

    return sites


# =========================================================
# Model
# =========================================================
def build_mlp(input_dim: int, lr: float = 1e-3):
    l2 = regularizers.l2(1e-4)

    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, kernel_regularizer=l2)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(64, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def fedavg(weight_list):
    return [np.mean(ws, axis=0) for ws in zip(*weight_list)]


# =========================================================
# Metrics
# =========================================================
def best_threshold_by_f1(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        scores.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(ppv),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def hash_model_weights(weights):
    payload = b"".join([np.asarray(w).astype(np.float32).tobytes() for w in weights])
    return hashlib.sha256(payload).hexdigest()


# =========================================================
# Single-run FL experiment
# =========================================================
def run_single_experiment(
    seed=7,
    n_sites=3,
    rounds=20,
    local_epochs=10,
    batch_size=64,
    lr=1e-3,
    non_iid=False,
    use_smote=False,
):
    set_seed(seed)

    X, y, feature_names = load_data()

    # 70/10/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if non_iid:
        sites = split_sites_non_iid_by_label(X_train, y_train, n_sites=n_sites, alpha=0.85)
        partition_type = "non_iid"
    else:
        sites = split_sites_iid(X_train, y_train, n_sites=n_sites)
        partition_type = "iid"

    classes = np.unique(y_train)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    input_dim = X_train.shape[1]
    global_model = build_mlp(input_dim, lr=lr)

    history = []
    round_times = []

    print(f"\n=== Seed {seed} | Partition: {partition_type} | SMOTE: {use_smote} ===")

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        local_weights = []

        for site_id, (Xs, ys) in enumerate(sites, start=1):
            X_local, y_local = Xs, ys

            if use_smote:
                try:
                    X_local, y_local = SMOTE(
                        sampling_strategy="auto",
                        random_state=seed
                    ).fit_resample(X_local, y_local)
                except Exception:
                    pass

            local_model = build_mlp(input_dim, lr=lr)
            set_weights(local_model, get_weights(global_model))

            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=0
            )

            local_model.fit(
                X_local,
                y_local,
                validation_data=(X_val, y_val),
                epochs=local_epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                verbose=0,
                callbacks=[early_stop, reduce_lr]
            )

            local_weights.append(get_weights(local_model))

        aggregated = fedavg(local_weights)
        set_weights(global_model, aggregated)

        val_proba = global_model.predict(X_val, verbose=0).ravel()
        best_thr, best_val_f1 = best_threshold_by_f1(y_val, val_proba)

        test_proba = global_model.predict(X_test, verbose=0).ravel()
        metrics = compute_metrics(y_test, test_proba, best_thr)

        elapsed = time.time() - t0
        round_times.append(elapsed)

        metrics.update({
            "seed": int(seed),
            "round": int(rnd),
            "val_best_f1": float(best_val_f1),
            "elapsed_sec": float(elapsed),
            "sites": int(n_sites),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(lr),
            "partition_type": partition_type,
            "smote": bool(use_smote),
            "model_hash": hash_model_weights(get_weights(global_model)),
        })

        history.append(metrics)

        print(
            f"Seed {seed} | Round {rnd:02d} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"prec={metrics['precision']:.4f} | "
            f"rec={metrics['recall']:.4f} | "
            f"spec={metrics['specificity']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auc={metrics['auc']:.4f}"
        )

    history_df = pd.DataFrame(history)
    best_row = history_df.iloc[history_df["f1"].idxmax()].to_dict()
    best_row["avg_round_time_sec"] = float(np.mean(round_times))
    best_row["feature_count"] = int(len(feature_names))
    best_row["features"] = feature_names

    return history_df, best_row


# =========================================================
# Multi-seed aggregation
# =========================================================
def summarize_multi_seed(all_runs_df):
    final_round_df = all_runs_df.sort_values("round").groupby("seed", as_index=False).tail(1)

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
        "elapsed_sec",
    ]

    rows = []
    for metric in metrics:
        if metric in final_round_df.columns:
            mean_val = final_round_df[metric].mean()
            std_val = final_round_df[metric].std(ddof=1) if len(final_round_df) > 1 else 0.0
            rows.append({
                "Metric": metric,
                "Mean": round(float(mean_val), 6),
                "Std": round(float(std_val), 6),
                "Mean ± Std": f"{mean_val:.4f} ± {std_val:.4f}",
            })

    summary_df = pd.DataFrame(rows)
    return final_round_df, summary_df


def save_outputs(
    all_runs_df,
    best_rows,
    final_round_df,
    summary_df,
    partition_type,
    use_smote,
):
    run_tag = f"{partition_type}_{'smote' if use_smote else 'nosmote'}_multiseed"

    per_round_csv = EVAL_DIR / f"fl_metrics_{run_tag}.csv"
    best_json = EVAL_DIR / f"fl_best_runs_{run_tag}.json"
    final_round_csv = EVAL_DIR / f"fl_final_round_{run_tag}.csv"
    summary_csv = EVAL_DIR / f"fl_stat_summary_{run_tag}.csv"
    summary_json = EVAL_DIR / f"fl_stat_summary_{run_tag}.json"

    all_runs_df.to_csv(per_round_csv, index=False)

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_rows, f, indent=2)

    final_round_df.to_csv(final_round_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Saved Multi-Seed Outputs ===")
    print(per_round_csv)
    print(best_json)
    print(final_round_csv)
    print(summary_csv)
    print(summary_json)
    print("================================\n")


# =========================================================
# Main pipeline
# =========================================================
def run_multi_seed_pipeline(
    seeds,
    n_sites=3,
    rounds=20,
    local_epochs=10,
    batch_size=64,
    lr=1e-3,
    non_iid=False,
    use_smote=False,
):
    all_runs = []
    best_rows = []

    partition_type = "non_iid" if non_iid else "iid"

    for seed in seeds:
        history_df, best_row = run_single_experiment(
            seed=seed,
            n_sites=n_sites,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            non_iid=non_iid,
            use_smote=use_smote,
        )
        all_runs.append(history_df)
        best_rows.append(best_row)

    all_runs_df = pd.concat(all_runs, ignore_index=True)
    final_round_df, summary_df = summarize_multi_seed(all_runs_df)

    save_outputs(
        all_runs_df=all_runs_df,
        best_rows=best_rows,
        final_round_df=final_round_df,
        summary_df=summary_df,
        partition_type=partition_type,
        use_smote=use_smote,
    )

    print("=== Final-Round Multi-Seed Summary ===")
    print(summary_df.to_string(index=False))
    print("======================================")

    return all_runs_df, final_round_df, summary_df


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--smote", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,42,99,123,2024",
        help="Comma-separated seeds, e.g. 7,42,99,123,2024"
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    run_multi_seed_pipeline(
        seeds=seeds,
        n_sites=args.sites,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        non_iid=args.non_iid,
        use_smote=args.smote,
    )