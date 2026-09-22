import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers

from ct_simulation import generate_synthetic_ct_dataset
from eicu_dataset import prepare_eicu_demo_dataset


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetes.csv"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 7):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(
    dataset: str = "pima",
    seed: int = 7,
    synthetic_sites: int = 8,
    synthetic_participants_per_site: int = 120,
    synthetic_visits: int = 12,
    synthetic_n_samples: int = 12000,
    eicu_site_groups: int = 8,
):
    if dataset == "pima":
        df = pd.read_csv(DATA_FILE)
        X = df.drop(columns=["Outcome"]).values.astype(np.float32)
        y = df["Outcome"].values.astype(np.int32)
        feature_names = df.drop(columns=["Outcome"]).columns.tolist()
        site_ids = None
    elif dataset == "breast_cancer":
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        X = df.drop(columns=["target"]).values.astype(np.float32)
        y = df["target"].values.astype(np.int32)
        feature_names = df.drop(columns=["target"]).columns.tolist()
        site_ids = None
    elif dataset == "synthetic_ct":
        df = generate_synthetic_ct_dataset(
            n_sites=synthetic_sites,
            participants_per_site=synthetic_participants_per_site,
            visits_per_participant=synthetic_visits,
            random_state=seed,
        )
        site_ids = df["site_id"].values.astype(np.int32)
        drop_cols = ["Outcome", "site_id", "participant_id", "visit_id"]
        X = df.drop(columns=drop_cols).values.astype(np.float32)
        y = df["Outcome"].values.astype(np.int32)
        feature_names = df.drop(columns=drop_cols).columns.tolist()
    elif dataset == "synthetic_large":
        X, y = make_classification(
            n_samples=synthetic_n_samples,
            n_features=32,
            n_informative=14,
            n_redundant=8,
            n_clusters_per_class=2,
            class_sep=1.0,
            weights=[0.56, 0.44],
            flip_y=0.04,
            random_state=seed,
        )
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        feature_names = [f"feature_{idx}" for idx in range(X.shape[1])]
        site_ids = None
    elif dataset == "eicu_demo":
        df, _, _ = prepare_eicu_demo_dataset(n_site_groups=eicu_site_groups, refresh=False)
        site_ids = df["site_group"].values.astype(np.int32)
        drop_cols = ["patientunitstayid", "hospitalid", "site_group", "Outcome"]
        X = df.drop(columns=drop_cols).values.astype(np.float32)
        y = df["Outcome"].values.astype(np.int32)
        feature_names = df.drop(columns=drop_cols).columns.tolist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return X, y, feature_names, site_ids


def split_sites_iid(X, y, n_sites, rng):
    idx = np.arange(len(y))
    rng.shuffle(idx)
    shards = np.array_split(idx, n_sites)
    return [(X[s], y[s]) for s in shards]


def split_sites_non_iid_by_label(X, y, n_sites, rng, alpha=0.85):
    idx0 = np.where(y == 0)[0].tolist()
    idx1 = np.where(y == 1)[0].tolist()
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    per_site = len(y) // n_sites
    sites = []

    for site_idx in range(n_sites):
        dominant = 0 if site_idx % 2 == 0 else 1
        n_dom = int(alpha * per_site)
        n_min = per_site - n_dom
        if dominant == 0:
            chosen = idx0[:n_dom] + idx1[:n_min]
            idx0 = idx0[n_dom:]
            idx1 = idx1[n_min:]
        else:
            chosen = idx1[:n_dom] + idx0[:n_min]
            idx1 = idx1[n_dom:]
            idx0 = idx0[n_min:]
        rng.shuffle(chosen)
        chosen = np.array(chosen, dtype=int)
        sites.append((X[chosen], y[chosen]))

    leftovers = np.array(idx0 + idx1, dtype=int)
    for offset, sample_idx in enumerate(leftovers):
        site_idx = offset % n_sites
        Xs, ys = sites[site_idx]
        sites[site_idx] = (np.vstack([Xs, X[sample_idx : sample_idx + 1]]), np.concatenate([ys, y[sample_idx : sample_idx + 1]]))

    return sites


def split_sites_from_site_ids(X, y, site_ids):
    sites = []
    for site_idx in sorted(np.unique(site_ids)):
        mask = site_ids == site_idx
        sites.append((X[mask], y[mask]))
    return sites


def build_mlp(input_dim: int, lr: float = 1e-3, model_family: str = "default"):
    if model_family == "tabular_small":
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.Dropout(0.15)(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    l2 = regularizers.l2(1e-4)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def hash_model_weights(weights):
    payload = b"".join(np.asarray(weight).astype(np.float32).tobytes() for weight in weights)
    return hashlib.sha256(payload).hexdigest()


def coordinate_median(weight_list):
    return [np.median(np.stack(weights, axis=0), axis=0) for weights in zip(*weight_list)]


def trimmed_mean(weight_list, trim_ratio: float = 0.2):
    aggregated = []
    for weights in zip(*weight_list):
        stacked = np.stack(weights, axis=0)
        if stacked.shape[0] < 3:
            aggregated.append(np.mean(stacked, axis=0))
            continue
        lower = int(trim_ratio * stacked.shape[0])
        upper = stacked.shape[0] - lower
        sorted_vals = np.sort(stacked, axis=0)
        trimmed = sorted_vals[lower:upper]
        aggregated.append(np.mean(trimmed, axis=0))
    return aggregated


def aggregate_weights(weight_list, strategy: str = "mean", client_sizes=None):
    if strategy == "mean":
        if client_sizes is None:
            return [np.mean(np.stack(weights, axis=0), axis=0) for weights in zip(*weight_list)]
        total = float(np.sum(client_sizes))
        normalized = [size / total for size in client_sizes]
        aggregated = []
        for weights in zip(*weight_list):
            stacked = np.stack(weights, axis=0)
            agg = np.tensordot(np.asarray(normalized, dtype=np.float32), stacked, axes=(0, 0))
            aggregated.append(agg)
        return aggregated
    if strategy == "median":
        return coordinate_median(weight_list)
    if strategy == "trimmed_mean":
        return trimmed_mean(weight_list)
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")


def best_threshold_by_f1(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
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
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def estimate_privacy_budget_upper_bound(
    num_steps: int,
    sampling_rate: float,
    noise_multiplier: float,
    delta: float,
):
    if noise_multiplier <= 0:
        return np.inf
    epsilon = sampling_rate * np.sqrt(2.0 * num_steps * np.log(1.0 / max(delta, 1e-12))) / noise_multiplier
    return float(epsilon)


def maybe_flip_labels(y, attack_strength: float, rng):
    y_new = y.copy()
    flip_mask = rng.uniform(size=len(y_new)) < attack_strength
    y_new[flip_mask] = 1 - y_new[flip_mask]
    return y_new


def apply_weight_attack(global_weights, local_weights, attack_type: str, attack_strength: float, rng):
    if attack_type == "none":
        return local_weights
    if attack_type == "gaussian":
        return [weight + rng.normal(0.0, attack_strength, size=weight.shape).astype(np.float32) for weight in local_weights]
    if attack_type in {"sign_flip", "gradient_poisoning"}:
        attacked = []
        for global_weight, local_weight in zip(global_weights, local_weights):
            delta = local_weight - global_weight
            attacked.append(global_weight - attack_strength * delta)
        return attacked
    if attack_type == "byzantine":
        attacked = []
        for weight in local_weights:
            attacked.append(rng.normal(0.0, 1.0 + attack_strength, size=weight.shape).astype(np.float32))
        return attacked
    raise ValueError(f"Unsupported weight attack type: {attack_type}")


def train_local_model(
    model,
    X_local,
    y_local,
    X_val,
    y_val,
    batch_size: int,
    local_epochs: int,
    class_weight: dict,
    use_dp: bool,
    dp_clip_norm: float,
    dp_noise_multiplier: float,
):
    if not use_dp:
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)
        model.fit(
            X_local,
            y_local,
            validation_data=(X_val, y_val),
            epochs=local_epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            verbose=0,
            callbacks=[early_stop, reduce_lr],
        )
        return 0

    optimizer = optimizers.Adam(learning_rate=model.optimizer.learning_rate.numpy())
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    dataset = tf.data.Dataset.from_tensor_slices((X_local.astype(np.float32), y_local.astype(np.float32)))
    dataset = dataset.shuffle(buffer_size=len(X_local), reshuffle_each_iteration=True).batch(batch_size)

    steps = 0
    for _ in range(local_epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss = loss_fn(tf.expand_dims(y_batch, axis=1), y_pred)
                if model.losses:
                    loss += tf.add_n(model.losses)

            grads = tape.gradient(loss, model.trainable_variables)
            global_norm = tf.linalg.global_norm(grads)
            clip_coef = tf.minimum(1.0, dp_clip_norm / (global_norm + 1e-6))
            clipped = [grad * clip_coef for grad in grads]

            if dp_noise_multiplier > 0:
                noisy = []
                for grad in clipped:
                    noise = tf.random.normal(
                        tf.shape(grad),
                        stddev=dp_noise_multiplier * dp_clip_norm / max(1, batch_size),
                    )
                    noisy.append(grad + noise)
                clipped = noisy

            optimizer.apply_gradients(zip(clipped, model.trainable_variables))
            steps += 1
    return steps


def run_experiment(
    n_sites: int = 3,
    rounds: int = 20,
    local_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    non_iid: bool = False,
    use_smote: bool = False,
    seed: int = 7,
    dataset: str = "pima",
    aggregation: str = "mean",
    dp_clip_norm: float = 1.0,
    dp_noise_multiplier: float = 0.0,
    dp_delta: float = 1e-5,
    attack_type: str = "none",
    attack_fraction: float = 0.0,
    attack_strength: float = 0.5,
    synthetic_sites: int = 8,
    synthetic_participants_per_site: int = 120,
    synthetic_visits: int = 12,
    synthetic_n_samples: int = 12000,
    eicu_site_groups: int = 8,
    feature_selection_top_k: int = 0,
    model_family: str = "default",
):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    X, y, feature_names, site_ids = load_data(
        dataset=dataset,
        seed=seed,
        synthetic_sites=synthetic_sites,
        synthetic_participants_per_site=synthetic_participants_per_site,
        synthetic_visits=synthetic_visits,
        synthetic_n_samples=synthetic_n_samples,
        eicu_site_groups=eicu_site_groups,
    )

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train,
        y_train,
        idx_train,
        test_size=0.125,
        stratify=y_train,
        random_state=seed,
    )

    if feature_selection_top_k > 0 and feature_selection_top_k < X_train.shape[1]:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
        top_indices = np.argsort(mi_scores)[::-1][:feature_selection_top_k]
        X_train = X_train[:, top_indices]
        X_val = X_val[:, top_indices]
        X_test = X_test[:, top_indices]
        feature_names = [feature_names[idx] for idx in top_indices]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if dataset in {"synthetic_ct", "eicu_demo"} and site_ids is not None:
        train_site_ids = site_ids[idx_train]
        sites = split_sites_from_site_ids(X_train, y_train, train_site_ids)
        partition_type = "natural_site_split"
        n_sites = len(sites)
    elif non_iid:
        sites = split_sites_non_iid_by_label(X_train, y_train, n_sites=n_sites, rng=rng)
        partition_type = "non_iid"
    else:
        sites = split_sites_iid(X_train, y_train, n_sites=n_sites, rng=rng)
        partition_type = "iid"

    classes = np.unique(y_train)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    input_dim = X_train.shape[1]
    global_model = build_mlp(input_dim, lr=lr, model_family=model_family)
    use_dp = dp_noise_multiplier > 0.0

    history = []
    round_times = []
    total_dp_steps = 0
    malicious_clients = max(0, int(round(len(sites) * attack_fraction)))
    model_bytes = np.sum([np.asarray(weight).size for weight in global_model.get_weights()]) * 4

    print("\n=== Federated Learning Run Started ===")
    print(f"Dataset          : {dataset}")
    print(f"Sites            : {n_sites}")
    print(f"Rounds           : {rounds}")
    print(f"Local epochs     : {local_epochs}")
    print(f"Batch size       : {batch_size}")
    print(f"Learning rate    : {lr}")
    print(f"Partition        : {partition_type}")
    print(f"SMOTE            : {use_smote}")
    print(f"Aggregation      : {aggregation}")
    print(f"DP enabled       : {use_dp}")
    print(f"Attack           : {attack_type} ({attack_fraction:.2f})")
    print("======================================\n")

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        local_weights = []
        client_sizes = []
        malicious_indices = set(rng.choice(len(sites), size=malicious_clients, replace=False).tolist()) if malicious_clients > 0 else set()

        for site_idx, (Xs, ys) in enumerate(sites):
            X_local, y_local = Xs.copy(), ys.copy()

            if site_idx in malicious_indices and attack_type == "label_flip":
                y_local = maybe_flip_labels(y_local, attack_strength=attack_strength, rng=rng)

            if use_smote:
                try:
                    X_local, y_local = SMOTE(sampling_strategy="auto", random_state=seed).fit_resample(X_local, y_local)
                except Exception:
                    pass

            local_model = build_mlp(input_dim, lr=lr, model_family=model_family)
            set_weights(local_model, get_weights(global_model))

            dp_steps = train_local_model(
                model=local_model,
                X_local=X_local,
                y_local=y_local,
                X_val=X_val,
                y_val=y_val,
                batch_size=batch_size,
                local_epochs=local_epochs,
                class_weight=class_weight,
                use_dp=use_dp,
                dp_clip_norm=dp_clip_norm,
                dp_noise_multiplier=dp_noise_multiplier,
            )
            total_dp_steps += dp_steps

            trained_weights = get_weights(local_model)
            if site_idx in malicious_indices and attack_type in {"gaussian", "sign_flip", "gradient_poisoning", "byzantine"}:
                trained_weights = apply_weight_attack(
                    global_weights=get_weights(global_model),
                    local_weights=trained_weights,
                    attack_type=attack_type,
                    attack_strength=attack_strength,
                    rng=rng,
                )

            local_weights.append(trained_weights)
            client_sizes.append(len(y_local))

        aggregated = aggregate_weights(local_weights, strategy=aggregation, client_sizes=client_sizes)
        set_weights(global_model, aggregated)

        val_proba = global_model.predict(X_val, verbose=0).ravel()
        best_thr, best_val_f1 = best_threshold_by_f1(y_val, val_proba)
        test_proba = global_model.predict(X_test, verbose=0).ravel()
        metrics = compute_metrics(y_test, test_proba, best_thr)

        elapsed = time.time() - t0
        round_times.append(elapsed)
        steps_per_client = int(np.ceil(np.mean([len(site[0]) for site in sites]) / batch_size)) * local_epochs
        sampling_rate = min(1.0, batch_size / max(1, int(np.mean([len(site[0]) for site in sites]))))
        approx_epsilon = (
            estimate_privacy_budget_upper_bound(
                num_steps=max(1, total_dp_steps),
                sampling_rate=sampling_rate,
                noise_multiplier=dp_noise_multiplier,
                delta=dp_delta,
            )
            if use_dp
            else np.inf
        )

        metrics.update(
            {
                "round": rnd,
                "val_best_f1": float(best_val_f1),
                "elapsed_sec": float(elapsed),
                "sites": int(n_sites),
                "local_epochs": int(local_epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(lr),
                "partition_type": partition_type,
                "smote": bool(use_smote),
                "dataset": dataset,
                "aggregation": aggregation,
                "feature_selection_top_k": int(feature_selection_top_k),
                "model_family": model_family,
                "dp_enabled": bool(use_dp),
                "dp_clip_norm": float(dp_clip_norm),
                "dp_noise_multiplier": float(dp_noise_multiplier),
                "dp_delta": float(dp_delta),
                "dp_steps_total": int(total_dp_steps),
                "approx_privacy_epsilon_upper": float(approx_epsilon) if np.isfinite(approx_epsilon) else None,
                "attack_type": attack_type,
                "attack_fraction": float(attack_fraction),
                "attack_strength": float(attack_strength),
                "malicious_clients": int(malicious_clients),
                "communication_mb_round": float((model_bytes * len(sites)) / (1024 * 1024)),
                "model_hash": hash_model_weights(get_weights(global_model)),
                "estimated_steps_per_client": int(steps_per_client),
            }
        )
        history.append(metrics)

        print(
            f"Round {rnd:02d} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"prec={metrics['precision']:.4f} | "
            f"rec={metrics['recall']:.4f} | "
            f"spec={metrics['specificity']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auc={metrics['auc']:.4f}"
        )

    history_df = pd.DataFrame(history)
    attack_tag = attack_type if attack_type != "none" else "clean"
    dp_tag = "dp" if use_dp else "nodp"
    run_tag = (
        f"{dataset}_sites{n_sites}_{partition_type}_{aggregation}_{attack_tag}_{dp_tag}_"
        f"{'smote' if use_smote else 'nosmote'}"
    )
    csv_path = EVAL_DIR / f"fl_metrics_{run_tag}.csv"
    json_path = EVAL_DIR / f"fl_metrics_{run_tag}.json"
    best_path = EVAL_DIR / f"fl_best_summary_{run_tag}.json"

    history_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_row = history_df.iloc[history_df["f1"].idxmax()].to_dict()
    best_row["avg_round_time_sec"] = float(np.mean(round_times))
    best_row["feature_count"] = int(len(feature_names))
    best_row["features"] = feature_names

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print("\n=== Saved Outputs ===")
    print(csv_path)
    print(json_path)
    print(best_path)
    print("=====================\n")
    return history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pima", choices=["pima", "breast_cancer", "synthetic_ct", "synthetic_large", "eicu_demo"])
    parser.add_argument("--sites", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--non_iid", action="store_true")
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "trimmed_mean"])
    parser.add_argument("--dp_clip_norm", type=float, default=1.0)
    parser.add_argument("--dp_noise_multiplier", type=float, default=0.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument(
        "--attack_type",
        type=str,
        default="none",
        choices=["none", "label_flip", "gaussian", "sign_flip", "gradient_poisoning", "byzantine"],
    )
    parser.add_argument("--attack_fraction", type=float, default=0.0)
    parser.add_argument("--attack_strength", type=float, default=0.5)
    parser.add_argument("--synthetic_sites", type=int, default=8)
    parser.add_argument("--synthetic_participants_per_site", type=int, default=120)
    parser.add_argument("--synthetic_visits", type=int, default=12)
    parser.add_argument("--synthetic_n_samples", type=int, default=12000)
    parser.add_argument("--eicu_site_groups", type=int, default=8)
    parser.add_argument("--feature_selection_top_k", type=int, default=0)
    parser.add_argument("--model_family", type=str, default="default", choices=["default", "tabular_small"])
    args = parser.parse_args()

    run_experiment(
        n_sites=args.sites,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        non_iid=args.non_iid,
        use_smote=args.smote,
        seed=args.seed,
        dataset=args.dataset,
        aggregation=args.aggregation,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_delta=args.dp_delta,
        attack_type=args.attack_type,
        attack_fraction=args.attack_fraction,
        attack_strength=args.attack_strength,
        synthetic_sites=args.synthetic_sites,
        synthetic_participants_per_site=args.synthetic_participants_per_site,
        synthetic_visits=args.synthetic_visits,
        synthetic_n_samples=args.synthetic_n_samples,
        eicu_site_groups=args.eicu_site_groups,
        feature_selection_top_k=args.feature_selection_top_k,
        model_family=args.model_family,
    )
