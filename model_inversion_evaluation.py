import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from federated_mlp_run import build_mlp, load_data


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "privacy"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_sample_gradients(model, x_sample, y_sample):
    x_tensor = tf.convert_to_tensor(x_sample[None, :], dtype=tf.float32)
    y_tensor = tf.convert_to_tensor([[float(y_sample)]], dtype=tf.float32)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        y_pred = model(x_tensor, training=False)
        loss = loss_fn(y_tensor, y_pred)
        if model.losses:
            loss += tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    return [tf.identity(grad) for grad in grads]


def clip_and_noise_gradients(grads, clip_norm: float, noise_multiplier: float):
    global_norm = tf.linalg.global_norm(grads)
    coef = tf.minimum(1.0, clip_norm / (global_norm + 1e-6))
    clipped = [grad * coef for grad in grads]
    if noise_multiplier > 0:
        noisy = []
        for grad in clipped:
            noise = tf.random.normal(tf.shape(grad), stddev=noise_multiplier * clip_norm)
            noisy.append(grad + noise)
        clipped = noisy
    return clipped


def flatten_grads(grads):
    return tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)


def reconstruct_input_from_gradients(
    model,
    target_grads,
    y_label: int,
    input_dim: int,
    steps: int = 250,
    lr: float = 0.05,
):
    x_hat = tf.Variable(tf.random.normal([1, input_dim], stddev=0.5), trainable=True)
    y_tensor = tf.convert_to_tensor([[float(y_label)]], dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    target_flat = flatten_grads(target_grads)

    for _ in range(steps):
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                pred = model(x_hat, training=False)
                loss = loss_fn(y_tensor, pred)
                if model.losses:
                    loss += tf.add_n(model.losses)
            dummy_grads = inner_tape.gradient(loss, model.trainable_variables)
            dummy_flat = flatten_grads(dummy_grads)
            grad_loss = tf.reduce_mean(tf.square(dummy_flat - target_flat))
            reg_loss = 1e-3 * tf.reduce_mean(tf.square(x_hat))
            total_loss = grad_loss + reg_loss
        grads_x = outer_tape.gradient(total_loss, [x_hat])
        optimizer.apply_gradients(zip(grads_x, [x_hat]))
    return x_hat.numpy().reshape(-1)


def evaluate_reconstruction(true_x, recon_x):
    mse = float(np.mean((true_x - recon_x) ** 2))
    mae = float(np.mean(np.abs(true_x - recon_x)))
    corr = float(cosine_similarity(true_x.reshape(1, -1), recon_x.reshape(1, -1))[0, 0])
    return {"recon_mse": mse, "recon_mae": mae, "recon_cosine": corr}


def run_model_inversion_benchmark(
    dataset: str,
    seed: int,
    n_targets: int,
    reconstruction_steps: int,
    feature_selection_top_k: int = 0,
    model_family: str = "default",
    lr: float = 1e-3,
):
    set_seed(seed)
    X, y, feature_names, _ = load_data(dataset=dataset, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    if feature_selection_top_k > 0 and feature_selection_top_k < X_train.shape[1]:
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(X_train, y_train, random_state=seed)
        top_indices = np.argsort(mi_scores)[::-1][:feature_selection_top_k]
        X_train = X_train[:, top_indices]
        X_test = X_test[:, top_indices]
        feature_names = [feature_names[idx] for idx in top_indices]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = build_mlp(X_train.shape[1], lr=lr, model_family=model_family)
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

    target_indices = np.linspace(0, len(X_test) - 1, num=min(n_targets, len(X_test)), dtype=int)
    threat_rows = []
    settings = [
        ("no_defense", 10.0, 0.0),
        ("dp_moderate", 1.0, 0.05),
        ("dp_strong", 1.0, 0.15),
    ]

    for target_idx in target_indices:
        x_true = X_test[target_idx]
        y_true = int(y_test[target_idx])
        clean_grads = compute_sample_gradients(model, x_true, y_true)

        for setting_name, clip_norm, noise_multiplier in settings:
            observed_grads = clip_and_noise_gradients(clean_grads, clip_norm=clip_norm, noise_multiplier=noise_multiplier)
            x_recon = reconstruct_input_from_gradients(
                model=model,
                target_grads=observed_grads,
                y_label=y_true,
                input_dim=X_test.shape[1],
                steps=reconstruction_steps,
            )
            metrics = evaluate_reconstruction(x_true, x_recon)
            threat_rows.append(
                {
                    "dataset": dataset,
                    "target_index": int(target_idx),
                    "setting": setting_name,
                    "clip_norm": float(clip_norm),
                    "noise_multiplier": float(noise_multiplier),
                    **metrics,
                }
            )

    results_df = pd.DataFrame(threat_rows)
    summary_df = results_df.groupby("setting")[["recon_mse", "recon_mae", "recon_cosine"]].mean().reset_index()
    return results_df, summary_df, feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="breast_cancer", choices=["pima", "breast_cancer", "synthetic_ct", "synthetic_large", "eicu_demo"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_targets", type=int, default=5)
    parser.add_argument("--reconstruction_steps", type=int, default=200)
    parser.add_argument("--feature_selection_top_k", type=int, default=0)
    parser.add_argument("--model_family", type=str, default="default", choices=["default", "tabular_small"])
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    results_df, summary_df, feature_names = run_model_inversion_benchmark(
        dataset=args.dataset,
        seed=args.seed,
        n_targets=args.n_targets,
        reconstruction_steps=args.reconstruction_steps,
        feature_selection_top_k=args.feature_selection_top_k,
        model_family=args.model_family,
        lr=args.lr,
    )

    tag = f"{args.dataset}_seed{args.seed}"
    results_path = EVAL_DIR / f"model_inversion_{tag}.csv"
    summary_path = EVAL_DIR / f"model_inversion_summary_{tag}.csv"
    meta_path = EVAL_DIR / f"model_inversion_meta_{tag}.json"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed, "features": feature_names}, f, indent=2)

    print("\n=== Model Inversion Summary ===")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(meta_path)


if __name__ == "__main__":
    main()
