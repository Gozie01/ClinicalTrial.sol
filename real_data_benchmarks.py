import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

from eicu_dataset import prepare_eicu_demo_dataset


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "paper_bundle"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def select_threshold_by_f1(y_true, scores):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_predictions(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "threshold": float(threshold),
    }


def build_mlp(input_dim: int):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.15),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=optimizers.Adam(5e-4), loss="binary_crossentropy")
    return model


def run_benchmarks(seed: int = 42, feature_top_k: int = 40):
    df, _, _ = prepare_eicu_demo_dataset(refresh=False)
    X = df.drop(columns=["patientunitstayid", "hospitalid", "site_group", "Outcome"])
    y = df["Outcome"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        stratify=y_train_full,
        random_state=seed,
    )
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    mi_scores = mutual_info_classif(X_train_imp, y_train, random_state=seed)
    selected_cols = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False).head(feature_top_k).index.tolist()
    X_train_sel = X_train_imp[selected_cols]
    X_val_sel = X_val_imp[selected_cols]
    X_test_sel = X_test_imp[selected_cols]

    scaler = StandardScaler().fit(X_train_sel)
    X_train_scaled = scaler.transform(X_train_sel).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_sel).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_sel).astype(np.float32)

    benchmarks = []

    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )
    logreg.fit(X_train_sel, y_train)
    val_scores = logreg.predict_proba(X_val_sel)[:, 1]
    threshold, _ = select_threshold_by_f1(y_val, val_scores)
    scores = logreg.predict_proba(X_test_sel)[:, 1]
    benchmarks.append({"model": "logistic_regression", **evaluate_predictions(y_test, scores, threshold)})

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train_sel, y_train)
    val_scores = rf.predict_proba(X_val_sel)[:, 1]
    threshold, _ = select_threshold_by_f1(y_val, val_scores)
    scores = rf.predict_proba(X_test_sel)[:, 1]
    benchmarks.append({"model": "random_forest", **evaluate_predictions(y_test, scores, threshold)})

    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=seed,
    )
    hgb.fit(X_train_sel, y_train)
    val_scores = hgb.predict_proba(X_val_sel)[:, 1]
    threshold, _ = select_threshold_by_f1(y_val, val_scores)
    scores = hgb.predict_proba(X_test_sel)[:, 1]
    benchmarks.append({"model": "hist_gradient_boosting", **evaluate_predictions(y_test, scores, threshold)})

    tf.random.set_seed(seed)
    mlp = build_mlp(X_train_scaled.shape[1])
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    class_weight = {0: 1.0, 1: float(neg / max(pos, 1))}
    mlp.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=40,
        batch_size=64,
        verbose=0,
        class_weight=class_weight,
    )
    val_scores = mlp.predict(X_val_scaled, verbose=0).ravel()
    threshold, _ = select_threshold_by_f1(y_val, val_scores)
    scores = mlp.predict(X_test_scaled, verbose=0).ravel()
    benchmarks.append({"model": "centralized_mlp_topk", **evaluate_predictions(y_test, scores, threshold)})

    results_df = pd.DataFrame(benchmarks)
    return results_df, selected_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_top_k", type=int, default=40)
    args = parser.parse_args()

    results_df, selected_cols = run_benchmarks(seed=args.seed, feature_top_k=args.feature_top_k)
    csv_path = EVAL_DIR / "real_data_benchmarks.csv"
    meta_path = EVAL_DIR / "real_data_benchmarks_meta.json"
    results_df.to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "feature_top_k": args.feature_top_k, "selected_features": selected_cols}, f, indent=2)
    print(results_df.to_string(index=False))
    print(csv_path)
    print(meta_path)


if __name__ == "__main__":
    main()
