"""
P3 — BETTER-BP Nonattendance Prediction and Two-Client Federated Learning
=========================================================================
Primary task  : predict 6-month visit nonattendance (Y6, n=58/402, 14.4%)
Secondary task: predict 12-month visit nonattendance (Y12, n=89/402, 22.1%)
Sensitivity   : confirmed discontinuation only (n=65/402; 25 unknown excluded)

Two modelling datasets
  prognostic : pre-treatment covariates only (no arm assignment)
  uplift     : prognostic features + arm_bin + arm×feature interactions
               (used in P4; constructed here but NOT used in the FL comparison)

Feature set (~30 variables)
  Demographics    : age, age_top_coded, sex_female, hispanic, race_black,
                    race_white, race_other
  Socioeconomic   : insurance_{medicaid,medicare,private,uninsured},
                    edu_less_hs, marital_partnered
  Lifestyle       : tobacco, alcohol, illicit
  Clinical        : bmi, sbp_baseline, dbp_baseline, obese, afib, gi_bleeding
  Patient-reported: mases, tsrq_autonomous, tsrq_controlled, tsrq_amotivation,
                    charlson, phq8, sf12_mcs, sf12_pcs

Model configurations (8 total)
  1  LR-A        Site A local logistic regression (prognostic)
  2  LR-B        Site B local logistic regression (prognostic)
  3  LR-C        Centralized logistic regression (A+B pooled)
  4  EN-C        Centralized elastic-net logistic regression
  5  GBM-C       Centralized LightGBM
  6  MLP-C       Small centralized MLP (64-32, relu)
  7  FL-FedAvg   Two-client FedAvg (weighted by client size)
  8  FL-FedAvg-E Two-client equal-weight FedAvg (ablation)
  9  FL-FedProx  Two-client FedProx (mu=0.1)

Validation
  Repeated participant-level stratified 5-fold CV (stratify on site×outcome)
  Five fixed seeds: [7, 11, 19, 23, 37]  → 25 train/test splits per model
  Primary metric  : PR-AUC (precision-recall area under curve)
  Secondary       : AUROC, MCC, balanced accuracy, sensitivity, specificity,
                    Brier score, ECE (expected calibration error, 10 bins)

Cross-site transfer (single fixed split)
  Train A, test B  (generalisation to smaller site)
  Train B, test A  (small-site sensitivity)

Outputs: processed/better_bp/p3_fl_prediction/
  features_prognostic.parquet   — prognostic feature matrix (one row/participant)
  features_uplift.parquet       — uplift feature matrix
  results_cv.csv                — per-seed per-fold per-model PR-AUC + all metrics
  results_summary.csv           — mean ± 95%-CI across 25 folds
  results_transfer.csv          — cross-site transfer results
  p3_report.txt                 — plain-text report
  figure_pr_curves.png          — PR curves (pooled test sets)
  figure_auroc.png              — AUROC comparison bar chart
"""

import json
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, brier_score_loss,
    precision_recall_curve
)
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RAW_DIR  = pathlib.Path("New/BETTER-BP")
OUT_DIR  = pathlib.Path("processed/better_bp/p3_fl_prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 11, 19, 23, 37]

# ──────────────────────────────────────────────────────────────────────────────
# §1  FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_site(site_id: str) -> pd.DataFrame:
    fname = f"BETTERBP_De-identification-DATA_LABELS_Site{site_id}.xlsx"
    df = pd.read_excel(RAW_DIR / fname, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df["Repeat Instrument"] = df.get("Repeat Instrument", pd.Series(dtype=str)).fillna("").str.strip()
    df["Event Name"] = df["Event Name"].fillna("").str.strip()
    df["Record ID"] = df["Record ID"].str.strip()
    df["participant_id"] = site_id + "_" + df["Record ID"]
    return df

def yn_to_int(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("yes", "checked", "1", "true"):
        return 1.0
    if s in ("no", "unchecked", "0", "false"):
        return 0.0
    return np.nan

def safe_float(val) -> float:
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return np.nan

def first_val(series: pd.Series):
    vals = series.dropna()
    return vals.iloc[0] if len(vals) > 0 else None

def extract_baseline_features(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    """Extract the canonical ~30-variable prognostic feature set from baseline rows."""
    main = df[df["Repeat Instrument"] == ""]
    baseline = main[main["Event Name"].str.lower().str.startswith("baseline")].copy()

    # Column aliases (after strip) that differ between sites
    # All names are normalised after the load_raw_site() column-strip
    OBESITY_COL = next(
        (c for c in baseline.columns if "obese" in c.lower()), None
    )
    TSRQ_AUTO_COL = next(
        (c for c in baseline.columns if "autonomous motivation" in c.lower()), None
    )

    records = []
    for pid, grp in baseline.groupby("participant_id"):
        def g(col):
            return first_val(grp[col]) if col and col in grp.columns else None

        # Age with top-coding
        age_raw = g("Age")
        if age_raw is not None and "90 or older" in str(age_raw).lower():
            age_val      = 90.0
            age_top      = 1.0
        else:
            age_val      = safe_float(age_raw)
            age_top      = 0.0

        # Sex
        sex_str    = str(g("Sex") or "").strip().lower()
        sex_female = 1.0 if sex_str == "female" else (0.0 if sex_str == "male" else np.nan)

        # Ethnicity
        eth_str  = str(g("Ethnicity") or "").strip().lower()
        hispanic = 1.0 if "hispanic" in eth_str else (0.0 if "not hispanic" in eth_str else np.nan)

        # Race (choice= columns)
        def race_col(label):
            col = next((c for c in grp.columns if f"race  (choice={label.lower()}" in c.lower()), None)
            return yn_to_int(g(col)) if col else np.nan

        race_black = race_col("Black or African American")
        race_white = race_col("White")
        race_other = race_col("Other Specify")

        # Insurance (one-hot: Medicaid, Medicare, Private, Uninsured)
        ins_str = str(g("Insurance Status") or "").strip().lower()
        ins_medicaid  = 1.0 if "medicaid"  in ins_str else (0.0 if ins_str else np.nan)
        ins_medicare  = 1.0 if "medicare"  in ins_str else (0.0 if ins_str else np.nan)
        ins_private   = 1.0 if "private"   in ins_str else (0.0 if ins_str else np.nan)
        ins_uninsured = 1.0 if "uninsured" in ins_str else (0.0 if ins_str else np.nan)

        # Education (less than college)
        edu_str   = str(g("What is the highest level of education you have completed?") or "").strip().lower()
        edu_less_hs = 1.0 if any(x in edu_str for x in ["high school", "less than", "ged", "some high"]) \
                      else (0.0 if edu_str else np.nan)

        # Marital status (partnered = married or living with partner)
        mar_str  = str(g("Which best describes your current marital status?") or "").strip().lower()
        marital_partnered = 1.0 if any(x in mar_str for x in ["married", "partner", "domestic"]) \
                            else (0.0 if mar_str else np.nan)

        # Lifestyle
        tob_col  = next((c for c in grp.columns if c.strip().lower().startswith("tobacco use")), None)
        alc_col  = next((c for c in grp.columns if c.strip().lower().startswith("alcohol use")), None)
        ill_col  = next((c for c in grp.columns if "illicit use" in c.strip().lower()), None)
        tobacco  = yn_to_int(g(tob_col) if tob_col else None)
        alcohol  = yn_to_int(g(alc_col) if alc_col else None)
        illicit  = yn_to_int(g(ill_col) if ill_col else None)

        # Clinical
        bmi        = safe_float(g("BMI"))
        sbp        = safe_float(g("Systolic Blood Pressure Avg"))
        dbp_col    = next((c for c in grp.columns if "diastolic blood pressure" in c.lower() and "avg" in c.lower()), None)
        dbp        = safe_float(g(dbp_col) if dbp_col else None)
        obese      = yn_to_int(g(OBESITY_COL) if OBESITY_COL else None)
        afib       = yn_to_int(g("Has the patient had atrial fibrillation?"))
        gi_bleed_col = next((c for c in grp.columns if "gi bleeding" in c.lower()), None)
        gi_bleeding  = yn_to_int(g(gi_bleed_col) if gi_bleed_col else None)

        # Patient-reported outcomes
        mases          = safe_float(g("Calculated MASES Total score"))
        tsrq_auto_col  = TSRQ_AUTO_COL
        tsrq_autonomous = safe_float(g(tsrq_auto_col) if tsrq_auto_col else None)
        tsrq_controlled = safe_float(g("Controlled Motivation Score:"))
        tsrq_amotiv     = safe_float(g("Amotivation Score:"))
        charlson_col    = next((c for c in grp.columns if c == "Enter Total Score"), None)
        charlson        = safe_float(g(charlson_col) if charlson_col else None)
        phq8_col        = next((c for c in grp.columns if c == "Enter Total Score.2"), None)
        phq8            = safe_float(g(phq8_col) if phq8_col else None)
        sf12_mcs        = safe_float(g("SF-12 MCS"))
        sf12_pcs        = safe_float(g("SF-12 PCS"))

        records.append({
            "participant_id": pid,
            "site": site_id,
            # Demographics
            "age": age_val, "age_top_coded": age_top,
            "sex_female": sex_female, "hispanic": hispanic,
            "race_black": race_black, "race_white": race_white, "race_other": race_other,
            # Socioeconomic
            "ins_medicaid": ins_medicaid, "ins_medicare": ins_medicare,
            "ins_private": ins_private, "ins_uninsured": ins_uninsured,
            "edu_less_hs": edu_less_hs, "marital_partnered": marital_partnered,
            # Lifestyle
            "tobacco": tobacco, "alcohol": alcohol, "illicit": illicit,
            # Clinical
            "bmi": bmi, "sbp_baseline": sbp, "dbp_baseline": dbp,
            "obese": obese, "afib": afib, "gi_bleeding": gi_bleeding,
            # Patient-reported
            "mases": mases, "tsrq_autonomous": tsrq_autonomous,
            "tsrq_controlled": tsrq_controlled, "tsrq_amotivation": tsrq_amotiv,
            "charlson": charlson, "phq8": phq8,
            "sf12_mcs": sf12_mcs, "sf12_pcs": sf12_pcs,
        })

    return pd.DataFrame(records)


print("Extracting baseline features from raw Site A and B data...")
raw_a = load_raw_site("A")
raw_b = load_raw_site("B")
feat_a = extract_baseline_features(raw_a, "A")
feat_b = extract_baseline_features(raw_b, "B")
feat_all = pd.concat([feat_a, feat_b], ignore_index=True)

print(f"  Site A: {len(feat_a)} baseline rows")
print(f"  Site B: {len(feat_b)} baseline rows")
print(f"  Combined: {len(feat_all)} rows")

# ── Load outcomes and join ─────────────────────────────────────────────────────
outcomes = pd.read_parquet("processed/better_bp/trial_outcomes.parquet")
df = outcomes.merge(feat_all, on=["participant_id","site"], how="inner")
df = df[df["treatment_arm"].notna()].copy()   # randomized only
print(f"  After inner join with outcomes (randomized): {len(df)} rows")

# Canonical feature list (prognostic = no arm)
FEATURE_COLS = [
    "age", "age_top_coded", "sex_female", "hispanic",
    "race_black", "race_white", "race_other",
    "ins_medicaid", "ins_medicare", "ins_private", "ins_uninsured",
    "edu_less_hs", "marital_partnered",
    "tobacco", "alcohol", "illicit",
    "bmi", "sbp_baseline", "dbp_baseline",
    "obese", "afib", "gi_bleeding",
    "mases", "tsrq_autonomous", "tsrq_controlled", "tsrq_amotivation",
    "charlson", "phq8", "sf12_mcs", "sf12_pcs",
]

# Filter to features with at least 50% non-missing across both sites
missing_rate = df[FEATURE_COLS].isna().mean()
FEATURE_COLS = [c for c in FEATURE_COLS if missing_rate[c] < 0.5]
print(f"  Features retained (>50% complete): {len(FEATURE_COLS)}")
print(f"  Dropped (>50% missing): {[c for c in FEATURE_COLS if c not in FEATURE_COLS]}")

# Save feature matrices
df_prog = df[["participant_id", "site", "treatment_arm",
              "nonattendance_6m", "nonattendance_12m",
              "explicit_noncompletion", "missing_eos_record"] + FEATURE_COLS].copy()
df_prog.to_parquet(OUT_DIR / "features_prognostic.parquet", index=False)

# Uplift feature matrix (add arm + interactions with key features)
df_upl = df_prog.copy()
df_upl["arm_bin"] = (df_upl["treatment_arm"] == "Intervention").astype(float)
for c in ["age", "sbp_baseline", "mases", "charlson"]:
    if c in FEATURE_COLS:
        df_upl[f"arm_x_{c}"] = df_upl["arm_bin"] * df_upl[c].fillna(df_upl[c].median())
df_upl.to_parquet(OUT_DIR / "features_uplift.parquet", index=False)

print(f"  Feature matrices saved.")

# ──────────────────────────────────────────────────────────────────────────────
# §2  PREPROCESSING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

def preprocess(X_train, X_test):
    pp = get_preprocessor()
    X_tr = pp.fit_transform(X_train)
    X_te = pp.transform(X_test)
    return X_tr, X_te, pp


# ──────────────────────────────────────────────────────────────────────────────
# §3  FEDERATED LEARNING IMPLEMENTATION
# ──────────────────────────────────────────────────────────────────────────────

class FedLogisticRegression:
    """Federated logistic regression via mini-batch gradient descent.

    mu=0 → FedAvg; mu>0 → FedProx (proximal penalty on divergence from global).
    equal_weight=True → client weight = 1/K regardless of dataset size (ablation).
    """

    def __init__(self, mu=0.0, n_rounds=20, local_epochs=5,
                 lr=0.05, equal_weight=False, random_state=0):
        self.mu           = mu
        self.n_rounds     = n_rounds
        self.local_epochs = local_epochs
        self.lr           = lr
        self.equal_weight = equal_weight
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _local_step(self, X, y, w, b, gw, gb):
        n = len(y)
        for _ in range(self.local_epochs):
            p       = self._sigmoid(X @ w + b)
            grad_w  = X.T @ (p - y) / n + self.mu * (w - gw)
            grad_b  = np.mean(p - y)     + self.mu * (b - gb)
            w       = w - self.lr * grad_w
            b       = b - self.lr * grad_b
        return w, b

    def fit(self, X_list, y_list):
        n_feat = X_list[0].shape[1]
        rng    = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0, 0.01, n_feat)
        self.b_ = 0.0

        n_list = [len(y) for y in y_list]
        if self.equal_weight:
            weights = np.ones(len(n_list)) / len(n_list)
        else:
            total   = sum(n_list)
            weights = np.array([n / total for n in n_list])

        for _ in range(self.n_rounds):
            local_ws, local_bs = [], []
            for k, (X, y) in enumerate(zip(X_list, y_list)):
                w_k, b_k = self._local_step(
                    X.astype(np.float64), y.astype(np.float64),
                    self.w_.copy(), self.b_, self.w_, self.b_
                )
                local_ws.append(w_k)
                local_bs.append(b_k)

            self.w_ = sum(weights[k] * local_ws[k] for k in range(len(local_ws)))
            self.b_ = float(sum(weights[k] * local_bs[k] for k in range(len(local_bs))))

        return self

    def predict_proba(self, X):
        p = self._sigmoid(X.astype(np.float64) @ self.w_ + self.b_)
        return np.column_stack([1 - p, p])


# ──────────────────────────────────────────────────────────────────────────────
# §4  METRICS
# ──────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    n     = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "pr_auc":    average_precision_score(y_true, y_prob),
        "auroc":     roc_auc_score(y_true, y_prob),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "bal_acc":   balanced_accuracy_score(y_true, y_pred),
        "sens":      sens,
        "spec":      spec,
        "brier":     brier_score_loss(y_true, y_prob),
        "ece":       expected_calibration_error(y_true, y_prob),
    }


# ──────────────────────────────────────────────────────────────────────────────
# §5  MODEL FACTORY
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAMES = [
    "LR-A", "LR-B", "LR-C", "EN-C", "GBM-C", "MLP-C",
    "FL-FedAvg", "FL-FedAvg-E", "FL-FedProx"
]

def build_and_predict(model_name, X_a, y_a, X_b, y_b, X_test, seed):
    """Train model and return predicted probabilities for class=1 on X_test."""

    if model_name == "LR-A":
        m = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        m.fit(X_a, y_a)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "LR-B":
        if len(y_b) == 0 or y_b.sum() == 0:
            return np.full(len(X_test), y_b.mean() if len(y_b) > 0 else 0.14)
        m = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        m.fit(X_b, y_b)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "LR-C":
        X_c = np.vstack([X_a, X_b])
        y_c = np.concatenate([y_a, y_b])
        m = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        m.fit(X_c, y_c)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "EN-C":
        X_c = np.vstack([X_a, X_b])
        y_c = np.concatenate([y_a, y_b])
        m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                               max_iter=2000, solver="saga", random_state=seed)
        m.fit(X_c, y_c)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "GBM-C":
        X_c = np.vstack([X_a, X_b])
        y_c = np.concatenate([y_a, y_b])
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                               num_leaves=15, min_child_samples=5,
                               is_unbalance=True, random_state=seed,
                               verbose=-1)
        m.fit(X_c, y_c)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "MLP-C":
        X_c = np.vstack([X_a, X_b])
        y_c = np.concatenate([y_a, y_b])
        m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          max_iter=300, random_state=seed, early_stopping=True,
                          validation_fraction=0.1)
        m.fit(X_c, y_c)
        return m.predict_proba(X_test)[:, 1]

    if model_name == "FL-FedAvg":
        m = FedLogisticRegression(mu=0.0, equal_weight=False, random_state=seed)
        m.fit([X_a, X_b], [y_a, y_b])
        return m.predict_proba(X_test)[:, 1]

    if model_name == "FL-FedAvg-E":
        m = FedLogisticRegression(mu=0.0, equal_weight=True, random_state=seed)
        m.fit([X_a, X_b], [y_a, y_b])
        return m.predict_proba(X_test)[:, 1]

    if model_name == "FL-FedProx":
        m = FedLogisticRegression(mu=0.1, equal_weight=False, random_state=seed)
        m.fit([X_a, X_b], [y_a, y_b])
        return m.predict_proba(X_test)[:, 1]

    raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# §6  MAIN CV LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_cv(df, outcome_col, feature_cols, label="Y6"):
    print(f"\nRunning CV for outcome: {outcome_col} ({label})")
    df_sub = df[df[outcome_col].notna()].copy()
    X_all  = df_sub[feature_cols].values
    y_all  = df_sub[outcome_col].values.astype(int)
    sites  = df_sub["site"].values
    print(f"  N={len(y_all)}, positives={y_all.sum()} ({y_all.mean()*100:.1f}%)")

    # Stratification label: site + outcome
    strat_label = np.array([f"{s}_{y}" for s, y in zip(sites, y_all)])

    cv_records = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_all, strat_label)):
            X_tr_raw, X_te_raw = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te         = y_all[tr_idx], y_all[te_idx]
            s_tr, s_te         = sites[tr_idx], sites[te_idx]

            X_tr, X_te, _ = preprocess(X_tr_raw, X_te_raw)

            # Split training data by site
            mask_a_tr = s_tr == "A"
            mask_b_tr = s_tr == "B"
            X_a_tr = X_tr[mask_a_tr]
            y_a_tr = y_tr[mask_a_tr]
            X_b_tr = X_tr[mask_b_tr]
            y_b_tr = y_tr[mask_b_tr]

            if y_a_tr.sum() == 0 or y_b_tr.sum() == 0:
                continue  # skip folds with no positives in one site

            for mname in MODEL_NAMES:
                try:
                    y_prob = build_and_predict(
                        mname, X_a_tr, y_a_tr, X_b_tr, y_b_tr, X_te, seed
                    )
                    if y_te.sum() == 0:
                        continue
                    # Pooled test metrics
                    m = compute_metrics(y_te, y_prob)
                    m.update({
                        "model": mname, "outcome": label, "site_test": "all",
                        "seed": seed, "fold": fold_idx,
                        "n_test": len(y_te), "n_pos_test": int(y_te.sum()),
                    })
                    cv_records.append(m)
                    # Per-site test metrics (A-test portion, B-test portion)
                    for st in ["A", "B"]:
                        mask_st = s_te == st
                        if mask_st.sum() == 0 or y_te[mask_st].sum() == 0:
                            continue
                        m_st = compute_metrics(y_te[mask_st], y_prob[mask_st])
                        m_st.update({
                            "model": mname, "outcome": label, "site_test": st,
                            "seed": seed, "fold": fold_idx,
                            "n_test": int(mask_st.sum()),
                            "n_pos_test": int(y_te[mask_st].sum()),
                        })
                        cv_records.append(m_st)
                except Exception as e:
                    print(f"    [{mname} fold={fold_idx} seed={seed}] error: {e}")

    return pd.DataFrame(cv_records)


def build_single_site(model_name, X_train, y_train, X_test, seed):
    """Train on a SINGLE site only, test on another — no cross-site leakage.

    All models here train only on the provided X_train / y_train.
    For FL models, one client degenerates to local gradient descent (valid baseline).
    """
    if model_name in ("LR-local", "LR-A", "LR-B", "LR-C"):
        m = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    if model_name == "EN-local":
        m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                               max_iter=2000, solver="saga", random_state=seed)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    if model_name == "GBM-local":
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05,
                               num_leaves=15, min_child_samples=5,
                               is_unbalance=True, random_state=seed, verbose=-1)
        m.fit(X_train, y_train)
        return m.predict_proba(X_test)[:, 1]
    if model_name in ("FL-local", "FL-FedAvg", "FL-FedProx"):
        # 1-client FL is equivalent to local logistic gradient descent
        m = FedLogisticRegression(mu=0.0, equal_weight=False, random_state=seed)
        m.fit([X_train], [y_train])
        return m.predict_proba(X_test)[:, 1]
    raise ValueError(f"Unknown model in single-site transfer: {model_name}")


def run_cross_site_transfer(df, outcome_col, feature_cols, label="Y6"):
    """Pure cross-site transfer: ALL models train on source site only.

    This avoids data leakage — centralized and FL models also train
    only on the source site's data. This is the correct design for
    measuring how well a model trained at one site generalises to another.
    Reported as 'small-site sensitivity' when B→A.
    """
    print(f"\nCross-site transfer for: {outcome_col} ({label})")
    df_sub = df[df[outcome_col].notna()].copy()
    X_all  = df_sub[feature_cols].values
    y_all  = df_sub[outcome_col].values.astype(int)
    sites  = df_sub["site"].values

    mask_a = sites == "A"
    mask_b = sites == "B"
    X_a, y_a = X_all[mask_a], y_all[mask_a]
    X_b, y_b = X_all[mask_b], y_all[mask_b]

    TRANSFER_MODELS = ["LR-local", "EN-local", "GBM-local", "FL-local"]

    records = []
    for seed in SEEDS:
        # A → B
        X_a_pp, X_b_pp, _ = preprocess(X_a, X_b)
        for mname in TRANSFER_MODELS:
            try:
                y_prob = build_single_site(mname, X_a_pp, y_a, X_b_pp, seed)
                if y_b.sum() == 0:
                    continue
                m = compute_metrics(y_b, y_prob)
                m.update({"model": mname, "outcome": label, "seed": seed,
                           "transfer": "A_train_B_test",
                           "n_train": len(y_a), "n_test": len(y_b),
                           "n_pos_test": int(y_b.sum())})
                records.append(m)
            except Exception as e:
                print(f"    [A->B {mname} seed={seed}] error: {e}")

        # B → A (small-site sensitivity)
        X_b_pp2, X_a_pp2, _ = preprocess(X_b, X_a)
        for mname in TRANSFER_MODELS:
            try:
                y_prob = build_single_site(mname, X_b_pp2, y_b, X_a_pp2, seed)
                if y_a.sum() == 0:
                    continue
                m = compute_metrics(y_a, y_prob)
                m.update({"model": mname, "outcome": label, "seed": seed,
                           "transfer": "B_train_A_test",
                           "n_train": len(y_b), "n_test": len(y_a),
                           "n_pos_test": int(y_a.sum())})
                records.append(m)
            except Exception as e:
                print(f"    [B->A {mname} seed={seed}] error: {e}")

    return pd.DataFrame(records)


# ── Sensitivity: confirmed discontinuation (exclude missing EOS) ───────────────
def add_disc_sensitivity(df):
    df = df.copy()
    df["disc_sensitivity"] = df["explicit_noncompletion"].astype(float)
    df.loc[df["missing_eos_record"] == 1, "disc_sensitivity"] = np.nan
    return df

df = add_disc_sensitivity(df)

# ── Run all CVs ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running cross-validated evaluation (25 splits per model)")
print("="*60)

cv6  = run_cv(df, "nonattendance_6m",  FEATURE_COLS, label="Y6-nonatt")
cv12 = run_cv(df, "nonattendance_12m", FEATURE_COLS, label="Y12-nonatt")
cvd  = run_cv(df, "disc_sensitivity",  FEATURE_COLS, label="Ydisc-sens")
cv_all = pd.concat([cv6, cv12, cvd], ignore_index=True)
cv_all.to_csv(OUT_DIR / "results_cv.csv", index=False)

# ── Cross-site transfer ───────────────────────────────────────────────────────
print("\nCross-site transfer evaluation")
tr6  = run_cross_site_transfer(df, "nonattendance_6m",  FEATURE_COLS, "Y6-nonatt")
tr12 = run_cross_site_transfer(df, "nonattendance_12m", FEATURE_COLS, "Y12-nonatt")
tr_all = pd.concat([tr6, tr12], ignore_index=True)
tr_all.to_csv(OUT_DIR / "results_transfer.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# §7  SUMMARY TABLES
# ──────────────────────────────────────────────────────────────────────────────

METRIC_COLS = ["pr_auc","auroc","mcc","bal_acc","sens","spec","brier","ece"]

def summarise(cv_df, label_filter=None, site_filter="all"):
    sub = cv_df if label_filter is None else cv_df[cv_df["outcome"]==label_filter]
    if "site_test" in sub.columns:
        sub = sub[sub["site_test"] == site_filter]
    rows = []
    for mname in MODEL_NAMES:
        m_df = sub[sub["model"]==mname]
        if m_df.empty:
            continue
        row = {"model": mname, "site_test": site_filter}
        for mc in METRIC_COLS:
            if mc not in m_df.columns:
                continue
            vals = m_df[mc].dropna().values
            if len(vals) == 0:
                row[mc] = "—"
                row[mc+"_ci"] = "—"
                continue
            mn  = vals.mean()
            sem = vals.std(ddof=1) / len(vals)**0.5
            ci  = 1.96 * sem
            row[mc]       = round(mn, 4)
            row[mc+"_ci"] = round(ci, 4)
        rows.append(row)
    return pd.DataFrame(rows)

summ6  = summarise(cv_all, "Y6-nonatt",  site_filter="all")
summ12 = summarise(cv_all, "Y12-nonatt", site_filter="all")
summd  = summarise(cv_all, "Ydisc-sens", site_filter="all")
# Per-site CV summaries
summ6A  = summarise(cv_all, "Y6-nonatt",  site_filter="A")
summ6B  = summarise(cv_all, "Y6-nonatt",  site_filter="B")
summ12A = summarise(cv_all, "Y12-nonatt", site_filter="A")
summ12B = summarise(cv_all, "Y12-nonatt", site_filter="B")

summ_all = pd.concat(
    [summ6.assign(outcome="Y6-nonatt"),
     summ12.assign(outcome="Y12-nonatt"),
     summd.assign(outcome="Ydisc-sens"),
     summ6A.assign(outcome="Y6-nonatt-siteA"),
     summ6B.assign(outcome="Y6-nonatt-siteB"),
     summ12A.assign(outcome="Y12-nonatt-siteA"),
     summ12B.assign(outcome="Y12-nonatt-siteB")],
    ignore_index=True
)
summ_all.to_csv(OUT_DIR / "results_summary.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# §8  FIGURES
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_bar(summ_df, metric, title, out_path):
    models = summ_df["model"].tolist()
    means  = pd.to_numeric(summ_df[metric], errors="coerce").values
    cis    = pd.to_numeric(summ_df[metric+"_ci"], errors="coerce").values
    colors = ["#2196F3"]*2 + ["#4CAF50"]*4 + ["#FF9800"]*3

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(models[::-1], means[::-1], xerr=cis[::-1],
                   color=colors[::-1], alpha=0.8, capsize=3, error_kw={"elinewidth":1})
    ax.set_xlabel(metric.upper().replace("_"," "))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, lw=0.7, ls="--", color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

plot_metric_bar(summ6,  "pr_auc",  "PR-AUC — 6m Nonattendance (25-fold)",
                OUT_DIR / "figure_prauc_6m.png")
plot_metric_bar(summ6,  "auroc",   "AUROC  — 6m Nonattendance (25-fold)",
                OUT_DIR / "figure_auroc_6m.png")
plot_metric_bar(summ12, "pr_auc",  "PR-AUC — 12m Nonattendance (25-fold)",
                OUT_DIR / "figure_prauc_12m.png")


# ──────────────────────────────────────────────────────────────────────────────
# §9  PLAIN-TEXT REPORT
# ──────────────────────────────────────────────────────────────────────────────

report = []
def R(s=""): report.append(s)
def S(t):
    R(); R("="*70); R(f"  {t}"); R("="*70)

S("P3 REPORT — BETTER-BP Nonattendance Prediction and FL")
R(f"Dataset: {len(df)} randomized participants (Site A={df['site'].eq('A').sum()}, B={df['site'].eq('B').sum()})")
R(f"Features retained: {len(FEATURE_COLS)}")
R()

S("Feature completeness (randomized participants)")
miss = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
for feat, rate in miss.items():
    R(f"  {feat:<30s}: {(1-rate)*100:5.1f}% complete ({int((1-rate)*len(df))} non-missing)")

S("Primary outcome: 6-month nonattendance")
R()
R(f"{'Model':<16} {'PR-AUC':>8} {'±95%CI':>8} {'AUROC':>8} {'MCC':>7} {'BalAcc':>8} {'Brier':>7}")
R("-"*65)
for _, row in summ6.iterrows():
    pr  = row.get("pr_auc",  "—")
    ci  = row.get("pr_auc_ci","—")
    au  = row.get("auroc",   "—")
    mc  = row.get("mcc",     "—")
    ba  = row.get("bal_acc", "—")
    br  = row.get("brier",   "—")
    R(f"  {row['model']:<14} {str(pr):>8} ±{str(ci):<6} {str(au):>8} {str(mc):>7} {str(ba):>8} {str(br):>7}")

S("Secondary outcome: 12-month nonattendance")
R()
R(f"{'Model':<16} {'PR-AUC':>8} {'±95%CI':>8} {'AUROC':>8} {'Brier':>7}")
R("-"*50)
for _, row in summ12.iterrows():
    pr = row.get("pr_auc","—"); ci = row.get("pr_auc_ci","—")
    au = row.get("auroc","—");  br = row.get("brier","—")
    R(f"  {row['model']:<14} {str(pr):>8} ±{str(ci):<6} {str(au):>8} {str(br):>7}")

S("Cross-site transfer — PR-AUC (averaged across 5 seeds)")
R()
for transfer_dir in ["A_train_B_test", "B_train_A_test"]:
    R(f"  {transfer_dir}:")
    sub = tr_all[tr_all["outcome"]=="Y6-nonatt"][tr_all["transfer"]==transfer_dir]
    for mname in sub["model"].unique():
        vals = sub[sub["model"]==mname]["pr_auc"].dropna().values
        if len(vals) == 0: continue
        R(f"    {mname:<16}: PR-AUC = {vals.mean():.4f} (SD={vals.std():.4f})")
    R()

S("Interpretation notes")
R("  - Primary metric is PR-AUC because 6m nonattendance is imbalanced (14.4%).")
R("  - LR-B (site B alone, n=90) is expected to be noisy; reported for transparency.")
R("  - FL models use no Byzantine-robustness mechanisms (only 2 clients; not applicable).")
R("  - Median, trimmed mean, and Krum are NOT reported here; see eICU P6 for those.")
R("  - Uplift estimation (P4) will be built from features_uplift.parquet.")
R("  - DQN policy is unsuitable here; BETTER-BP has no sequential action data.")

with open(OUT_DIR / "p3_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# ──────────────────────────────────────────────────────────────────────────────
# Final console output
# ──────────────────────────────────────────────────────────────────────────────
print()
print("\n".join(report[:60]).encode("ascii", errors="replace").decode("ascii"))
print()
print(f"Outputs written to: {OUT_DIR.resolve()}")
print("  features_prognostic.parquet, features_uplift.parquet")
print("  results_cv.csv, results_summary.csv, results_transfer.csv")
print("  figure_prauc_6m.png, figure_auroc_6m.png, figure_prauc_12m.png")
print("  p3_report.txt")
