from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel, wilcoxon, shapiro


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation"

METRICS = ["accuracy", "f1", "auc", "mcc", "balanced_accuracy"]


def cohens_d_paired(x, y):
    diff = np.array(x, dtype=float) - np.array(y, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(diff.mean() / sd)


def interpret_cohens_d(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def load_and_align(file_a, file_b):
    a = pd.read_csv(file_a)
    b = pd.read_csv(file_b)

    if "seed" not in a.columns:
        raise ValueError(f"'seed' column not found in {file_a}")
    if "seed" not in b.columns:
        raise ValueError(f"'seed' column not found in {file_b}")

    a["seed"] = pd.to_numeric(a["seed"], errors="coerce")
    b["seed"] = pd.to_numeric(b["seed"], errors="coerce")

    a = a.dropna(subset=["seed"]).copy()
    b = b.dropna(subset=["seed"]).copy()

    a["seed"] = a["seed"].astype(int)
    b["seed"] = b["seed"].astype(int)

    seeds_a_all = sorted(set(a["seed"].tolist()))
    seeds_b_all = sorted(set(b["seed"].tolist()))
    common = sorted(set(seeds_a_all).intersection(seeds_b_all))

    if len(common) == 0:
        raise ValueError(
            f"No overlapping seeds between {file_a} and {file_b}. "
            f"Seeds in {file_a.name}: {seeds_a_all}. "
            f"Seeds in {file_b.name}: {seeds_b_all}."
        )

    a = a[a["seed"].isin(common)].sort_values("seed").reset_index(drop=True)
    b = b[b["seed"].isin(common)].sort_values("seed").reset_index(drop=True)

    if len(a) < 3:
        raise ValueError(
            f"At least 3 overlapping seeds are recommended for statistical testing. "
            f"Found only {len(a)} between {file_a} and {file_b}. "
            f"Overlapping seeds: {common}. "
            f"Seeds in {file_a.name}: {seeds_a_all}. "
            f"Seeds in {file_b.name}: {seeds_b_all}."
        )

    return a, b


def run_pairwise_test(name_a, file_a, name_b, file_b, metrics=METRICS):
    a, b = load_and_align(file_a, file_b)
    rows = []

    for metric in metrics:
        if metric not in a.columns:
            raise ValueError(f"Metric '{metric}' not found in {file_a}")
        if metric not in b.columns:
            raise ValueError(f"Metric '{metric}' not found in {file_b}")

        x = pd.to_numeric(a[metric], errors="coerce").values.astype(float)
        y = pd.to_numeric(b[metric], errors="coerce").values.astype(float)

        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]

        if len(x) < 3:
            raise ValueError(
                f"Metric '{metric}' has fewer than 3 valid paired observations for "
                f"{name_a} vs {name_b}."
            )

        diff = x - y

        shapiro_p = np.nan
        recommended_test = "paired_t_test"

        if len(diff) >= 3:
            try:
                shapiro_p = float(shapiro(diff).pvalue)
                if shapiro_p < 0.05:
                    recommended_test = "wilcoxon"
            except Exception:
                shapiro_p = np.nan

        try:
            t_stat, t_p = ttest_rel(x, y)
            t_stat = float(t_stat)
            t_p = float(t_p)
        except Exception:
            t_stat, t_p = np.nan, np.nan

        try:
            # zero_method='wilcox' ignores zero-difference pairs more safely
            w_stat, w_p = wilcoxon(x, y, zero_method="wilcox")
            w_stat = float(w_stat)
            w_p = float(w_p)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        d = cohens_d_paired(x, y)

        rows.append({
            "comparison": f"{name_a} vs {name_b}",
            "metric": metric,
            "n_seeds": int(len(x)),
            f"{name_a}_mean": round(float(np.mean(x)), 6),
            f"{name_b}_mean": round(float(np.mean(y)), 6),
            "mean_difference": round(float(np.mean(diff)), 6),
            "shapiro_p_diff": round(shapiro_p, 6) if not np.isnan(shapiro_p) else np.nan,
            "recommended_test": recommended_test,
            "paired_t_stat": round(t_stat, 6) if not np.isnan(t_stat) else np.nan,
            "paired_t_p": round(t_p, 6) if not np.isnan(t_p) else np.nan,
            "wilcoxon_stat": round(w_stat, 6) if not np.isnan(w_stat) else np.nan,
            "wilcoxon_p": round(w_p, 6) if not np.isnan(w_p) else np.nan,
            "cohens_d": round(float(d), 6),
            "effect_size": interpret_cohens_d(d),
            "significant_ttest": "Yes" if (not np.isnan(t_p) and t_p < 0.05) else "No",
            "significant_wilcoxon": "Yes" if (not np.isnan(w_p) and w_p < 0.05) else "No",
        })

    return pd.DataFrame(rows)


def main():
    comparisons = [
        (
            "FL IID No-SMOTE",
            EVAL_DIR / "fl_final_round_iid_nosmote_multiseed.csv",
            "Centralized Tuned",
            EVAL_DIR / "centralized_tuned_multiseed.csv",
        ),
        (
            "FL IID SMOTE",
            EVAL_DIR / "fl_final_round_iid_smote_multiseed.csv",
            "Centralized Tuned",
            EVAL_DIR / "centralized_tuned_multiseed.csv",
        ),
        (
            "FL Non-IID No-SMOTE",
            EVAL_DIR / "fl_final_round_non_iid_nosmote_multiseed.csv",
            "Centralized Tuned",
            EVAL_DIR / "centralized_tuned_multiseed.csv",
        ),
        (
            "FL Non-IID SMOTE",
            EVAL_DIR / "fl_final_round_non_iid_smote_multiseed.csv",
            "Centralized Tuned",
            EVAL_DIR / "centralized_tuned_multiseed.csv",
        ),
        (
            "FL IID SMOTE",
            EVAL_DIR / "fl_final_round_iid_smote_multiseed.csv",
            "FL IID No-SMOTE",
            EVAL_DIR / "fl_final_round_iid_nosmote_multiseed.csv",
        ),
        (
            "FL Non-IID No-SMOTE",
            EVAL_DIR / "fl_final_round_non_iid_nosmote_multiseed.csv",
            "FL IID No-SMOTE",
            EVAL_DIR / "fl_final_round_iid_nosmote_multiseed.csv",
        ),
        (
            "FL Non-IID SMOTE",
            EVAL_DIR / "fl_final_round_non_iid_smote_multiseed.csv",
            "FL Non-IID No-SMOTE",
            EVAL_DIR / "fl_final_round_non_iid_nosmote_multiseed.csv",
        ),
    ]

    all_results = []
    skipped = []

    for name_a, file_a, name_b, file_b in comparisons:
        if not file_a.exists() or not file_b.exists():
            reason = f"Missing file(s): {file_a} | {file_b}"
            skipped.append((name_a, name_b, reason))
            print(f"Skipping comparison: {name_a} vs {name_b}\n  Reason: {reason}")
            continue

        try:
            df = run_pairwise_test(name_a, file_a, name_b, file_b)
            all_results.append(df)
        except ValueError as exc:
            skipped.append((name_a, name_b, str(exc)))
            print(f"Skipping comparison: {name_a} vs {name_b}\n  Reason: {exc}")

    if not all_results:
        raise FileNotFoundError("No valid comparison files found in evaluation/")

    results = pd.concat(all_results, ignore_index=True)

    out_csv = EVAL_DIR / "statistical_test_results.csv"
    out_xlsx = EVAL_DIR / "statistical_test_results.xlsx"

    results.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    try:
        results.to_excel(out_xlsx, index=False)
        print(f"Saved: {out_xlsx}")
    except Exception as exc:
        print(f"Skipping Excel export: {exc}")

    print(results.to_string(index=False))

    if skipped:
        skipped_df = pd.DataFrame(skipped, columns=["Method_A", "Method_B", "Reason"])
        skipped_csv = EVAL_DIR / "statistical_test_skipped.csv"
        skipped_df.to_csv(skipped_csv, index=False)
        print(f"\nSaved skipped comparisons log: {skipped_csv}")
        print("\nSkipped comparisons:")
        for name_a, name_b, reason in skipped:
            print(f"- {name_a} vs {name_b}: {reason}")


if __name__ == "__main__":
    main()