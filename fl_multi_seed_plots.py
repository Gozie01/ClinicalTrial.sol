from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation"
PLOT_DIR = EVAL_DIR / "multi_seed_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(name: str) -> str:
    label = name.replace("_multiseed", "")
    label = label.replace("_", " ")
    label = label.replace("non iid", "Non-IID")
    label = label.replace("iid", "IID")
    label = label.replace("nosmote", "No-SMOTE")
    label = label.replace("smote", "SMOTE")
    return label


def load_multiseed_csvs():
    csv_files = sorted(EVAL_DIR.glob("fl_metrics_*_multiseed.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No multi-seed CSV files found in {EVAL_DIR}. "
            f"Run fl_multi_seed_pipeline.py first."
        )

    data = {}
    for f in csv_files:
        name = f.stem.replace("fl_metrics_", "")
        df = pd.read_csv(f)
        required = {"seed", "round"}
        if required.issubset(df.columns):
            data[name] = df.sort_values(["seed", "round"])
    return data


def aggregate_by_round(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby("round")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    )
    grouped[f"{metric}_std"] = grouped[f"{metric}_std"].fillna(0.0)
    return grouped


def save_mean_std_lineplot(results, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for name, df in results.items():
        if metric not in df.columns:
            continue

        agg = aggregate_by_round(df, metric)
        x = agg["round"]
        y = agg[f"{metric}_mean"]
        yerr = agg[f"{metric}_std"]

        plt.plot(x, y, marker="o", linewidth=1.8, label=clean_label(name))
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.18)

    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Across Rounds (Mean ± Std Across Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_errorbar_plot(results, metric, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for name, df in results.items():
        if metric not in df.columns:
            continue

        agg = aggregate_by_round(df, metric)
        x = agg["round"]
        y = agg[f"{metric}_mean"]
        yerr = agg[f"{metric}_std"]

        plt.errorbar(
            x, y, yerr=yerr,
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=clean_label(name)
        )

    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Across Rounds With Error Bars")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_precision_recall_plot(results):
    plt.figure(figsize=(8, 5))

    for name, df in results.items():
        if "precision" not in df.columns or "recall" not in df.columns:
            continue

        p_agg = aggregate_by_round(df, "precision")
        r_agg = aggregate_by_round(df, "recall")

        plt.plot(
            p_agg["round"], p_agg["precision_mean"],
            marker="o", linewidth=1.5,
            label=f"{clean_label(name)} - Precision"
        )
        plt.plot(
            r_agg["round"], r_agg["recall_mean"],
            marker="s", linewidth=1.5,
            label=f"{clean_label(name)} - Recall"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Score")
    plt.title("Precision and Recall Across Rounds (Mean Across Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "precision_recall_mean.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_sensitivity_specificity_plot(results):
    plt.figure(figsize=(8, 5))

    for name, df in results.items():
        if "sensitivity" not in df.columns or "specificity" not in df.columns:
            continue

        sen_agg = aggregate_by_round(df, "sensitivity")
        spec_agg = aggregate_by_round(df, "specificity")

        plt.plot(
            sen_agg["round"], sen_agg["sensitivity_mean"],
            marker="o", linewidth=1.5,
            label=f"{clean_label(name)} - Sensitivity"
        )
        plt.plot(
            spec_agg["round"], spec_agg["specificity_mean"],
            marker="s", linewidth=1.5,
            label=f"{clean_label(name)} - Specificity"
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Score")
    plt.title("Sensitivity and Specificity Across Rounds (Mean Across Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "sensitivity_specificity_mean.png", dpi=300, bbox_inches="tight")
    plt.close()


def load_final_round_files():
    csv_files = sorted(EVAL_DIR.glob("fl_final_round_*_multiseed.csv"))
    final_data = {}

    for f in csv_files:
        name = f.stem.replace("fl_final_round_", "")
        df = pd.read_csv(f)
        if "seed" in df.columns:
            final_data[name] = df

    return final_data


def save_final_metric_barplot(final_data, metric, ylabel, filename):
    labels = []
    means = []
    stds = []

    for name, df in final_data.items():
        if metric in df.columns:
            labels.append(clean_label(name))
            means.append(df[metric].mean())
            stds.append(df[metric].std(ddof=1) if len(df) > 1 else 0.0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel(ylabel)
    plt.title(f"Final-Round {ylabel} (Mean ± Std Across Seeds)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_table(final_data):
    rows = []

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

    for name, df in final_data.items():
        row = {"Setting": clean_label(name)}
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std(ddof=1) if len(df) > 1 else 0.0
                row[f"{metric}_mean_std"] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = EVAL_DIR / "fl_multi_seed_mean_std_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved summary table: {out_path}")
    print(summary_df.to_string(index=False))


def main():
    results = load_multiseed_csvs()

    save_mean_std_lineplot(results, "accuracy", "Accuracy", "accuracy_mean_std.png")
    save_mean_std_lineplot(results, "f1", "F1 Score", "f1_mean_std.png")
    save_mean_std_lineplot(results, "auc", "AUC", "auc_mean_std.png")
    save_mean_std_lineplot(results, "balanced_accuracy", "Balanced Accuracy", "balanced_accuracy_mean_std.png")
    save_mean_std_lineplot(results, "mcc", "MCC", "mcc_mean_std.png")
    save_mean_std_lineplot(results, "elapsed_sec", "Elapsed Time (s)", "elapsed_time_mean_std.png")
    save_mean_std_lineplot(results, "threshold", "Threshold", "threshold_mean_std.png")

    save_errorbar_plot(results, "accuracy", "Accuracy", "accuracy_errorbar.png")
    save_errorbar_plot(results, "f1", "F1 Score", "f1_errorbar.png")
    save_errorbar_plot(results, "auc", "AUC", "auc_errorbar.png")

    save_precision_recall_plot(results)
    save_sensitivity_specificity_plot(results)

    final_data = load_final_round_files()
    save_final_metric_barplot(final_data, "accuracy", "Accuracy", "final_accuracy_barplot.png")
    save_final_metric_barplot(final_data, "f1", "F1 Score", "final_f1_barplot.png")
    save_final_metric_barplot(final_data, "auc", "AUC", "final_auc_barplot.png")
    save_final_metric_barplot(final_data, "mcc", "MCC", "final_mcc_barplot.png")

    save_summary_table(final_data)

    print(f"\nAll multi-seed plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()