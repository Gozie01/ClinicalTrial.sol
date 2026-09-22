from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation"
PLOT_DIR = EVAL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(name: str) -> str:
    label = name.replace("_", " ")
    label = label.replace("non iid", "Non-IID")
    label = label.replace("iid", "IID")
    label = label.replace("nosmote", "No-SMOTE")
    label = label.replace("smote", "SMOTE")
    return label


def load_csv_files():
    csv_files = sorted(EVAL_DIR.glob("fl_metrics_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {EVAL_DIR}")

    data = {}
    for f in csv_files:
        name = f.stem.replace("fl_metrics_", "")
        df = pd.read_csv(f)
        if "round" in df.columns:
            data[name] = df.sort_values("round")
    return data


def plot_metric(results, metric, ylabel):
    plt.figure(figsize=(8, 5))

    for name, df in results.items():
        if metric in df.columns:
            plt.plot(
                df["round"],
                df[metric],
                marker="o",
                linewidth=1.8,
                label=clean_label(name)
            )

    plt.xlabel("Communication Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs FL Rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{metric}.png", dpi=300)
    plt.close()


def main():
    results = load_csv_files()

    plot_metric(results, "accuracy", "Accuracy")
    plot_metric(results, "f1", "F1 Score")
    plot_metric(results, "auc", "AUC")
    plot_metric(results, "precision", "Precision")
    plot_metric(results, "recall", "Recall")
    plot_metric(results, "specificity", "Specificity")
    plot_metric(results, "mcc", "MCC")
    plot_metric(results, "balanced_accuracy", "Balanced Accuracy")
    plot_metric(results, "elapsed_sec", "Time (s)")
    plot_metric(results, "threshold", "Threshold")

    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()