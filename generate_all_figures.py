from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "evaluation" / "paper_bundle"
BLOCKCHAIN_DIR = ROOT / "evaluation" / "blockchain_bundle"
FIG_DIR = ROOT / "evaluation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_real_fl():
    df = pd.read_csv(PAPER_DIR / "real_data_summary.csv")
    metrics = ["f1_mean", "auc_mean", "pr_auc_mean", "mcc_mean"]
    labels = ["F1", "AUC", "PR-AUC", "MCC"]
    long_df = pd.DataFrame(
        {
            "setting": df["setting"].repeat(len(metrics)),
            "metric": labels * len(df),
            "value": [row[m] for _, row in df.iterrows() for m in metrics],
        }
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=long_df, x="metric", y="value", hue="setting", palette="Set2")
    plt.ylim(0, 1.0)
    plt.title("Real-Data Federated Learning Performance")
    save_fig(FIG_DIR / "figure_real_fl_comparison.png")


def plot_real_baselines():
    df = pd.read_csv(PAPER_DIR / "real_data_benchmarks.csv")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="f1_mean", hue="model", palette="crest", legend=False)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Centralized Real-Data Baselines")
    save_fig(FIG_DIR / "figure_real_baselines.png")


def plot_rl_summary():
    df = pd.read_csv(PAPER_DIR / "synthetic_rl_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.barplot(data=df, x="method", y="total_reward_mean", hue="method", palette="rocket", legend=False, ax=axes[0])
    axes[0].set_title("RL Mean Reward")
    sns.barplot(data=df, x="method", y="attendance_rate_mean", hue="method", palette="viridis", legend=False, ax=axes[1])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("RL Attendance")
    sns.barplot(data=df, x="method", y="dropouts_mean", hue="method", palette="magma", legend=False, ax=axes[2])
    axes[2].set_title("RL Dropouts")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    save_fig(FIG_DIR / "figure_rl_summary.png")


def plot_attack_summary():
    df = pd.read_csv(PAPER_DIR / "security_attack_summary.csv")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="attack_type", y="f1_mean", hue="aggregation", palette="tab10")
    plt.ylim(0, 1.0)
    plt.title("Adversarial FL Robustness")
    save_fig(FIG_DIR / "figure_attack_summary.png")


def plot_privacy_summary():
    df = pd.read_csv(PAPER_DIR / "model_inversion_summary.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="setting", y="recon_cosine_mean", hue="setting", palette="flare", legend=False)
    plt.ylim(0, max(0.6, df["recon_cosine_mean"].max() * 1.15))
    plt.title("Model Inversion Reconstruction Similarity")
    save_fig(FIG_DIR / "figure_privacy_summary.png")


def plot_governance_summary():
    df = pd.read_csv(PAPER_DIR / "governance_summary.csv")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="threat", y="attack_success_rate_mean", hue="defense_profile", palette="Set1")
    plt.xticks(rotation=18, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Governance Threat Success Rates")
    save_fig(FIG_DIR / "figure_governance_summary.png")


def plot_scalability_summary():
    df = pd.read_csv(PAPER_DIR / "scalability_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=df, x="sites_swept", y="total_runtime_sec_mean", marker="o", ax=axes[0])
    axes[0].set_title("Scalability Runtime")
    sns.lineplot(data=df, x="sites_swept", y="communication_mb_total_mean", marker="o", ax=axes[1])
    axes[1].set_title("Scalability Communication")
    save_fig(FIG_DIR / "figure_scalability_summary.png")


def plot_contract_security():
    df = pd.read_csv(BLOCKCHAIN_DIR / "contract_security_summary.csv")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x="scenario", y="blocked_rate", hue="scenario", palette="cubehelix", legend=False)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Contract Runtime Security Matrix")
    save_fig(FIG_DIR / "figure_contract_security_matrix.png")


def plot_validator_summary():
    df = pd.read_csv(BLOCKCHAIN_DIR / "validator_threat_summary.csv")
    plot_df = df[
        [
            "scenario",
            "delayed_finality_rate_mean",
            "guard_blocked_conflict_rate_mean",
            "canonical_execution_success_rate_mean",
        ]
    ].copy()
    long_df = plot_df.melt(id_vars="scenario", var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_df, x="scenario", y="value", hue="metric", palette="Spectral")
    plt.xticks(rotation=18, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Validator Threat Simulation Summary")
    save_fig(FIG_DIR / "figure_validator_threat_summary.png")


def plot_slither_summary():
    df = pd.read_csv(BLOCKCHAIN_DIR / "slither_severity_counts.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="impact", y="count", hue="impact", palette="coolwarm", legend=False)
    plt.title("Slither Severity Counts")
    save_fig(FIG_DIR / "figure_slither_summary.png")


def plot_network_comparison():
    df = pd.read_csv(BLOCKCHAIN_DIR / "network_comparison_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.barplot(data=df, x="network", y="block_latency_ms", hue="network", palette="Set2", legend=False, ax=axes[0, 0])
    axes[0, 0].set_title("RPC Block Query Latency")
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[0, 0].tick_params(axis="x", rotation=15)

    sns.barplot(data=df, x="network", y="avg_block_time_sec", hue="network", palette="Set3", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("Average Block Interval")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].tick_params(axis="x", rotation=15)

    sns.barplot(data=df, x="network", y="gas_price_wei", hue="network", palette="crest", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Gas Price")
    axes[1, 0].set_ylabel("Wei")
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=15)

    sns.barplot(
        data=df,
        x="network",
        y="deployment_cost_paris_native",
        hue="network",
        palette="rocket",
        legend=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Projected Contract Deployment Cost")
    axes[1, 1].set_ylabel("Native token")
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=15)

    save_fig(FIG_DIR / "figure_network_comparison.png")


def write_figure_index():
    lines = [
        "# Figure Index",
        "",
        "- figure_real_fl_comparison.png",
        "- figure_real_baselines.png",
        "- figure_rl_summary.png",
        "- figure_attack_summary.png",
        "- figure_privacy_summary.png",
        "- figure_governance_summary.png",
        "- figure_scalability_summary.png",
        "- figure_contract_security_matrix.png",
        "- figure_validator_threat_summary.png",
        "- figure_slither_summary.png",
        "- figure_network_comparison.png",
    ]
    (FIG_DIR / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    plot_real_fl()
    plot_real_baselines()
    plot_rl_summary()
    plot_attack_summary()
    plot_privacy_summary()
    plot_governance_summary()
    plot_scalability_summary()
    plot_contract_security()
    plot_validator_summary()
    plot_slither_summary()
    plot_network_comparison()
    write_figure_index()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
