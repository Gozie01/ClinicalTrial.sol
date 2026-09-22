import json
from pathlib import Path

import pandas as pd

from attack_simulations import run_attack_benchmark
from ct_simulation import CTSimulatorConfig
from eicu_dataset import prepare_eicu_demo_dataset
from federated_mlp_run import run_experiment
from governance_threat_simulation import GovernanceConfig, run_governance_threat_benchmark
from model_inversion_evaluation import run_model_inversion_benchmark
from real_data_benchmarks import run_benchmarks
from rl_ct_experiments import run_multi_seed_experiments
from scalability_analysis import run_scalability_sweep


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "paper_bundle"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

REAL_SEEDS = [7, 21, 42]
SYNTHETIC_SEEDS = [7, 21, 42]
SECURITY_SEEDS = [7, 21]
SCALING_SEEDS = [7, 21]

REAL_FL_BASE = {
    "dataset": "eicu_demo",
    "n_sites": 8,
    "rounds": 12,
    "local_epochs": 3,
    "batch_size": 32,
    "lr": 5e-4,
    "use_smote": True,
    "feature_selection_top_k": 60,
    "model_family": "tabular_small",
}


def flatten_agg_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = [
        col if isinstance(col, str) else "_".join([item for item in col if item])
        for col in df.columns.to_flat_index()
    ]
    return df


def aggregate_mean_std(df: pd.DataFrame, group_cols, metric_cols):
    grouped = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    return flatten_agg_columns(grouped)


def select_best_validation_row(history_df: pd.DataFrame):
    ranked = history_df.sort_values(
        ["val_best_f1", "auc", "pr_auc", "f1", "mcc"],
        ascending=False,
    )
    return ranked.iloc[0].to_dict()


def run_real_data_suite():
    prepared_df, cache_path, meta_path = prepare_eicu_demo_dataset(refresh=False)

    fl_settings = [
        {
            "setting": "eicu_clean",
            "aggregation": "mean",
            "dp_noise_multiplier": 0.0,
            "attack_type": "none",
            "attack_fraction": 0.0,
            "attack_strength": 0.0,
        },
        {
            "setting": "eicu_dp_moderate",
            "aggregation": "mean",
            "dp_noise_multiplier": 0.05,
            "attack_type": "none",
            "attack_fraction": 0.0,
            "attack_strength": 0.0,
        },
        {
            "setting": "eicu_dp_attack_robust",
            "aggregation": "median",
            "dp_noise_multiplier": 0.05,
            "attack_type": "gradient_poisoning",
            "attack_fraction": 0.25,
            "attack_strength": 0.35,
        },
    ]

    fl_rows = []
    benchmark_frames = []
    selected_feature_map = {}

    for seed in REAL_SEEDS:
        benchmarks_df, selected_cols = run_benchmarks(seed=seed, feature_top_k=60)
        benchmark_frames.append(benchmarks_df.assign(seed=seed))
        selected_feature_map[str(seed)] = selected_cols

        for config in fl_settings:
            history_df = run_experiment(
                seed=seed,
                aggregation=config["aggregation"],
                dp_noise_multiplier=config["dp_noise_multiplier"],
                attack_type=config["attack_type"],
                attack_fraction=config["attack_fraction"],
                attack_strength=config["attack_strength"],
                **REAL_FL_BASE,
            )
            best_row = select_best_validation_row(history_df)
            fl_rows.append({"seed": seed, **config, **best_row})

    fl_raw_df = pd.DataFrame(fl_rows)
    fl_summary_df = aggregate_mean_std(
        fl_raw_df,
        ["setting"],
        [
            "round",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
            "pr_auc",
            "mcc",
            "threshold",
            "val_best_f1",
            "approx_privacy_epsilon_upper",
        ],
    )

    benchmark_raw_df = pd.concat(benchmark_frames, ignore_index=True)
    benchmark_summary_df = aggregate_mean_std(
        benchmark_raw_df,
        ["model"],
        ["accuracy", "balanced_accuracy", "f1", "auc", "pr_auc", "mcc", "threshold"],
    )

    return {
        "prepared_dataset_path": str(cache_path),
        "prepared_dataset_meta_path": str(meta_path),
        "prepared_rows": int(len(prepared_df)),
        "prepared_positive": int(prepared_df["Outcome"].sum()),
        "selected_features": selected_feature_map,
        "fl_raw": fl_raw_df,
        "fl_summary": fl_summary_df,
        "bench_raw": benchmark_raw_df,
        "bench_summary": benchmark_summary_df,
    }


def run_synthetic_behavior_suite():
    config = CTSimulatorConfig(
        n_sites=8,
        participants_per_site=40,
        horizon=20,
        initial_budget=5000.0,
        communication_base_delay_ms=125.0,
        communication_delay_std_ms=35.0,
        packet_loss_rate=0.035,
        seed=SYNTHETIC_SEEDS[0],
    )
    train_df, eval_df, summary_by_seed_df, summary_df = run_multi_seed_experiments(
        config=config,
        seeds=SYNTHETIC_SEEDS,
        episodes=140,
        eval_episodes=30,
    )
    return {
        "config": config,
        "train": train_df,
        "eval": eval_df,
        "summary_by_seed": summary_by_seed_df,
        "summary": summary_df,
    }


def run_security_suite():
    attack_frames = []
    inversion_frames = []
    governance_frames = []

    for seed in SECURITY_SEEDS:
        attack_df = run_attack_benchmark(
            dataset="eicu_demo",
            attack_types=["none", "gradient_poisoning", "byzantine"],
            aggregation_methods=["mean", "median", "trimmed_mean"],
            seed=seed,
            rounds=12,
            local_epochs=3,
            batch_size=32,
            lr=5e-4,
            dp_noise_multiplier=0.05,
            feature_selection_top_k=60,
            model_family="tabular_small",
            use_smote=True,
            attack_fraction=0.25,
            attack_strength=0.35,
            n_sites=8,
            select_by="best_val",
        )
        attack_frames.append(attack_df.assign(seed=seed))

        inversion_df, _, _ = run_model_inversion_benchmark(
            dataset="eicu_demo",
            seed=seed,
            n_targets=4,
            reconstruction_steps=120,
            feature_selection_top_k=60,
            model_family="tabular_small",
            lr=5e-4,
        )
        inversion_frames.append(inversion_df.assign(seed=seed))

        _, governance_summary = run_governance_threat_benchmark(
            config=GovernanceConfig(
                n_validators=7,
                malicious_validator_fraction=0.28,
                collusion_threshold=0.51,
                participants=80,
                visits_per_participant=12,
                seed=seed,
            ),
            trials_per_threat=60,
        )
        governance_frames.append(governance_summary.assign(seed=seed))

    attack_raw_df = pd.concat(attack_frames, ignore_index=True)
    attack_summary_df = aggregate_mean_std(
        attack_raw_df,
        ["attack_type", "aggregation"],
        ["accuracy", "balanced_accuracy", "f1", "auc", "pr_auc", "mcc", "communication_mb_round"],
    )

    inversion_raw_df = pd.concat(inversion_frames, ignore_index=True)
    inversion_summary_df = aggregate_mean_std(
        inversion_raw_df,
        ["setting"],
        ["recon_mse", "recon_mae", "recon_cosine"],
    )

    governance_raw_df = pd.concat(governance_frames, ignore_index=True)
    governance_summary_df = aggregate_mean_std(
        governance_raw_df,
        ["threat", "defense_profile"],
        ["approval_rate", "attack_success_rate", "malicious_validator_share"],
    )

    return {
        "attack_raw": attack_raw_df,
        "attack_summary": attack_summary_df,
        "inversion_raw": inversion_raw_df,
        "inversion_summary": inversion_summary_df,
        "governance_raw": governance_raw_df,
        "governance_summary": governance_summary_df,
    }


def run_scalability_suite():
    frames = []
    for seed in SCALING_SEEDS:
        sweep_df = run_scalability_sweep(
            dataset="synthetic_ct",
            site_grid=[4, 8, 12, 16],
            rounds=4,
            local_epochs=2,
            batch_size=128,
            seed=seed,
            dp_noise_multiplier=0.0,
            attack_type="none",
            lr=7e-4,
            aggregation="mean",
            use_smote=False,
            feature_selection_top_k=0,
            model_family="default",
        )
        frames.append(sweep_df.assign(seed=seed))

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["communication_mb_total"] = raw_df["communication_mb_round"] * raw_df["round"]
    summary_df = aggregate_mean_std(
        raw_df,
        ["sites_swept"],
        [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "pr_auc",
            "mcc",
            "elapsed_sec",
            "total_runtime_sec",
            "communication_mb_round",
            "communication_mb_total",
        ],
    )
    return {"raw": raw_df, "summary": summary_df}


def build_results_note(real_suite, synthetic_suite, security_suite, scalability_suite):
    real_top = real_suite["fl_summary"].set_index("setting")
    bench_top = real_suite["bench_summary"].set_index("model")
    synth_top = synthetic_suite["summary"].set_index("method")
    attack_top = security_suite["attack_summary"].set_index(["attack_type", "aggregation"])
    inversion_top = security_suite["inversion_summary"].set_index("setting")
    governance_top = security_suite["governance_summary"].set_index(["threat", "defense_profile"])

    lines = [
        "# Results Note",
        "",
        "## Real-Data FL",
        f"- Prepared eICU demo cohort: `{real_suite['prepared_rows']}` stays with `{real_suite['prepared_positive']}` positive outcomes.",
        f"- Clean FL mean F1: `{real_top.loc['eicu_clean', 'f1_mean']:.4f}`, AUC: `{real_top.loc['eicu_clean', 'auc_mean']:.4f}`, PR-AUC: `{real_top.loc['eicu_clean', 'pr_auc_mean']:.4f}`.",
        f"- Moderate-DP FL mean F1: `{real_top.loc['eicu_dp_moderate', 'f1_mean']:.4f}`, AUC: `{real_top.loc['eicu_dp_moderate', 'auc_mean']:.4f}`, epsilon upper bound: `{real_top.loc['eicu_dp_moderate', 'approx_privacy_epsilon_upper_mean']:.2f}`.",
        f"- Robust attacked FL mean F1: `{real_top.loc['eicu_dp_attack_robust', 'f1_mean']:.4f}`, AUC: `{real_top.loc['eicu_dp_attack_robust', 'auc_mean']:.4f}`.",
        "",
        "## Centralized Upper Bounds",
        f"- Logistic regression mean AUC: `{bench_top.loc['logistic_regression', 'auc_mean']:.4f}`.",
        f"- HistGradientBoosting mean F1: `{bench_top.loc['hist_gradient_boosting', 'f1_mean']:.4f}`, PR-AUC: `{bench_top.loc['hist_gradient_boosting', 'pr_auc_mean']:.4f}`.",
        f"- Centralized MLP mean F1: `{bench_top.loc['centralized_mlp_topk', 'f1_mean']:.4f}`.",
        "",
        "## Synthetic CT Simulator",
        f"- DQN mean reward: `{synth_top.loc['dqn', 'total_reward_mean']:.4f}`, attendance: `{synth_top.loc['dqn', 'attendance_rate_mean']:.4f}`, dropouts: `{synth_top.loc['dqn', 'dropouts_mean']:.4f}`.",
        f"- PPO mean reward: `{synth_top.loc['ppo', 'total_reward_mean']:.4f}`, attendance: `{synth_top.loc['ppo', 'attendance_rate_mean']:.4f}`.",
        f"- Static mean reward: `{synth_top.loc['static', 'total_reward_mean']:.4f}`, attendance: `{synth_top.loc['static', 'attendance_rate_mean']:.4f}`, dropouts: `{synth_top.loc['static', 'dropouts_mean']:.4f}`.",
        "",
        "## Security and Privacy",
        f"- Gradient poisoning with median aggregation mean F1: `{attack_top.loc[('gradient_poisoning', 'median'), 'f1_mean']:.4f}`.",
        f"- Byzantine attack with median aggregation mean F1: `{attack_top.loc[('byzantine', 'median'), 'f1_mean']:.4f}`.",
        f"- Model inversion cosine similarity falls from `{inversion_top.loc['no_defense', 'recon_cosine_mean']:.4f}` to `{inversion_top.loc['dp_strong', 'recon_cosine_mean']:.4f}` under strong DP.",
        f"- Guarded replay-attack success rate: `{governance_top.loc[('replay_attack', 'guarded'), 'attack_success_rate_mean']:.4f}`.",
        f"- Unguarded replay-attack success rate: `{governance_top.loc[('replay_attack', 'unguarded'), 'attack_success_rate_mean']:.4f}`.",
        "",
        "## Scalability",
        f"- 4-site mean runtime: `{scalability_suite['summary'].set_index('sites_swept').loc[4, 'total_runtime_sec_mean']:.4f}` seconds.",
        f"- 16-site mean runtime: `{scalability_suite['summary'].set_index('sites_swept').loc[16, 'total_runtime_sec_mean']:.4f}` seconds.",
        f"- 16-site mean communication volume: `{scalability_suite['summary'].set_index('sites_swept').loc[16, 'communication_mb_total_mean']:.4f}` MB.",
    ]
    return "\n".join(lines) + "\n"


def main():
    real_suite = run_real_data_suite()
    synthetic_suite = run_synthetic_behavior_suite()
    security_suite = run_security_suite()
    scalability_suite = run_scalability_suite()

    file_map = {
        "real_data_summary": EVAL_DIR / "real_data_summary.csv",
        "real_data_fl_by_seed": EVAL_DIR / "real_data_fl_by_seed.csv",
        "real_data_benchmarks": EVAL_DIR / "real_data_benchmarks.csv",
        "real_data_benchmarks_by_seed": EVAL_DIR / "real_data_benchmarks_by_seed.csv",
        "synthetic_rl_summary": EVAL_DIR / "synthetic_rl_summary.csv",
        "synthetic_rl_summary_by_seed": EVAL_DIR / "synthetic_rl_summary_by_seed.csv",
        "synthetic_rl_eval": EVAL_DIR / "synthetic_rl_eval.csv",
        "security_attack_summary": EVAL_DIR / "security_attack_summary.csv",
        "security_attacks_by_seed": EVAL_DIR / "security_attacks_by_seed.csv",
        "model_inversion_summary": EVAL_DIR / "model_inversion_summary.csv",
        "model_inversion_by_seed": EVAL_DIR / "model_inversion_by_seed.csv",
        "governance_summary": EVAL_DIR / "governance_summary.csv",
        "governance_by_seed": EVAL_DIR / "governance_by_seed.csv",
        "scalability_summary": EVAL_DIR / "scalability_summary.csv",
        "scalability_by_seed": EVAL_DIR / "scalability_by_seed.csv",
        "results_note": EVAL_DIR / "RESULTS_NOTE.md",
        "meta": EVAL_DIR / "bundle_meta.json",
    }

    real_suite["fl_summary"].to_csv(file_map["real_data_summary"], index=False)
    real_suite["fl_raw"].to_csv(file_map["real_data_fl_by_seed"], index=False)
    real_suite["bench_summary"].to_csv(file_map["real_data_benchmarks"], index=False)
    real_suite["bench_raw"].to_csv(file_map["real_data_benchmarks_by_seed"], index=False)

    synthetic_suite["summary"].to_csv(file_map["synthetic_rl_summary"], index=False)
    synthetic_suite["summary_by_seed"].to_csv(file_map["synthetic_rl_summary_by_seed"], index=False)
    synthetic_suite["eval"].to_csv(file_map["synthetic_rl_eval"], index=False)

    security_suite["attack_summary"].to_csv(file_map["security_attack_summary"], index=False)
    security_suite["attack_raw"].to_csv(file_map["security_attacks_by_seed"], index=False)
    security_suite["inversion_summary"].to_csv(file_map["model_inversion_summary"], index=False)
    security_suite["inversion_raw"].to_csv(file_map["model_inversion_by_seed"], index=False)
    security_suite["governance_summary"].to_csv(file_map["governance_summary"], index=False)
    security_suite["governance_raw"].to_csv(file_map["governance_by_seed"], index=False)

    scalability_suite["summary"].to_csv(file_map["scalability_summary"], index=False)
    scalability_suite["raw"].to_csv(file_map["scalability_by_seed"], index=False)

    results_note = build_results_note(real_suite, synthetic_suite, security_suite, scalability_suite)
    file_map["results_note"].write_text(results_note, encoding="utf-8")

    meta = {
        "real_seeds": REAL_SEEDS,
        "synthetic_seeds": SYNTHETIC_SEEDS,
        "security_seeds": SECURITY_SEEDS,
        "scaling_seeds": SCALING_SEEDS,
        "real_dataset_rows": real_suite["prepared_rows"],
        "real_dataset_positive": real_suite["prepared_positive"],
        "prepared_dataset_path": real_suite["prepared_dataset_path"],
        "prepared_dataset_meta_path": real_suite["prepared_dataset_meta_path"],
        "selected_features_by_seed": real_suite["selected_features"],
        "files": {name: str(path) for name, path in file_map.items()},
    }
    with open(file_map["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Submission-Grade Paper Bundle Generated ===")
    print(f"Real dataset rows: {real_suite['prepared_rows']}")
    print(f"Real dataset positives: {real_suite['prepared_positive']}")
    print("\nReal-data FL summary:")
    print(real_suite["fl_summary"].to_string(index=False))
    print("\nCentralized benchmark summary:")
    print(real_suite["bench_summary"].to_string(index=False))
    print("\nSynthetic RL summary:")
    print(synthetic_suite["summary"].to_string(index=False))
    print("\nAttack summary:")
    print(security_suite["attack_summary"].to_string(index=False))
    print("\nGovernance summary:")
    print(security_suite["governance_summary"].to_string(index=False))
    print("\nScalability summary:")
    print(scalability_suite["summary"].to_string(index=False))
    print("\nSaved bundle:")
    for name, path in file_map.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
