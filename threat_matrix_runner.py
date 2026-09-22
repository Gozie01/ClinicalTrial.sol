import argparse
import json
from pathlib import Path

import pandas as pd

from federated_mlp_run import run_experiment
from governance_threat_simulation import GovernanceConfig, run_governance_threat_benchmark
from model_inversion_evaluation import run_model_inversion_benchmark


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "threat_matrix"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def fl_attack_row(dataset: str, attack_type: str, aggregation: str, seed: int):
    clean_df = run_experiment(
        dataset=dataset,
        n_sites=6,
        rounds=2,
        local_epochs=1,
        batch_size=96,
        non_iid=True,
        aggregation=aggregation,
        dp_noise_multiplier=0.5,
        attack_type="none",
        synthetic_sites=10,
        synthetic_participants_per_site=100,
        synthetic_visits=10,
        synthetic_n_samples=12000,
        seed=seed,
    )
    attack_df = run_experiment(
        dataset=dataset,
        n_sites=6,
        rounds=2,
        local_epochs=1,
        batch_size=96,
        non_iid=True,
        aggregation=aggregation,
        dp_noise_multiplier=0.5,
        attack_type=attack_type,
        attack_fraction=0.25,
        attack_strength=0.35,
        synthetic_sites=10,
        synthetic_participants_per_site=100,
        synthetic_visits=10,
        synthetic_n_samples=12000,
        seed=seed,
    )
    clean_final = clean_df.iloc[-1]
    attack_final = attack_df.iloc[-1]
    return {
        "threat": attack_type,
        "category": "federated",
        "defense_or_setting": aggregation,
        "clean_f1": float(clean_final["f1"]),
        "attack_f1": float(attack_final["f1"]),
        "delta_f1": float(attack_final["f1"] - clean_final["f1"]),
        "clean_auc": float(clean_final["auc"]),
        "attack_auc": float(attack_final["auc"]),
        "delta_auc": float(attack_final["auc"] - clean_final["auc"]),
    }


def run_threat_matrix(dataset: str, seed: int):
    rows = []

    rows.append(fl_attack_row(dataset=dataset, attack_type="gradient_poisoning", aggregation="mean", seed=seed))
    rows.append(fl_attack_row(dataset=dataset, attack_type="byzantine", aggregation="median", seed=seed))

    _, inversion_summary, _ = run_model_inversion_benchmark(
        dataset="breast_cancer" if dataset == "pima" else dataset,
        seed=seed,
        n_targets=4,
        reconstruction_steps=120,
    )
    for _, row in inversion_summary.iterrows():
        rows.append(
            {
                "threat": "model_inversion",
                "category": "privacy",
                "defense_or_setting": row["setting"],
                "recon_mse": float(row["recon_mse"]),
                "recon_mae": float(row["recon_mae"]),
                "recon_cosine": float(row["recon_cosine"]),
            }
        )

    _, governance_summary = run_governance_threat_benchmark(
        config=GovernanceConfig(seed=seed),
        trials_per_threat=24,
    )
    for _, row in governance_summary.iterrows():
        rows.append(
            {
                "threat": row["threat"],
                "category": "governance",
                "defense_or_setting": row["defense_profile"],
                "approval_rate": float(row["approval_rate"]),
                "attack_success_rate": float(row["attack_success_rate"]),
                "malicious_validator_share": float(row["malicious_validator_share"]),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_ct", choices=["pima", "breast_cancer", "synthetic_ct", "synthetic_large", "eicu_demo"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    matrix_df = run_threat_matrix(dataset=args.dataset, seed=args.seed)
    tag = f"{args.dataset}_seed{args.seed}"
    csv_path = EVAL_DIR / f"threat_matrix_{tag}.csv"
    json_path = EVAL_DIR / f"threat_matrix_{tag}.json"
    matrix_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matrix_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Threat Matrix Summary ===")
    print(matrix_df.to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
