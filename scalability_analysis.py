import argparse
import json
from pathlib import Path

import pandas as pd

from federated_mlp_run import run_experiment


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "scalability"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_scalability_sweep(
    dataset: str,
    site_grid,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    seed: int,
    dp_noise_multiplier: float,
    attack_type: str,
    lr: float = 1e-3,
    aggregation: str = "mean",
    use_smote: bool = False,
    feature_selection_top_k: int = 0,
    model_family: str = "default",
    attack_fraction: float = 0.2,
):
    rows = []
    for n_sites in site_grid:
        history_df = run_experiment(
            dataset=dataset,
            n_sites=n_sites,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
            non_iid=(dataset != "pima"),
            use_smote=use_smote,
            aggregation=aggregation,
            dp_noise_multiplier=dp_noise_multiplier,
            attack_type=attack_type,
            attack_fraction=attack_fraction if attack_type != "none" else 0.0,
            synthetic_sites=n_sites,
            synthetic_participants_per_site=100,
            synthetic_visits=10,
            synthetic_n_samples=max(10000, n_sites * 1200),
            lr=lr,
            feature_selection_top_k=feature_selection_top_k,
            model_family=model_family,
        )
        final_row = history_df.iloc[-1].to_dict()
        final_row["total_runtime_sec"] = float(history_df["elapsed_sec"].sum())
        final_row["sites_swept"] = int(n_sites)
        rows.append(final_row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_ct", choices=["pima", "breast_cancer", "synthetic_ct", "synthetic_large", "eicu_demo"])
    parser.add_argument("--site_grid", type=str, default="3,5,8,12")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dp_noise_multiplier", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "trimmed_mean"])
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--feature_selection_top_k", type=int, default=0)
    parser.add_argument("--model_family", type=str, default="default", choices=["default", "tabular_small"])
    parser.add_argument("--attack_fraction", type=float, default=0.2)
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "label_flip", "gaussian", "sign_flip", "byzantine"])
    args = parser.parse_args()

    site_grid = [int(item.strip()) for item in args.site_grid.split(",") if item.strip()]
    sweep_df = run_scalability_sweep(
        dataset=args.dataset,
        site_grid=site_grid,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        dp_noise_multiplier=args.dp_noise_multiplier,
        attack_type=args.attack_type,
        lr=args.lr,
        aggregation=args.aggregation,
        use_smote=args.smote,
        feature_selection_top_k=args.feature_selection_top_k,
        model_family=args.model_family,
        attack_fraction=args.attack_fraction,
    )

    tag = f"{args.dataset}_sites{'-'.join(map(str, site_grid))}_seed{args.seed}"
    csv_path = EVAL_DIR / f"scalability_{tag}.csv"
    json_path = EVAL_DIR / f"scalability_{tag}.json"
    sweep_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Scalability Summary ===")
    print(
        sweep_df[
            [
                "sites_swept",
                "f1",
                "auc",
                "elapsed_sec",
                "total_runtime_sec",
                "communication_mb_round",
                "approx_privacy_epsilon_upper",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
