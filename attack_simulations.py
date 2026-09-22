import argparse
import json
from pathlib import Path

import pandas as pd

from federated_mlp_run import run_experiment


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "attacks"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_attack_benchmark(
    dataset: str,
    attack_types,
    aggregation_methods,
    seed: int,
    rounds: int,
    local_epochs: int,
    batch_size: int = 96,
    lr: float = 1e-3,
    dp_noise_multiplier: float = 0.5,
    feature_selection_top_k: int = 0,
    model_family: str = "default",
    use_smote: bool = False,
    attack_fraction: float = 0.25,
    attack_strength: float = 0.35,
    n_sites: int | None = None,
    select_by: str = "final",
):
    rows = []
    for attack_type in attack_types:
        for aggregation in aggregation_methods:
            history_df = run_experiment(
                dataset=dataset,
                n_sites=(6 if dataset != "pima" else 4) if n_sites is None else n_sites,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                non_iid=True,
                use_smote=use_smote,
                aggregation=aggregation,
                dp_noise_multiplier=dp_noise_multiplier,
                attack_type=attack_type,
                attack_fraction=attack_fraction if attack_type != "none" else 0.0,
                attack_strength=attack_strength,
                synthetic_sites=10,
                synthetic_participants_per_site=100,
                synthetic_visits=10,
                synthetic_n_samples=12000,
                seed=seed,
                lr=lr,
                feature_selection_top_k=feature_selection_top_k,
                model_family=model_family,
            )
            if select_by == "best_val":
                selected_row = history_df.sort_values(["val_best_f1", "auc", "pr_auc", "f1"], ascending=False).iloc[0].to_dict()
            else:
                selected_row = history_df.iloc[-1].to_dict()
            selected_row["attack_type"] = attack_type
            selected_row["aggregation"] = aggregation
            rows.append(selected_row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic_ct", choices=["pima", "breast_cancer", "synthetic_ct", "synthetic_large", "eicu_demo"])
    parser.add_argument("--attack_types", type=str, default="none,label_flip,gaussian,gradient_poisoning,byzantine")
    parser.add_argument("--aggregations", type=str, default="mean,median,trimmed_mean")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dp_noise_multiplier", type=float, default=0.5)
    parser.add_argument("--feature_selection_top_k", type=int, default=0)
    parser.add_argument("--model_family", type=str, default="default", choices=["default", "tabular_small"])
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--attack_fraction", type=float, default=0.25)
    parser.add_argument("--attack_strength", type=float, default=0.35)
    parser.add_argument("--sites", type=int, default=0)
    parser.add_argument("--select_by", type=str, default="final", choices=["final", "best_val"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    attack_types = [item.strip() for item in args.attack_types.split(",") if item.strip()]
    aggregations = [item.strip() for item in args.aggregations.split(",") if item.strip()]
    results_df = run_attack_benchmark(
        dataset=args.dataset,
        attack_types=attack_types,
        aggregation_methods=aggregations,
        seed=args.seed,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dp_noise_multiplier=args.dp_noise_multiplier,
        feature_selection_top_k=args.feature_selection_top_k,
        model_family=args.model_family,
        use_smote=args.smote,
        attack_fraction=args.attack_fraction,
        attack_strength=args.attack_strength,
        n_sites=args.sites if args.sites > 0 else None,
        select_by=args.select_by,
    )

    tag = f"{args.dataset}_seed{args.seed}"
    csv_path = EVAL_DIR / f"attack_benchmark_{tag}.csv"
    json_path = EVAL_DIR / f"attack_benchmark_{tag}.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)

    print("\n=== Attack Benchmark Summary ===")
    print(results_df[["attack_type", "aggregation", "f1", "auc", "mcc", "communication_mb_round"]].to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
