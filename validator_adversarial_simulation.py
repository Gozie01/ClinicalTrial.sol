import math
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class ValidatorScenarioConfig:
    n_validators: int = 12
    malicious_fraction: float = 0.25
    quorum_ratio: float = 0.67
    epochs: int = 200
    base_delay_slots: float = 1.5
    delay_jitter_slots: float = 0.8
    target_finality_slots: float = 3.5
    fork_probability: float = 0.05
    malicious_consensus_probability: float = 0.10
    conflicting_reward_probability: float = 0.06
    delayed_finality_probability: float = 0.08
    seed: int = 7


SCENARIO_PROFILES = {
    "baseline": {},
    "validator_collusion": {"malicious_fraction": 0.42, "quorum_ratio": 0.58, "malicious_consensus_probability": 0.35},
    "delayed_finality": {"base_delay_slots": 2.8, "delay_jitter_slots": 1.4, "delayed_finality_probability": 0.35},
    "malicious_consensus_behavior": {"malicious_fraction": 0.34, "malicious_consensus_probability": 0.45},
    "chain_forks": {"fork_probability": 0.24, "conflicting_reward_probability": 0.20},
    "conflicting_reward_execution": {
        "fork_probability": 0.28,
        "malicious_fraction": 0.50,
        "quorum_ratio": 0.58,
        "malicious_consensus_probability": 0.40,
        "conflicting_reward_probability": 0.45,
    },
}


def _build_validator_graph(n_validators: int, rng: np.random.Generator):
    graph = nx.watts_strogatz_graph(n_validators, k=min(max(4, n_validators // 3), n_validators - 1), p=0.28, seed=int(rng.integers(1, 1_000_000)))
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for left, right in zip(components[:-1], components[1:]):
            graph.add_edge(next(iter(left)), next(iter(right)))
    return graph


def _simulate_scenario(scenario: str, config: ValidatorScenarioConfig):
    rng = np.random.default_rng(config.seed)
    graph = _build_validator_graph(config.n_validators, rng)
    quorum = math.ceil(config.n_validators * config.quorum_ratio)
    malicious_count = max(1, int(round(config.n_validators * config.malicious_fraction)))
    malicious_validators = set(rng.choice(config.n_validators, size=malicious_count, replace=False).tolist())
    rows = []

    for epoch in range(config.epochs):
        proposer = int(rng.integers(0, config.n_validators))
        proposer_malicious = proposer in malicious_validators
        forked = rng.random() < config.fork_probability
        malicious_consensus = proposer_malicious and rng.random() < config.malicious_consensus_probability
        conflicting_claim = forked and rng.random() < config.conflicting_reward_probability
        forced_delay = rng.random() < config.delayed_finality_probability

        branch_votes = {"canonical": 0, "fork": 0}
        branch_delays = {"canonical": [], "fork": []}
        double_signed = 0
        withholders = 0

        for validator in range(config.n_validators):
            hop_distance = nx.shortest_path_length(graph, proposer, validator)
            delay = (
                config.base_delay_slots
                + hop_distance * 0.35
                + abs(rng.normal(0.0, config.delay_jitter_slots))
            )
            if forced_delay:
                delay += rng.uniform(1.0, 2.5)

            is_malicious = validator in malicious_validators
            if is_malicious and malicious_consensus and rng.random() < 0.30:
                withholders += 1
                continue

            if forked and is_malicious and rng.random() < 0.65:
                branch_votes["fork"] += 1
                branch_delays["fork"].append(delay + rng.uniform(0.0, 1.2))
                if conflicting_claim and rng.random() < 0.60:
                    branch_votes["canonical"] += 1
                    branch_delays["canonical"].append(delay)
                    double_signed += 1
                continue

            if forked and not is_malicious:
                split_probability = 0.22 if conflicting_claim else 0.08
                if rng.random() < split_probability:
                    branch_votes["fork"] += 1
                    branch_delays["fork"].append(delay + rng.uniform(0.0, 0.8))
                    continue

            branch_votes["canonical"] += 1
            branch_delays["canonical"].append(delay)

        finalized_branches = []
        for branch_name in ("canonical", "fork"):
            if branch_votes[branch_name] >= quorum:
                delays = sorted(branch_delays[branch_name])
                finalized_branches.append((branch_name, delays[quorum - 1]))

        finalized_branches.sort(key=lambda item: item[1])
        finality_slots = finalized_branches[0][1] if finalized_branches else config.target_finality_slots + 3.0
        delayed_finality = finality_slots > config.target_finality_slots
        canonical_execution_success = int(any(branch == "canonical" for branch, _ in finalized_branches))
        safety_violation_without_guard = int(
            conflicting_claim
            and branch_votes["canonical"] >= quorum
            and branch_votes["fork"] >= quorum
        )
        guard_blocked_conflict = safety_violation_without_guard
        conflicting_reward_execution = int(conflicting_claim)

        rows.append(
            {
                "scenario": scenario,
                "epoch": epoch,
                "malicious_share": len(malicious_validators) / config.n_validators,
                "proposer_malicious": int(proposer_malicious),
                "forked": int(forked),
                "malicious_consensus": int(malicious_consensus),
                "conflicting_reward_execution": conflicting_reward_execution,
                "double_signed": int(double_signed),
                "withholders": int(withholders),
                "canonical_votes": int(branch_votes["canonical"]),
                "fork_votes": int(branch_votes["fork"]),
                "quorum": int(quorum),
                "finalized_branch_count": int(len(finalized_branches)),
                "finality_slots": float(finality_slots),
                "delayed_finality": int(delayed_finality),
                "canonical_execution_success": canonical_execution_success,
                "guard_blocked_conflict": int(guard_blocked_conflict),
                "safety_violation_without_guard": int(safety_violation_without_guard),
            }
        )

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby("scenario")[
            [
                "malicious_share",
                "forked",
                "malicious_consensus",
                "conflicting_reward_execution",
                "double_signed",
                "withholders",
                "finalized_branch_count",
                "finality_slots",
                "delayed_finality",
                "canonical_execution_success",
                "guard_blocked_conflict",
                "safety_violation_without_guard",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "forked": "fork_rate",
                "malicious_consensus": "malicious_consensus_rate",
                "conflicting_reward_execution": "conflicting_reward_execution_rate",
                "double_signed": "double_sign_events",
                "withholders": "withholding_events",
                "finalized_branch_count": "mean_finalized_branch_count",
                "finality_slots": "mean_finality_slots",
                "delayed_finality": "delayed_finality_rate",
                "canonical_execution_success": "canonical_execution_success_rate",
                "guard_blocked_conflict": "guard_blocked_conflict_rate",
                "safety_violation_without_guard": "safety_violation_without_guard_rate",
            }
        )
    )
    return raw_df, summary_df


def run_validator_suite(base_config: ValidatorScenarioConfig, seeds):
    raw_frames = []
    summary_frames = []
    for scenario_name, overrides in SCENARIO_PROFILES.items():
        for seed in seeds:
            config = replace(base_config, seed=seed, **overrides)
            raw_df, summary_df = _simulate_scenario(scenario_name, config)
            raw_frames.append(raw_df.assign(seed=seed))
            summary_frames.append(summary_df.assign(seed=seed))

    raw_all = pd.concat(raw_frames, ignore_index=True)
    summary_seed = pd.concat(summary_frames, ignore_index=True)
    metric_cols = [
        "malicious_share",
        "fork_rate",
        "malicious_consensus_rate",
        "conflicting_reward_execution_rate",
        "double_sign_events",
        "withholding_events",
        "mean_finalized_branch_count",
        "mean_finality_slots",
        "delayed_finality_rate",
        "canonical_execution_success_rate",
        "guard_blocked_conflict_rate",
        "safety_violation_without_guard_rate",
    ]
    summary = summary_seed.groupby("scenario")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "scenario" if col == ("scenario", "") else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return raw_all, summary_seed, summary
