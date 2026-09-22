from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GovernanceConfig:
    n_validators: int = 7
    malicious_validator_fraction: float = 0.28
    collusion_threshold: float = 0.51
    participants: int = 80
    visits_per_participant: int = 12
    seed: int = 7


THREATS = [
    "replay_attack",
    "malicious_participant_manipulation",
    "smart_contract_exploit_attempt",
    "validator_collusion",
]

DEFENSE_PROFILES = ["guarded", "unguarded"]


def _bounded_probability(value: float):
    return float(np.clip(value, 0.0, 1.0))


def _simulate_single_trial(threat: str, defense_profile: str, config: GovernanceConfig, rng: np.random.Generator):
    malicious_share = _bounded_probability(
        rng.normal(loc=config.malicious_validator_fraction, scale=0.035)
    )

    if threat == "replay_attack":
        attack_success_rate = 0.0 if defense_profile == "guarded" else 1.0
        approval_rate = _bounded_probability(0.965 - 0.03 * malicious_share)
    elif threat == "malicious_participant_manipulation":
        attack_success_rate = 0.015 if defense_profile == "guarded" else 0.88
        approval_rate = _bounded_probability(0.955 - 0.05 * malicious_share)
    elif threat == "smart_contract_exploit_attempt":
        guarded_base = 0.01 + 0.08 * max(0.0, malicious_share - config.collusion_threshold)
        attack_success_rate = guarded_base if defense_profile == "guarded" else 0.82
        approval_rate = _bounded_probability(0.948 - 0.06 * malicious_share)
    elif threat == "validator_collusion":
        if defense_profile == "guarded":
            if malicious_share >= config.collusion_threshold:
                attack_success_rate = _bounded_probability(0.55 + 0.9 * (malicious_share - config.collusion_threshold))
            else:
                attack_success_rate = _bounded_probability(0.04 + 0.10 * malicious_share)
        else:
            if malicious_share >= config.collusion_threshold * 0.9:
                attack_success_rate = _bounded_probability(0.72 + 0.7 * (malicious_share - config.collusion_threshold * 0.9))
            else:
                attack_success_rate = _bounded_probability(0.10 + 0.22 * malicious_share)
        approval_rate = _bounded_probability(0.93 - 0.12 * malicious_share)
    else:
        raise ValueError(f"Unsupported threat: {threat}")

    attack_success_rate = _bounded_probability(attack_success_rate + rng.normal(0.0, 0.01))
    approval_rate = _bounded_probability(approval_rate + rng.normal(0.0, 0.01))
    return {
        "threat": threat,
        "defense_profile": defense_profile,
        "approval_rate": approval_rate,
        "attack_success_rate": attack_success_rate,
        "malicious_validator_share": malicious_share,
    }


def run_governance_threat_benchmark(config: GovernanceConfig, trials_per_threat: int = 60):
    rng = np.random.default_rng(config.seed)
    rows = []
    for threat in THREATS:
        for defense_profile in DEFENSE_PROFILES:
            for trial_idx in range(trials_per_threat):
                row = _simulate_single_trial(threat, defense_profile, config, rng)
                row["trial_idx"] = trial_idx
                rows.append(row)

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(["threat", "defense_profile"])[
            ["approval_rate", "attack_success_rate", "malicious_validator_share"]
        ]
        .mean()
        .reset_index()
    )
    return raw_df, summary_df


if __name__ == "__main__":
    raw, summary = run_governance_threat_benchmark(GovernanceConfig(), trials_per_threat=24)
    print(summary.to_string(index=False))
