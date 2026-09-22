import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass
class CTSimulatorConfig:
    n_sites: int = 6
    participants_per_site: int = 40
    horizon: int = 30
    initial_budget: float = 8000.0
    base_reward: float = 10.0
    reward_levels: tuple = (0.0, 4.0, 8.0, 12.0, 16.0)
    dropout_threshold: int = 3
    fatigue_increment: float = 0.08
    fatigue_decay: float = 0.03
    reward_saturation: float = 18.0
    budget_pressure: float = 0.35
    base_attendance_bias: float = -0.25
    site_engagement_std: float = 0.35
    behavior_noise_std: float = 0.65
    missed_visit_penalty: float = 0.08
    dropout_penalty: float = 1.75
    incentive_scale: float = 0.14
    adherence_weight: float = 3.0
    cost_weight: float = 0.015
    fairness_weight: float = 0.35
    drift_weight: float = 0.2
    communication_base_delay_ms: float = 120.0
    communication_delay_std_ms: float = 32.0
    communication_jitter_ms: float = 18.0
    communication_workload_scale_ms: float = 90.0
    packet_loss_rate: float = 0.03
    retransmission_delay_ms: float = 55.0
    delay_weight: float = 0.45
    delay_sensitivity: float = 0.18
    temporal_drift_scale: float = 0.22
    temporal_drift_period: int = 8
    seed: int = 7


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class ClinicalTrialSimulator:
    """
    Stochastic multi-site simulator for adherence-aware incentive learning.

    The agent chooses one global incentive level per time step. Participant
    attendance depends on local site effects, incentive saturation, fatigue,
    missed-visit history, and residual stochasticity.
    """

    def __init__(self, config: CTSimulatorConfig):
        self.config = config
        self.reward_levels = np.asarray(config.reward_levels, dtype=np.float32)
        self.n_actions = len(self.reward_levels)
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def _init_population(self):
        cfg = self.config
        self.site_effects = self.rng.normal(0.0, cfg.site_engagement_std, size=cfg.n_sites)
        self.site_workload = self.rng.uniform(0.35, 0.95, size=cfg.n_sites)
        self.site_network_baseline = np.clip(
            self.rng.normal(cfg.communication_base_delay_ms, cfg.communication_delay_std_ms, size=cfg.n_sites),
            35.0,
            320.0,
        )
        self.site_delay_sensitivity = self.rng.uniform(0.8, 1.25, size=cfg.n_sites)
        self.participant_site = np.repeat(np.arange(cfg.n_sites), cfg.participants_per_site)
        n_participants = len(self.participant_site)
        self.participant_resilience = self.rng.uniform(-0.4, 0.5, size=n_participants)
        self.participant_fatigue = self.rng.uniform(0.0, 0.2, size=n_participants)
        self.participant_consecutive_misses = np.zeros(n_participants, dtype=np.int32)
        self.active = np.ones(n_participants, dtype=bool)
        self.attended_total = np.zeros(n_participants, dtype=np.int32)
        self.missed_total = np.zeros(n_participants, dtype=np.int32)

    def _sample_site_conditions(self):
        cfg = self.config
        phase = 2.0 * math.pi * self.t / max(1, cfg.temporal_drift_period)
        global_shock = cfg.temporal_drift_scale * math.sin(phase)
        local_shocks = self.rng.normal(0.0, cfg.temporal_drift_scale / 2.0, size=cfg.n_sites)
        temporal_shock = global_shock + local_shocks

        self.current_site_workload = np.clip(self.site_workload + temporal_shock, 0.15, 1.45)
        packet_loss = (
            cfg.packet_loss_rate
            + 0.055 * self.current_site_workload
            + 0.03 * np.abs(temporal_shock)
            + self.rng.normal(0.0, 0.008, size=cfg.n_sites)
        )
        packet_loss = np.clip(packet_loss, 0.0, 0.35)
        retransmissions = self.rng.binomial(3, packet_loss)
        jitter = self.rng.normal(0.0, cfg.communication_jitter_ms, size=cfg.n_sites)
        site_delay_ms = (
            self.site_network_baseline
            + cfg.communication_workload_scale_ms * self.current_site_workload * self.site_delay_sensitivity
            + jitter
            + retransmissions * cfg.retransmission_delay_ms
        )

        self.current_packet_loss = packet_loss.astype(np.float32)
        self.current_temporal_shock = temporal_shock.astype(np.float32)
        self.current_site_delay_ms = np.clip(site_delay_ms, 25.0, 1200.0).astype(np.float32)

    def _current_state(self):
        active_ratio = float(np.mean(self.active))
        total_events = self.total_attended + self.total_missed
        adherence = self.total_attended / max(1, total_events)
        missed_rate = self.total_missed / max(1, total_events)
        dropout_rate = self.total_dropouts / max(1, self.n_participants)
        fairness = 1.0 - float(np.std(self.site_incentives) / (np.mean(self.site_incentives) + 1e-6))
        site_balance = 1.0 - float(np.std(self.site_attendance_rate()) + 1e-6)
        budget_frac = self.remaining_budget / max(1.0, self.config.initial_budget)
        mean_fatigue = float(np.mean(self.participant_fatigue[self.active])) if np.any(self.active) else 1.0
        workload = float(np.mean(self.site_workload))
        drift = float(np.std(self.site_attendance_rate()))
        delay_norm = self.last_mean_delay_ms / max(self.config.communication_base_delay_ms, 1e-6)
        p95_delay_norm = self.last_p95_delay_ms / max(self.config.communication_base_delay_ms, 1e-6)
        packet_loss = self.last_packet_loss_rate

        return np.array(
            [
                adherence,
                missed_rate,
                dropout_rate,
                budget_frac,
                active_ratio,
                mean_fatigue,
                workload,
                fairness,
                site_balance,
                drift,
                delay_norm,
                p95_delay_norm,
                packet_loss,
                self.t / max(1, self.config.horizon),
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_population()
        self.n_participants = len(self.participant_site)
        self.t = 0
        self.remaining_budget = float(self.config.initial_budget)
        self.total_attended = 0
        self.total_missed = 0
        self.total_dropouts = 0
        self.site_incentives = np.zeros(self.config.n_sites, dtype=np.float32)
        self.site_attended = np.zeros(self.config.n_sites, dtype=np.int32)
        self.site_seen = np.zeros(self.config.n_sites, dtype=np.int32)
        self.last_adherence = 0.0
        self.last_reward = 0.0
        self.last_action = 0
        self.current_site_workload = self.site_workload.copy()
        self.current_packet_loss = np.full(self.config.n_sites, self.config.packet_loss_rate, dtype=np.float32)
        self.current_temporal_shock = np.zeros(self.config.n_sites, dtype=np.float32)
        self.current_site_delay_ms = self.site_network_baseline.astype(np.float32)
        self.last_mean_delay_ms = float(np.mean(self.current_site_delay_ms))
        self.last_p95_delay_ms = float(np.percentile(self.current_site_delay_ms, 95))
        self.last_packet_loss_rate = float(np.mean(self.current_packet_loss))
        self.episode_history = []
        return self._current_state()

    def site_attendance_rate(self):
        return self.site_attended / np.maximum(1, self.site_seen)

    def _attendance_probability(self, participant_idx: int, reward_amount: float):
        cfg = self.config
        site_idx = self.participant_site[participant_idx]
        fatigue = self.participant_fatigue[participant_idx]
        misses = self.participant_consecutive_misses[participant_idx]
        site_effect = self.site_effects[site_idx]
        resilience = self.participant_resilience[participant_idx]
        workload_drag = cfg.drift_weight * self.current_site_workload[site_idx]
        incentive_gain = cfg.incentive_scale * reward_amount * (
            1.0 - min(0.85, reward_amount / max(cfg.reward_saturation, 1e-6))
        )
        budget_drag = cfg.budget_pressure * (1.0 - self.remaining_budget / max(cfg.initial_budget, 1e-6))
        delay_norm = self.current_site_delay_ms[site_idx] / max(cfg.communication_base_delay_ms, 1e-6)
        delay_drag = cfg.delay_sensitivity * max(0.0, delay_norm - 1.0) + 0.45 * self.current_packet_loss[site_idx]
        noise = float(self.rng.normal(0.0, cfg.behavior_noise_std))
        logit = (
            cfg.base_attendance_bias
            + site_effect
            + resilience
            + incentive_gain
            - fatigue
            - cfg.missed_visit_penalty * misses
            - workload_drag
            - budget_drag
            - delay_drag
            + self.current_temporal_shock[site_idx]
            + noise
        )
        return float(np.clip(sigmoid(logit), 0.02, 0.98))

    def step(self, action_idx: int):
        cfg = self.config
        self._sample_site_conditions()
        reward_amount = float(self.reward_levels[action_idx])
        attended_step = 0
        missed_step = 0
        dropouts_step = 0
        site_rewards = np.zeros(cfg.n_sites, dtype=np.float32)

        for participant_idx in np.where(self.active)[0]:
            site_idx = self.participant_site[participant_idx]
            p_attend = self._attendance_probability(participant_idx, reward_amount)
            attends = self.rng.uniform() < p_attend
            self.site_seen[site_idx] += 1

            if attends:
                attended_step += 1
                self.total_attended += 1
                self.attended_total[participant_idx] += 1
                self.participant_consecutive_misses[participant_idx] = 0
                self.participant_fatigue[participant_idx] = max(
                    0.0,
                    self.participant_fatigue[participant_idx] - cfg.fatigue_decay,
                )
                self.remaining_budget -= reward_amount + cfg.base_reward
                site_rewards[site_idx] += reward_amount + cfg.base_reward
                self.site_attended[site_idx] += 1
            else:
                missed_step += 1
                self.total_missed += 1
                self.missed_total[participant_idx] += 1
                self.participant_consecutive_misses[participant_idx] += 1
                self.participant_fatigue[participant_idx] += cfg.fatigue_increment
                if self.participant_consecutive_misses[participant_idx] >= cfg.dropout_threshold:
                    self.active[participant_idx] = False
                    self.total_dropouts += 1
                    dropouts_step += 1

        total_events = max(1, attended_step + missed_step)
        adherence = attended_step / total_events
        fairness = 1.0 - float(np.std(site_rewards) / (np.mean(site_rewards) + 1e-6)) if np.mean(site_rewards) > 0 else 1.0
        attendance_drift = float(np.std(self.site_attendance_rate()))
        step_cost = float(np.sum(site_rewards))
        self.site_incentives += site_rewards
        mean_delay_ms = float(np.mean(self.current_site_delay_ms))
        p95_delay_ms = float(np.percentile(self.current_site_delay_ms, 95))
        packet_loss_rate = float(np.mean(self.current_packet_loss))
        network_instability = float(np.std(self.current_site_delay_ms) / (mean_delay_ms + 1e-6))
        delay_penalty = max(0.0, mean_delay_ms / max(cfg.communication_base_delay_ms, 1e-6) - 1.0) + packet_loss_rate + 0.25 * network_instability

        reward = (
            cfg.adherence_weight * (adherence - self.last_adherence)
            + cfg.adherence_weight * adherence
            - cfg.cost_weight * step_cost
            - cfg.dropout_penalty * dropouts_step
            + cfg.fairness_weight * fairness
            - cfg.drift_weight * attendance_drift
            - cfg.delay_weight * delay_penalty
        )

        self.last_adherence = adherence
        self.last_reward = reward
        self.last_action = action_idx
        self.last_mean_delay_ms = mean_delay_ms
        self.last_p95_delay_ms = p95_delay_ms
        self.last_packet_loss_rate = packet_loss_rate
        self.t += 1

        info = {
            "step": self.t,
            "attendance_rate": adherence,
            "attended": attended_step,
            "missed": missed_step,
            "dropouts": dropouts_step,
            "remaining_budget": self.remaining_budget,
            "step_cost": step_cost,
            "fairness": fairness,
            "attendance_drift": attendance_drift,
            "active_ratio": float(np.mean(self.active)),
            "mean_fatigue": float(np.mean(self.participant_fatigue[self.active])) if np.any(self.active) else 1.0,
            "reward_amount": reward_amount,
            "action_idx": int(action_idx),
            "episode_reward": reward,
            "mean_delay_ms": mean_delay_ms,
            "p95_delay_ms": p95_delay_ms,
            "packet_loss_rate": packet_loss_rate,
            "network_instability": network_instability,
            "delay_penalty": delay_penalty,
        }
        self.episode_history.append(info)
        done = self.t >= cfg.horizon or np.sum(self.active) == 0 or self.remaining_budget <= 0.0
        return self._current_state(), float(reward), done, info


def heuristic_policy(state: np.ndarray, reward_levels: np.ndarray):
    adherence = state[0]
    dropout_rate = state[2]
    budget_frac = state[3]
    fatigue = state[5]
    if budget_frac < 0.2:
        return 0
    if dropout_rate > 0.18 or adherence < 0.45:
        return len(reward_levels) - 1
    if fatigue > 0.45:
        return max(1, len(reward_levels) - 2)
    if adherence > 0.8:
        return 1
    return len(reward_levels) // 2


def evaluate_policy(env: ClinicalTrialSimulator, policy_fn, episodes: int = 10, seed: int = 7):
    episode_rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=seed + episode_idx)
        done = False
        total_reward = 0.0
        final_info = None
        while not done:
            action = policy_fn(state, env.reward_levels)
            state, reward, done, info = env.step(action)
            total_reward += reward
            final_info = info
        episode_rows.append(
            {
                "episode": episode_idx,
                "total_reward": total_reward,
                "final_attendance_rate": final_info["attendance_rate"],
                "dropouts": final_info["dropouts"],
                "remaining_budget": final_info["remaining_budget"],
                "fairness": final_info["fairness"],
                "active_ratio": final_info["active_ratio"],
                "mean_delay_ms": final_info["mean_delay_ms"],
                "p95_delay_ms": final_info["p95_delay_ms"],
                "packet_loss_rate": final_info["packet_loss_rate"],
            }
        )
    return pd.DataFrame(episode_rows)


def generate_synthetic_ct_dataset(
    n_sites: int = 8,
    participants_per_site: int = 120,
    visits_per_participant: int = 12,
    random_state: int = 7,
):
    """
    Generates a larger, heterogeneous classification dataset with site and
    temporal structure so FL experiments are not limited to Pima.
    """
    set_seed(random_state)
    n_samples = n_sites * participants_per_site * visits_per_participant
    X, y = make_classification(
        n_samples=n_samples,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.58, 0.42],
        class_sep=1.1,
        flip_y=0.04,
        random_state=random_state,
    )

    site_ids = np.repeat(np.arange(n_sites), participants_per_site * visits_per_participant)
    participant_ids = np.repeat(np.arange(n_sites * participants_per_site), visits_per_participant)
    visit_ids = np.tile(np.arange(visits_per_participant), n_sites * participants_per_site)

    site_bias = np.linspace(-0.8, 0.8, n_sites)
    site_delay_ms = np.clip(np.linspace(70.0, 220.0, n_sites) + np.random.normal(0.0, 12.0, size=n_sites), 40.0, 350.0)
    site_workload = np.clip(np.linspace(0.3, 1.0, n_sites) + np.random.normal(0.0, 0.05, size=n_sites), 0.15, 1.2)
    fatigue_feature = visit_ids / max(1, visits_per_participant - 1)
    engagement_feature = site_bias[site_ids] + np.random.normal(0.0, 0.15, size=n_samples)
    comm_delay_feature = site_delay_ms[site_ids] + np.random.normal(0.0, 14.0, size=n_samples)
    packet_loss_feature = np.clip(0.01 + 0.06 * site_workload[site_ids] + np.random.normal(0.0, 0.01, size=n_samples), 0.0, 0.30)
    dropout_risk = 1.0 / (
        1.0
        + np.exp(
            -(
                0.8 * fatigue_feature
                + 0.9 * packet_loss_feature
                + 0.004 * (comm_delay_feature - np.mean(comm_delay_feature))
                - 0.35 * engagement_feature
                + np.random.normal(0.0, 0.25, size=n_samples)
            )
        )
    )

    logits = (
        0.9 * y
        + 0.5 * engagement_feature
        - 0.25 * fatigue_feature
        - 0.10 * (comm_delay_feature / 100.0)
        - 0.65 * dropout_risk
        - 0.35 * packet_loss_feature
        + np.random.normal(0.0, 0.35, size=n_samples)
    )
    observed_outcome = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int32)

    columns = [f"feature_{idx}" for idx in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["site_id"] = site_ids
    df["participant_id"] = participant_ids
    df["visit_id"] = visit_ids
    df["fatigue_index"] = fatigue_feature
    df["engagement_index"] = engagement_feature
    df["comm_delay_ms"] = comm_delay_feature
    df["packet_loss_rate"] = packet_loss_feature
    df["site_workload_index"] = site_workload[site_ids]
    df["dropout_risk"] = dropout_risk
    df["Outcome"] = observed_outcome
    return df


def save_synthetic_ct_dataset(output_path: Path, **kwargs):
    df = generate_synthetic_ct_dataset(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
