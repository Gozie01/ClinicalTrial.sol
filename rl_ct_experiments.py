import argparse
import json
import os
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from ct_simulation import (
    CTSimulatorConfig,
    ClinicalTrialSimulator,
    evaluate_policy,
    heuristic_policy,
)


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "rl_ablations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 7):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def static_policy_factory(action_idx: int):
    def _policy(_state, _reward_levels):
        return action_idx

    return _policy


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QTableAgent:
    def __init__(
        self,
        n_actions: int,
        bins_per_dim: int = 8,
        learning_rate: float = 0.2,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.994,
    ):
        self.n_actions = n_actions
        self.bins_per_dim = bins_per_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = {}

    def _discretize(self, state: np.ndarray):
        clipped = np.clip(state, 0.0, 1.0)
        bins = np.floor(clipped * (self.bins_per_dim - 1)).astype(int)
        return tuple(bins.tolist())

    def _values(self, key):
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[key]

    def act(self, state: np.ndarray):
        key = self._discretize(state)
        if np.random.uniform() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self._values(key)))

    def greedy_action(self, state: np.ndarray):
        return int(np.argmax(self._values(self._discretize(state))))

    def update(self, state, action, reward, next_state, done):
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        q_values = self._values(key)
        next_values = self._values(next_key)
        target = reward + (0.0 if done else self.gamma * np.max(next_values))
        q_values[action] += self.learning_rate * (target - q_values[action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 7e-4,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.996,
        batch_size: int = 96,
        target_update_freq: int = 15,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer = ReplayBuffer(capacity=12000)
        self.online_model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.target_model.set_weights(self.online_model.get_weights())
        self.train_steps = 0

    def _build_model(self, learning_rate):
        model = models.Sequential(
            [
                layers.Input(shape=(self.state_dim,)),
                layers.Dense(96, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.n_actions),
            ]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        return model

    def act(self, state: np.ndarray):
        if np.random.uniform() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        q_values = self.online_model.predict(state[None, :], verbose=0)[0]
        return int(np.argmax(q_values))

    def greedy_action(self, state: np.ndarray):
        q_values = self.online_model.predict(state[None, :], verbose=0)[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        next_q_online = self.online_model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + (1.0 - dones) * self.gamma * next_q_target[np.arange(len(next_actions)), next_actions]

        q_values = self.online_model.predict(states, verbose=0)
        q_values[np.arange(len(actions)), actions] = target_q
        loss = self.online_model.train_on_batch(states, q_values)

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_model.set_weights(self.online_model.get_weights())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        actor_lr: float = 2e-4,
        critic_lr: float = 5e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.15,
        train_epochs: int = 8,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        self.actor = models.Sequential(
            [
                layers.Input(shape=(state_dim,)),
                layers.Dense(96, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(n_actions, activation="softmax"),
            ]
        )
        self.critic = models.Sequential(
            [
                layers.Input(shape=(state_dim,)),
                layers.Dense(96, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )
        self.actor_optimizer = optimizers.Adam(actor_lr)
        self.critic_optimizer = optimizers.Adam(critic_lr)

    def act(self, state: np.ndarray):
        probs = self.actor.predict(state[None, :], verbose=0)[0]
        probs = np.clip(probs, 1e-6, 1.0)
        probs = probs / np.sum(probs)
        action = int(np.random.choice(self.n_actions, p=probs))
        log_prob = float(np.log(probs[action]))
        return action, log_prob

    def greedy_action(self, state: np.ndarray):
        probs = self.actor.predict(state[None, :], verbose=0)[0]
        return int(np.argmax(probs))

    def train_rollout(self, states, actions, rewards, dones, log_probs):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        old_log_probs = np.asarray(log_probs, dtype=np.float32)

        returns = []
        discounted = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            discounted = reward + self.gamma * discounted * (1.0 - done)
            returns.append(discounted)
        returns = np.asarray(list(reversed(returns)), dtype=np.float32)
        values = self.critic.predict(states, verbose=0).reshape(-1)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        critic_losses = []

        for _ in range(self.train_epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                probs = self.actor(states, training=True)
                action_mask = tf.one_hot(actions, depth=self.n_actions)
                selected_probs = tf.reduce_sum(probs * action_mask, axis=1)
                new_log_probs = tf.math.log(selected_probs + 1e-8)
                ratios = tf.exp(new_log_probs - old_log_probs)
                unclipped = ratios * advantages
                clipped = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                value_preds = tf.squeeze(self.critic(states, training=True), axis=1)
                critic_loss = tf.reduce_mean(tf.square(returns - value_preds))

            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            actor_losses.append(float(actor_loss))
            critic_losses.append(float(critic_loss))

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))


def train_static(env, episodes: int, action_idx: int):
    rows = []
    policy = static_policy_factory(action_idx)
    for episode_idx in range(episodes):
        state = env.reset(seed=env.config.seed + episode_idx)
        done = False
        total_reward = 0.0
        step_rewards = []
        final_info = None
        while not done:
            action = policy(state, env.reward_levels)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_rewards.append(reward)
            final_info = info
        rows.append(
            {
                "method": "static",
                "episode": episode_idx,
                "total_reward": total_reward,
                "mean_step_reward": float(np.mean(step_rewards)),
                "attendance_rate": final_info["attendance_rate"],
                "dropouts": final_info["dropouts"],
                "remaining_budget": final_info["remaining_budget"],
                "fairness": final_info["fairness"],
                "active_ratio": final_info["active_ratio"],
                "mean_delay_ms": final_info["mean_delay_ms"],
                "p95_delay_ms": final_info["p95_delay_ms"],
                "packet_loss_rate": final_info["packet_loss_rate"],
                "policy_entropy": 0.0,
            }
        )
    return pd.DataFrame(rows)


def train_heuristic(env, episodes: int):
    rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=env.config.seed + episode_idx)
        done = False
        total_reward = 0.0
        actions = []
        final_info = None
        while not done:
            action = heuristic_policy(state, env.reward_levels)
            actions.append(action)
            state, reward, done, info = env.step(action)
            total_reward += reward
            final_info = info
        action_counts = np.bincount(actions, minlength=env.n_actions)
        probs = action_counts / max(1, np.sum(action_counts))
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-8), 0.0))
        rows.append(
            {
                "method": "heuristic",
                "episode": episode_idx,
                "total_reward": total_reward,
                "mean_step_reward": total_reward / max(1, len(actions)),
                "attendance_rate": final_info["attendance_rate"],
                "dropouts": final_info["dropouts"],
                "remaining_budget": final_info["remaining_budget"],
                "fairness": final_info["fairness"],
                "active_ratio": final_info["active_ratio"],
                "mean_delay_ms": final_info["mean_delay_ms"],
                "p95_delay_ms": final_info["p95_delay_ms"],
                "packet_loss_rate": final_info["packet_loss_rate"],
                "policy_entropy": float(entropy),
            }
        )
    return pd.DataFrame(rows)


def train_q_learning(env, episodes: int):
    agent = QTableAgent(n_actions=env.n_actions)
    rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=env.config.seed + episode_idx)
        done = False
        total_reward = 0.0
        actions = []
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            actions.append(action)
        rows.append(
            {
                "method": "tabular_q",
                "episode": episode_idx,
                "total_reward": total_reward,
                "mean_step_reward": total_reward / max(1, len(actions)),
                "attendance_rate": info["attendance_rate"],
                "dropouts": info["dropouts"],
                "remaining_budget": info["remaining_budget"],
                "fairness": info["fairness"],
                "active_ratio": info["active_ratio"],
                "mean_delay_ms": info["mean_delay_ms"],
                "p95_delay_ms": info["p95_delay_ms"],
                "packet_loss_rate": info["packet_loss_rate"],
                "epsilon": agent.epsilon,
                "policy_entropy": float(pd.Series(actions).value_counts(normalize=True).pipe(lambda s: -(s * np.log(s + 1e-8)).sum())),
            }
        )
    return agent, pd.DataFrame(rows)


def train_dqn(env, episodes: int):
    state_dim = len(env.reset())
    agent = DQNAgent(state_dim=state_dim, n_actions=env.n_actions)
    rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=env.config.seed + episode_idx)
        done = False
        total_reward = 0.0
        losses = []
        actions = []
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
            actions.append(action)
        rows.append(
            {
                "method": "dqn",
                "episode": episode_idx,
                "total_reward": total_reward,
                "mean_step_reward": total_reward / max(1, len(actions)),
                "attendance_rate": info["attendance_rate"],
                "dropouts": info["dropouts"],
                "remaining_budget": info["remaining_budget"],
                "fairness": info["fairness"],
                "active_ratio": info["active_ratio"],
                "mean_delay_ms": info["mean_delay_ms"],
                "p95_delay_ms": info["p95_delay_ms"],
                "packet_loss_rate": info["packet_loss_rate"],
                "epsilon": agent.epsilon,
                "mean_loss": float(np.mean(losses)) if losses else np.nan,
                "policy_entropy": float(pd.Series(actions).value_counts(normalize=True).pipe(lambda s: -(s * np.log(s + 1e-8)).sum())),
            }
        )
    return agent, pd.DataFrame(rows)


def train_ppo(env, episodes: int):
    state_dim = len(env.reset())
    agent = PPOAgent(state_dim=state_dim, n_actions=env.n_actions)
    rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=env.config.seed + episode_idx)
        done = False
        total_reward = 0.0
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        last_info = None

        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob)
            state = next_state
            total_reward += reward
            last_info = info

        actor_loss, critic_loss = agent.train_rollout(states, actions, rewards, dones, log_probs)
        rows.append(
            {
                "method": "ppo",
                "episode": episode_idx,
                "total_reward": total_reward,
                "mean_step_reward": total_reward / max(1, len(actions)),
                "attendance_rate": last_info["attendance_rate"],
                "dropouts": last_info["dropouts"],
                "remaining_budget": last_info["remaining_budget"],
                "fairness": last_info["fairness"],
                "active_ratio": last_info["active_ratio"],
                "mean_delay_ms": last_info["mean_delay_ms"],
                "p95_delay_ms": last_info["p95_delay_ms"],
                "packet_loss_rate": last_info["packet_loss_rate"],
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "policy_entropy": float(pd.Series(actions).value_counts(normalize=True).pipe(lambda s: -(s * np.log(s + 1e-8)).sum())),
            }
        )
    return agent, pd.DataFrame(rows)


def evaluate_trained_agent(env, method_name: str, action_fn, episodes: int = 12, seed_base: int = 10_000):
    rows = []
    for episode_idx in range(episodes):
        state = env.reset(seed=seed_base + episode_idx)
        done = False
        total_reward = 0.0
        actions = []
        final_info = None
        while not done:
            action = action_fn(state)
            actions.append(action)
            state, reward, done, info = env.step(action)
            total_reward += reward
            final_info = info
        rows.append(
            {
                "method": method_name,
                "eval_episode": episode_idx,
                "total_reward": total_reward,
                "attendance_rate": final_info["attendance_rate"],
                "dropouts": final_info["dropouts"],
                "remaining_budget": final_info["remaining_budget"],
                "fairness": final_info["fairness"],
                "active_ratio": final_info["active_ratio"],
                "mean_delay_ms": final_info["mean_delay_ms"],
                "p95_delay_ms": final_info["p95_delay_ms"],
                "packet_loss_rate": final_info["packet_loss_rate"],
                "policy_entropy": float(pd.Series(actions).value_counts(normalize=True).pipe(lambda s: -(s * np.log(s + 1e-8)).sum())),
            }
        )
    return pd.DataFrame(rows)


def run_all_experiments(config: CTSimulatorConfig, episodes: int = 120, eval_episodes: int = 24):
    set_seed(config.seed)
    env = ClinicalTrialSimulator(config)

    static_df = train_static(env, episodes=episodes, action_idx=len(env.reward_levels) // 2)
    heuristic_df = train_heuristic(env, episodes=episodes)
    q_agent, q_df = train_q_learning(env, episodes=episodes)
    dqn_agent, dqn_df = train_dqn(env, episodes=episodes)
    ppo_agent, ppo_df = train_ppo(env, episodes=episodes)

    train_df = pd.concat([static_df, heuristic_df, q_df, dqn_df, ppo_df], ignore_index=True)

    eval_frames = [
        evaluate_trained_agent(env, "tabular_q", q_agent.greedy_action, episodes=eval_episodes, seed_base=10_000),
        evaluate_trained_agent(env, "dqn", dqn_agent.greedy_action, episodes=eval_episodes, seed_base=20_000),
        evaluate_trained_agent(env, "ppo", ppo_agent.greedy_action, episodes=eval_episodes, seed_base=30_000),
        evaluate_policy(env, static_policy_factory(len(env.reward_levels) // 2), episodes=eval_episodes, seed=40_000)
        .rename(columns={"final_attendance_rate": "attendance_rate"})
        .assign(method="static"),
        evaluate_policy(env, heuristic_policy, episodes=eval_episodes, seed=50_000)
        .rename(columns={"final_attendance_rate": "attendance_rate"})
        .assign(method="heuristic"),
    ]
    eval_df = pd.concat(eval_frames, ignore_index=True)

    summary = (
        eval_df.groupby("method")[
            [
                "total_reward",
                "attendance_rate",
                "dropouts",
                "remaining_budget",
                "fairness",
                "active_ratio",
                "mean_delay_ms",
                "p95_delay_ms",
                "packet_loss_rate",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    return train_df, eval_df, summary


def run_multi_seed_experiments(config: CTSimulatorConfig, seeds, episodes: int = 120, eval_episodes: int = 24):
    train_frames = []
    eval_frames = []
    summary_frames = []

    for seed in seeds:
        seed_config = CTSimulatorConfig(**{**asdict(config), "seed": int(seed)})
        train_df, eval_df, summary_df = run_all_experiments(
            config=seed_config,
            episodes=episodes,
            eval_episodes=eval_episodes,
        )
        train_frames.append(train_df.assign(seed=int(seed)))
        eval_frames.append(eval_df.assign(seed=int(seed)))
        summary_frames.append(summary_df.assign(seed=int(seed)))

    train_all = pd.concat(train_frames, ignore_index=True)
    eval_all = pd.concat(eval_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    metrics = [
        "total_reward",
        "attendance_rate",
        "dropouts",
        "remaining_budget",
        "fairness",
        "active_ratio",
        "mean_delay_ms",
        "p95_delay_ms",
        "packet_loss_rate",
    ]
    summary_multi = summary_all.groupby("method")[metrics].agg(["mean", "std"]).reset_index()
    summary_multi.columns = [
        "method" if col == ("method", "") else f"{col[0]}_{col[1]}"
        for col in summary_multi.columns.to_flat_index()
    ]
    return train_all, eval_all, summary_all, summary_multi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--participants_per_site", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--eval_episodes", type=int, default=24)
    args = parser.parse_args()

    config = CTSimulatorConfig(
        n_sites=args.sites,
        participants_per_site=args.participants_per_site,
        horizon=args.horizon,
        initial_budget=args.budget,
        seed=args.seed,
    )
    if args.seeds.strip():
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        train_df, eval_df, summary_seed_df, summary_df = run_multi_seed_experiments(
            config=config,
            seeds=seeds,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
        )
    else:
        train_df, eval_df, summary_df = run_all_experiments(
            config=config,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
        )
        summary_seed_df = None

    tag = f"sites{config.n_sites}_pps{config.participants_per_site}_hor{config.horizon}_seed{config.seed}"
    train_path = EVAL_DIR / f"rl_training_curves_{tag}.csv"
    eval_path = EVAL_DIR / f"rl_eval_{tag}.csv"
    summary_path = EVAL_DIR / f"rl_summary_{tag}.csv"
    config_path = EVAL_DIR / f"rl_config_{tag}.json"
    summary_seed_path = EVAL_DIR / f"rl_summary_by_seed_{tag}.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    if summary_seed_df is not None:
        summary_seed_df.to_csv(summary_seed_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print("\n=== RL Ablation Summary ===")
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(train_path)
    print(eval_path)
    print(summary_path)
    if summary_seed_df is not None:
        print(summary_seed_path)
    print(config_path)


if __name__ == "__main__":
    main()
