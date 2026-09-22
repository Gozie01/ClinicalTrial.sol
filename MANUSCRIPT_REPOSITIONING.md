# TRUST-CT Revision Direction

## Stronger Positioning

Reframe TRUST-CT as an `adaptive socio-technical governance` framework for decentralized clinical trials rather than a simple integration of federated learning, reinforcement learning, and blockchain.

Core message:

- Clinical trials are modeled as a coupled system of `participant behavior`, `distributed learning`, and `auditable governance`.
- Incentive policies influence attendance and dropout.
- Attendance dynamics influence data volume, class balance, and site heterogeneity.
- These changes propagate into federated convergence and predictive reliability.
- Governance infrastructure records and enforces the resulting adaptive decisions.

## What The New Repo Additions Support

- `ct_simulation.py`
  - Stochastic multi-site participant simulator with fatigue, dropout, site variability, and budget pressure.
- `rl_ct_experiments.py`
  - RL ablations across `static`, `heuristic`, `tabular_q`, `dqn`, and `ppo`.
  - Learning curves and policy-performance comparisons.
- `federated_mlp_run.py`
  - Larger dataset options beyond Pima.
  - DP-style clipping and Gaussian noise during local training.
  - Malicious-client simulations and robust aggregation.
- `scalability_analysis.py`
  - Multi-site runtime, communication, and utility sweeps for scale-oriented evaluation.

## Suggested Updated Contributions

1. A socio-technical formulation of decentralized clinical trial management in which incentive adaptation, participant adherence, and federated utility are explicitly coupled.
2. A realistic multi-site trial simulator for evaluating incentive-learning strategies under dropout, fatigue, and budget constraints.
3. An adversarially evaluated federated pipeline with DP-style local privacy controls, malicious-client stress tests, and robust aggregation baselines.
4. A broader experimental evaluation spanning RL ablations, larger datasets, and scaling behavior.

## Claims To Avoid Unless Fully Supported

- Formal privacy guarantees without a validated accountant and clearly reported assumptions.
- Strong end-to-end security claims without blockchain-layer adversarial evaluation.
- Real-world deployment readiness unless the full stack is tested beyond simulation.

## Claims The Current Upgrade Can Defend Better

- `privacy-aware` or `reduced raw-data exposure`, strengthened by DP-style local update perturbation.
- `behavior-aware adaptive incentive optimization`, supported by RL ablations and learning curves.
- `malicious-client robustness analysis`, supported by poisoning and Byzantine-style simulations.
- `scaling trends`, supported by controlled multi-site sweeps on larger synthetic clinical datasets.
