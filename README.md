# Safe RL for Autonomous Highway Lane-Change Decision Making

A simulation-based reinforcement learning project for autonomous highway behavior decision making.  
The project investigates how reinforcement learning can balance **safety, traffic efficiency, and driving comfort** when an ego vehicle needs to decide whether to keep its lane or change lanes.

> **Project status:** In development  
> Experimental results will be added after reproducible training and evaluation are completed.

## 1. Motivation

Lane-change decision making is a representative high-level decision problem in autonomous driving. A useful policy should not only improve traffic efficiency, but also avoid unsafe maneuvers and unnecessary lane changes.

This project uses a lightweight highway traffic simulator to study:

- high-level lane-change decisions
- safety-aware reward design
- reinforcement learning policy learning
- quantitative evaluation under different traffic conditions
- robustness to imperfect observations

## 2. Problem Definition

The ego vehicle observes the surrounding traffic state and selects a discrete behavior action.

### Observation

The state representation includes information such as:

- ego vehicle speed
- ego lane
- relative position of surrounding vehicles
- relative speed of nearby vehicles
- lane occupancy / local traffic context

### Action Space

```text
0 - Keep Lane
1 - Lane Change Left
2 - Lane Change Right
```

### Objective

The agent is trained to maximize a reward that combines:

```text
Driving Efficiency
+ Safety
+ Comfort
- Unnecessary / Risky Behavior
```

## 3. Methodology

### Baseline RL Policy

The initial implementation uses **Proximal Policy Optimization (PPO)** with an MLP policy.

The training pipeline is:

```text
Highway Environment
        ↓
Observation
        ↓
PPO Policy
        ↓
Discrete Lane-Change Action
        ↓
Environment Transition
        ↓
Reward
        ↓
Policy Update
```

### Reward Design

The reward function is designed to reflect three main objectives:

1. **Efficiency** — encourage useful forward progress and appropriate speed.
2. **Safety** — penalize collisions and unsafe behavior.
3. **Comfort / Stability** — discourage unnecessary or aggressive maneuvers.

The exact reward coefficients are treated as experiment parameters and will be documented with the corresponding results.

## 4. Experimental Plan

### Experiment A — PPO Baseline

Establish a reproducible PPO baseline under a fixed environment configuration.

Evaluation metrics:

- episode reward
- collision rate
- average speed
- lane-change frequency
- episode duration

### Experiment B — Reward Ablation

Compare different reward configurations, for example:

```text
Baseline Reward
      ↓
+ Safety Term
      ↓
+ Comfort / Lane-Change Term
```

The goal is to quantify the trade-off between safety and efficiency.

### Experiment C — PPO vs DQN

Compare two reinforcement learning approaches under matched scenarios and evaluation settings.

Planned metrics:

| Metric | PPO | DQN |
|---|---:|---:|
| Collision Rate | TBD | TBD |
| Average Speed | TBD | TBD |
| Episode Reward | TBD | TBD |
| Lane Changes / Episode | TBD | TBD |

> Results will only be reported after the experiments are actually run.

### Experiment D — Robustness

Evaluate policy degradation under imperfect observations:

- Gaussian observation noise
- observation delay
- temporary observation dropout

Example evaluation levels:

```text
0%
2%
5%
10%
```

The exact noise settings will be reported together with the final results.

## 5. Evaluation

All evaluation experiments are intended to use fixed seeds / documented randomization settings and a separate evaluation phase.

Planned output:

```text
results/
├── metrics/
│   └── evaluation_results.csv
├── figures/
│   ├── training_curve.png
│   ├── collision_rate.png
│   ├── average_speed.png
│   └── lane_change_frequency.png
└── videos/
```

## 6. Project Structure

```text
rl-lane-change-decision/
├── README.md
├── requirements.txt
├── configs/
├── environments/
├── agents/
├── rewards/
├── evaluation/
├── visualization/
├── experiments/
├── models/
└── results/
```

## 7. Tech Stack

- Python
- Gymnasium
- highway-env
- Stable-Baselines3
- NumPy
- Pandas
- Matplotlib

PyTorch will be included once it is used directly in the implementation or model customization.

## 8. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

### Evaluate

```bash
python evaluate.py
```

The final repository will provide the exact scripts and configuration files used to reproduce the reported experiments.

## 9. Key Questions

This project is designed to investigate:

- Does reward shaping reduce unsafe or unnecessary lane changes?
- How does PPO compare with DQN for this decision problem?
- What is the trade-off between safety and traffic efficiency?
- How robust is the learned policy to observation noise and delay?

## 10. Limitations

This project is a simplified simulation study.

It does not attempt to model the full complexity of real-world autonomous driving, including:

- perception uncertainty from real sensors
- high-fidelity vehicle dynamics
- detailed road geometry
- full trajectory planning and low-level control
- real-world traffic behavior
- real-vehicle validation

The purpose is to study the **decision-making layer** in a controlled and reproducible environment.

## 11. Future Work

- Multi-agent interaction and cooperative decision making
- More realistic behavior prediction
- Hierarchical decision + motion planning
- Integration with MPC-based control
- More challenging edge-case scenarios
- Extension toward energy-aware / eco-driving objectives

## 12. Author

**Jieru Liang**  
M.Sc. Electromobility, TU Braunschweig (expected 2027)

Research interests: Autonomous Driving, Decision Making, Reinforcement Learning, Multi-Agent Systems, Vehicle Control.
