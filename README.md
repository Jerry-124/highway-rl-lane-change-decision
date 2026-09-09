# Highway Lane-Change Decision Making with Maskable PPO

[![Version](https://img.shields.io/badge/version-v1.0.1-blueviolet)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#quick-start)
[![Tests](https://img.shields.io/badge/tests-21-brightgreen)](#verification)

A reproducible simulation research project for high-level highway lane-change decision making with Maskable PPO, deterministic longitudinal control, action masking, and explicit execution constraints in `highway-env`.

> This repository evaluates a hybrid decision system in simulation. It is not a real-vehicle safety system and does not model production perception, actuation, hardware latency, or road risk.

## Highlights

- **Lateral-only learned policy:** Maskable PPO selects `LANE_LEFT`, `KEEP_LANE`, or `LANE_RIGHT`.
- **Deterministic longitudinal control:** speed control is separated from the reinforcement-learning action space.
- **Action masking and safety constraints:** unavailable or unsafe lane changes are excluded before policy sampling and guarded again at execution time.
- **Commitment constraint:** the lane-change cooldown matches the overtake window, preventing a new manoeuvre from silently replacing an active overtake attempt.
- **Held-out evaluation:** the bundled v1.0.0 checkpoint is evaluated on independent 100-episode validation and test splits.
- **Traceable artifacts:** raw episode records, summaries, a model card, and a SHA-256 checksum are committed with the release model.

## System Architecture

| Layer | Responsibility | Learned |
|---|---|:---:|
| Maskable PPO | Select left, keep lane, or right | Yes |
| Longitudinal controller | Cruise, prepare, and safe-following speed | No |
| Action mask / safety shield | Reject unsafe or unavailable lane changes | No |
| Commitment constraint | Prevent a new manoeuvre during the overtake window | No |

The observation contains 53 values: 10 nearby vehicles × 5 normalized kinematic features, followed by 3 action-availability flags.

The three-action lateral policy keeps high-level lane selection separate from longitudinal speed control. Earlier joint speed-and-lane experiments are not part of the stable public baseline.

## Key Results

The bundled `ppo_highway_v1.0.0.zip` policy was evaluated deterministically on two independent held-out sets of 100 episodes. Neither split was used for training. The v1.0.1 software-maintenance release does not change the model weights or these benchmark artifacts.

| Metric | Validation seeds 5000–5099 | Test seeds 9000–9099 | Target |
|---|---:|---:|---:|
| Collision rate | **1.0%** | **2.0%** | ≤5% |
| Episode completion | **99%** | **98%** | ≥95% |
| Average speed | **23.29 m/s** | **23.13 m/s** | 22–24 m/s |
| Lane changes / episode | **2.41** | **2.20** | 1–3 |
| Overtake success rate | 51.6% | 50.6% | Stretch: ≥70% |
| Shield intervention rate | **0.0%** | **0.0%** | ≤5% |
| Unavailable-action requests | **0.0%** | **0.0%** | 0% |
| Superseded overtake attempts | **0** | **0** | 0 |

Validation contained one lane-change contact. Test contained one rear-end by a follower and one lane-change contact; there were no ego-to-leader rear-end collisions in either held-out split.

Evidence is stored in [`results/v1.0.0_validation`](results/v1.0.0_validation), [`results/v1.0.0_test`](results/v1.0.0_test), and the [`v1.0.0 model card`](results/v1.0.0/MODEL_CARD.md).

## Core Configuration

### Environment and Controller

| Parameter | Value |
|---|---:|
| Environment | `highway-fast-v0` |
| Lanes / traffic vehicles | 4 / 25 |
| Vehicle density | 1.0 |
| Observed vehicles | 10 |
| Cruise / prepare speed | 30 / 25 m/s |
| Maximum acceleration / deceleration | 3 / 4 m/s² |
| Safe time headway | 1.5 s |
| Lane-change cooldown / overtake window | 8 / 8 decisions |

### Reward Coefficients

| Term | Weight |
|---|---:|
| Collision | -25.0 |
| High speed | +1.1 |
| Right lane | +0.05 |
| Completed overtake | +2.0 × congestion |
| Blocked keep-lane decision | -1.0 |
| Unsafe/unavailable request | -0.5 |
| Unnecessary lane change | -0.5 |
| Short headway | -0.5 |
| Alive reward | 0.0 |

### Maskable PPO

| Parameter | Value |
|---|---:|
| Network | MLP `[256, 256]` |
| `gamma` / `gae_lambda` | 0.99 / 0.95 |
| Learning rate | 1e-4 for final masked fine-tuning |
| Entropy coefficient | 0.01 |
| PPO epochs / clip range | 10 / 0.2 |
| `n_steps` / batch size | 256 / 256 |

The executable configuration source is [`src/highway_rl/config.py`](src/highway_rl/config.py).

## Quick Start

Python 3.10 or newer is required.

```bash
git clone https://github.com/Jerry-124/highway-rl-lane-change-decision.git
cd highway-rl-lane-change-decision
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Evaluate the Bundled Release Model

```powershell
python -m highway_rl.evaluate --algorithm maskable-ppo `
  --model-path models/ppo_highway_v1.0.0.zip `
  --split validation `
  --output-dir results/reproduced_validation

python -m highway_rl.evaluate --algorithm maskable-ppo `
  --model-path models/ppo_highway_v1.0.0.zip `
  --split test `
  --output-dir results/reproduced_test
```

The default configuration contains the v1.0.0 cooldown and overtake-window values, so no command-line override is required.

### Train a Fresh Masked PPO Policy

```powershell
python -m highway_rl.train --mask-actions `
  --total-timesteps 10000 `
  --n-envs 4 `
  --n-steps 256 `
  --batch-size 256 `
  --learning-rate 1e-4 `
  --ent-coef 0.01 `
  --gamma 0.99 `
  --vec-env subproc `
  --seed 42 `
  --model-path models/ppo_highway_custom `
  --log-dir logs/ppo_highway_custom
```

The bundled v1.0.0 checkpoint was warm-started from an earlier lateral PPO policy and then fine-tuned with action masking. A fresh training run is not expected to be bit-identical and should be evaluated independently before its metrics are reported.

## Verification

```bash
python -m pytest -q
```

The repository contains 21 pytest tests covering environment registration, observation/action dimensions, longitudinal-control behavior, reward timing, action masks, overtake accounting, lane-change cooldown behavior, evaluation provenance, and result serialization.

## Reproducibility

The stable evaluation artifact is:

```text
models/ppo_highway_v1.0.0.zip
SHA-256: 38969cd8dc3343d7be26751b9da4fb676f0af4d29031c6401ad83938663dcda8
```

The checksum is recorded in [`results/v1.0.0/SHA256SUMS.txt`](results/v1.0.0/SHA256SUMS.txt). Development seeds 3000–3019 were used during iteration and are not reported as final evidence; the final validation and test splits use seeds 5000–5099 and 9000–9099, respectively.

Version v1.0.1 is a software-maintenance baseline built on the same evaluated v1.0.0 model and result artifacts. It does not retroactively modify the historical v1.0.0 tag or benchmark values.

## Repository Structure

```text
highway-rl-lane-change-decision/
├── models/ppo_highway_v1.0.0.zip
├── results/
│   ├── v1.0.0/MODEL_CARD.md
│   ├── v1.0.0_validation/
│   └── v1.0.0_test/
├── scripts/
│   ├── crash_audit.py
│   └── diagnose.py
├── src/highway_rl/
│   ├── config.py
│   ├── environment.py
│   ├── evaluate.py
│   └── train.py
└── tests/
    ├── test_environment.py
    └── test_evaluate.py
```

## Scope and Limitations

- The 50–52% strict overtake success rate remains below the 70% stretch target.
- Safety metrics describe the complete hybrid system, not Maskable PPO in isolation.
- The evaluation covers `highway-env` simulation rather than perception errors, high-fidelity vehicle dynamics, hardware latency, or real-road validation.
- The reported percentages are estimates from 100 episodes per held-out split and are not real-world safety guarantees.
- The bundled checkpoint was warm-started; fresh training runs can produce different policies and metrics.

## License

This project is licensed under the [MIT License](LICENSE).
