# Highway RL Lane-Change Decision

Safe and efficient highway lane-change decision making with Maskable PPO,
Gymnasium, highway-env, and Stable-Baselines3.

**Release:** v1.0.0

**Scope:** simulation research and portfolio demonstration; not for real-vehicle
deployment.

## Overview

The learned policy makes only high-level lateral decisions:

- `LANE_LEFT`
- `KEEP_LANE`
- `LANE_RIGHT`

Longitudinal control is handled by a deterministic controller that cruises at
30 m/s, prepares for a lane change at 25 m/s, and follows slower traffic when no
safe adjacent lane is available. A safety layer exposes valid actions to
Maskable PPO and blocks unsafe or rapidly repeated lane changes.

This separation prevents the degenerate “always brake” policy observed when
speed and lane decisions were learned in a single five-action space.

## v1.0.0 results

The saved policy was evaluated deterministically on two independent, held-out
sets of 100 episodes. Neither split was used for training.

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

The 8-step lane-change cooldown matches the 8-step overtake window. This acts as
a commitment constraint: once an overtake starts, another manoeuvre cannot
silently replace it before the attempt resolves.

Raw summaries and per-episode records are stored in
[`results/v1.0.0_validation`](results/v1.0.0_validation) and
[`results/v1.0.0_test`](results/v1.0.0_test). Full model details are in the
[`v1.0.0 model card`](results/v1.0.0/MODEL_CARD.md).

## Architecture

| Layer | Responsibility | Learned |
|---|---|:---:|
| Maskable PPO | Select left, keep lane, or right | Yes |
| Longitudinal controller | Cruise, prepare, and safe following speed | No |
| Action mask / safety shield | Reject unsafe or unavailable lane changes | No |
| Commitment constraint | Prevent a new manoeuvre during the overtake window | No |

The observation has 53 values: 10 nearby vehicles × 5 normalized kinematic
features, followed by 3 action-availability flags.

## Core configuration

### Environment and controller

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

### Reward coefficients

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

### PPO

| Parameter | Value |
|---|---:|
| Network | MLP `[256, 256]` |
| `gamma` / `gae_lambda` | 0.99 / 0.95 |
| Learning rate | 1e-4 for final masked fine-tuning |
| Entropy coefficient | 0.01 |
| PPO epochs / clip range | 10 / 0.2 |
| `n_steps` / batch size | 256 / 256 |

The executable source of truth is [`src/highway_rl/config.py`](src/highway_rl/config.py).

## Installation

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

## Evaluate the release model

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

The default configuration already contains the v1.0.0 cooldown and overtake
window, so no command-line override is required.

## Train a fresh masked PPO policy

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

The bundled v1.0.0 checkpoint was warm-started from an earlier lateral PPO
policy and then fine-tuned with true action masking. Training a fresh policy can
produce different results; evaluate it on both held-out splits before reporting
metrics.

## Tests

```bash
python -m pytest -q
```

The test suite covers environment registration, observation/action dimensions,
reward timing, emergency speed control, action masks, overtake accounting, and
lane-change cooldown behaviour.

## Project structure

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
└── tests/test_environment.py
```

## Limitations

- The 50–52% strict overtake success rate is below the 70% stretch target.
- Safety is produced by the complete hybrid system, not by PPO alone.
- The evaluation covers highway-env simulation, not perception errors,
  high-fidelity vehicle dynamics, hardware latency, or real-road validation.
- The reported percentages are estimates from 100 episodes per held-out split,
  not real-world safety guarantees.
