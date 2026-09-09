# Highway Lane-Change Decision Making with Maskable PPO

[![Version](https://img.shields.io/badge/version-v1.0.1-blueviolet)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#quick-start)
[![CI](https://github.com/Jerry-124/highway-rl-lane-change-decision/actions/workflows/ci.yml/badge.svg)](https://github.com/Jerry-124/highway-rl-lane-change-decision/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-40-brightgreen)](#verification)

A reproducible simulation research project for high-level highway lane-change decision making with Maskable PPO, deterministic longitudinal control, action masking, and explicit execution constraints in `highway-env`.

> This repository evaluates a hybrid decision system in simulation. It is not a real-vehicle safety system and does not model production perception, actuation, hardware latency, or road risk.

## Highlights

- **Lateral-only learned policy:** Maskable PPO selects `LANE_LEFT`, `KEEP_LANE`, or `LANE_RIGHT`.
- **Deterministic longitudinal control:** speed control is separated from the reinforcement-learning action space.
- **Action masking and safety constraints:** unavailable or unsafe lane changes are excluded before policy sampling and guarded again at execution time.
- **Commitment constraint:** the lane-change cooldown matches the overtake window, preventing a new manoeuvre from silently replacing an active overtake attempt.
- **Held-out evaluation:** the bundled v1.0.0 checkpoint is evaluated on independent 100-episode validation and test splits.
- **Traceable artifacts:** raw episode records, summaries, a model card, and a SHA-256 checksum are committed with the release model.
- **Validated experiment interfaces:** configuration overrides are typed and transactional, evaluation reads action availability from the environment rather than observation layout, and new summaries record model/software provenance automatically.
- **Reproducible diagnostics:** auxiliary random baselines use explicit local RNG seeds, and diagnostic action masking also queries the environment interface rather than relying on observation layout.

## System Architecture

| Layer | Responsibility | Learned |
|---|---|:---:|
| Maskable PPO | Select left, keep lane, or right | Yes |
| Longitudinal controller | Cruise, prepare, and safe-following speed | No |
| Action mask / safety shield | Reject unsafe or unavailable lane changes | No |
| Commitment constraint | Prevent a new manoeuvre during the overtake window | No |

The observation contains 53 values in the stable masked configuration: 10 nearby vehicles × 5 normalized kinematic features, followed by 3 action-availability flags. Evaluation does not depend on those flags being at a hard-coded observation position; it queries the environment's action-mask interface directly.

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

The executable configuration source is [`src/highway_rl/config.py`](src/highway_rl/config.py). Command-line `--set KEY=VALUE` overrides are parsed according to the existing value type and validated transactionally, so inputs such as `shield_enabled=false` become the boolean `False` rather than the truthy string `"false"`.

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

Newly generated `summary.json` files include the software version, algorithm, resolved model path, model SHA-256, seed provenance, and applied configuration overrides in addition to the computed metrics.

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

Training arguments are validated before environments or model state are created. Invalid counts, non-finite hyperparameters, incompatible batch geometry, missing resume artifacts, and `--reset-critic` without `--resume-from` fail explicitly.

The bundled v1.0.0 checkpoint was warm-started from an earlier lateral PPO policy and then fine-tuned with action masking. A fresh training run is not expected to be bit-identical and should be evaluated independently before its metrics are reported.

## Verification

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check scripts tests
```

The repository contains 40 pytest tests covering environment registration, observation/action dimensions, longitudinal-control behavior, reward timing, action masks, overtake accounting, lane-change cooldown behavior, configuration override semantics, training-argument validation, evaluation provenance, diagnostic reproducibility, and result serialization. Scenario-sensitive controller and overtake checks use deterministic constructed traffic states instead of skipping when a sampled scene is unsuitable.

GitHub Actions runs on Python 3.10 and 3.12 and checks dependency consistency, source/test/script compilation, the complete pytest suite, repository-wide Ruff linting, and Ruff formatting for maintained scripts and tests. CI installs the CPU build of PyTorch because the regression suite does not require CUDA.

## Reproducibility

The stable evaluation artifact is:

```text
models/ppo_highway_v1.0.0.zip
SHA-256: 38969cd8dc3343d7be26751b9da4fb676f0af4d29031c6401ad83938663dcda8
```

The checksum is recorded in [`results/v1.0.0/SHA256SUMS.txt`](results/v1.0.0/SHA256SUMS.txt). Development seeds 3000–3019 were used during iteration and are not reported as final evidence; the final validation and test splits use seeds 5000–5099 and 9000–9099, respectively.

Version v1.0.1 is a software-maintenance baseline built on the same evaluated v1.0.0 model and result artifacts. It does not retroactively modify the v1.0.0 benchmark values.

## Repository Structure

```text
highway-rl-lane-change-decision/
├── .github/workflows/ci.yml
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
    ├── test_config.py
    ├── test_diagnostics.py
    ├── test_environment.py
    ├── test_evaluate.py
    └── test_train.py
```

## Scope and Limitations

- The 50–52% strict overtake success rate remains below the 70% stretch target.
- Safety metrics describe the complete hybrid system, not Maskable PPO in isolation.
- The evaluation covers `highway-env` simulation rather than perception errors, high-fidelity vehicle dynamics, hardware latency, or real-road validation.
- The reported percentages are estimates from 100 episodes per held-out split and are not real-world safety guarantees.
- The bundled checkpoint was warm-started; fresh training runs can produce different policies and metrics.

## License

This project is licensed under the [MIT License](LICENSE).
