# Model card — `ppo_highway_v1.0.0`

## Intended use

This model studies high-level highway lane-change decisions in simulation. It
selects `LANE_LEFT`, `KEEP_LANE`, or `LANE_RIGHT`. Longitudinal speed control and
the safety boundary are deterministic. It is not intended for real-vehicle use.

## Model and training

| Item | Value |
|---|---|
| Algorithm | Maskable PPO (`sb3-contrib`) |
| Network | MLP `[256, 256]` |
| Observation | 10 vehicles × 5 kinematic features + 3 action flags = 53 |
| Action space | 3 discrete lateral actions |
| Model file | `models/ppo_highway_v1.0.0.zip` |
| Initial weights | Earlier 5k-step lateral PPO checkpoint |
| Masked fine-tuning | 5,000 environment steps |
| Cumulative training budget | 10,000 environment steps |
| Parallel environments | 4, `SubprocVecEnv` |
| `n_steps` / batch size | 256 / 256 |
| Learning rate / entropy coefficient | 1e-4 / 0.01 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 |
| PPO epochs / clip range | 10 / 0.2 |
| Seed / device | 42 / CPU |

The release environment uses an 8-decision lane-change cooldown and an 8-step
overtake window. The checkpoint weights were not retrained for this parameter
change; the cooldown is an explicit execution constraint.

## Evaluation protocol

Evaluation is deterministic. Development seeds 3000–3019 were used while
iterating and are not reported as final evidence.

| Split | Seeds | Episodes | Role |
|---|---:|---:|---|
| Validation | 5000–5099 | 100 | Independent validation |
| Test | 9000–9099 | 100 | Held-out final test |

## Results

| Metric | Validation | Test | Target |
|---|---:|---:|---:|
| Collision rate | **1.0%** | **2.0%** | ≤5% |
| Episode completion | **99%** | **98%** | ≥95% |
| Average speed | **23.29 m/s** | **23.13 m/s** | 22–24 m/s |
| Lane changes / episode | **2.41** | **2.20** | 1–3 |
| Mean cumulative reward | 15.19 | 13.85 | Same reward only |
| Overtakes / episode | 0.95 | 0.91 | Diagnostic |
| Overtake success rate | 51.6% | 50.6% | Stretch: ≥70% |
| Shield intervention rate | **0.0%** | **0.0%** | ≤5% |
| Unavailable-action requests | **0.0%** | **0.0%** | 0% |
| Superseded attempts | **0** | **0** | 0 |

Validation contained one lane-change contact. Test contained one rear-end by a
follower and one lane-change contact; there were no ego-to-leader rear-ends in
either held-out set.

## Interpretation

The model meets the collision, completion, speed, lane-change-frequency, and
constraint-intervention targets on both held-out sets. Matching cooldown and
overtake-window lengths removes mid-pass abandonment without changing policy
weights. This is a system-level result: the learned lateral policy, rule-based
speed controller, action mask, and safety shield all contribute.

## Limitations

- Strict overtake success remains below the 70% stretch target.
- The checkpoint was warm-started; the bundled binary is the exact evaluated
  artifact, while a fresh training run is not expected to be bit-identical.
- highway-env does not model production perception, actuation, or road risk.
- Two 100-episode splits do not establish real-world safety.

## Evidence

- `results/v1.0.0_validation/summary.json`
- `results/v1.0.0_validation/episodes.csv`
- `results/v1.0.0_test/summary.json`
- `results/v1.0.0_test/episodes.csv`
