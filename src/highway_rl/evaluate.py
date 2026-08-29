"""Evaluate a saved policy and write reproducible episode metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from highway_rl.config import ACTION_NAMES, ENV_CONFIG, SEED_SPLITS, apply_overrides
from highway_rl.environment import OVERTAKE_OUTCOMES, make_env


class Policy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


KEEP_LANE = ACTION_NAMES.index("KEEP_LANE")
NO_CRASH = "none"
FRONT_REAR_END = "ego rear-ended leader"
REAR_REAR_END = "rear-ended by follower"
LANE_CHANGE_CONTACT = "lane-change contact"


@dataclass
class EpisodeMetrics:
    episode: int
    collision: bool
    average_speed_mps: float
    cumulative_reward: float
    lane_changes: int
    steps: int
    lane_changes_per_100_steps: float
    shield_interventions: int
    shield_unsafe_requests: int
    shield_cooldown_requests: int
    unavailable_action_requests: int
    overtakes: int
    overtake_attempts: int
    overtake_outcomes: dict[str, int]
    overtake_superseded: int
    keep_lane_share: float
    crash_cause: str
    action_mix: dict[str, float]
    action_availability: dict[str, float]


def _lane_id(env) -> int:
    return int(env.unwrapped.vehicle.lane_index[2])


def _nearest(road, vehicle, lane_index, ahead: bool):
    """(gap, vehicle) of the nearest vehicle ahead of / behind the ego."""
    lane = road.network.get_lane(lane_index)
    s_self = lane.local_coordinates(vehicle.position)[0]
    best_gap = float("inf")
    best = None
    for other in road.vehicles:
        if other is vehicle:
            continue
        s_other, lat_other = lane.local_coordinates(other.position)
        if abs(lat_other) > lane.width_at(0.0) / 2.0 + 1.0:
            continue
        delta = s_other - s_self
        if ahead and 0 <= delta < best_gap:
            best_gap, best = delta, other
        elif not ahead and delta < 0 and -delta < best_gap:
            best_gap, best = -delta, other
    return best_gap, best


def _crash_cause(env) -> str:
    """Attribute a collision so safety can be reported per failure mode."""
    vehicle = env.unwrapped.vehicle
    road = env.unwrapped.road
    front_gap, front = _nearest(road, vehicle, vehicle.lane_index, True)
    rear_gap, rear = _nearest(road, vehicle, vehicle.lane_index, False)
    if front is not None and front_gap < 8.0:
        return FRONT_REAR_END
    if rear is not None and rear_gap < 8.0:
        return REAR_REAR_END
    return LANE_CHANGE_CONTACT


def evaluate(
    policy: Policy,
    episodes: int,
    seed: int,
    *,
    use_action_masks: bool = False,
    env=None,
) -> list[EpisodeMetrics]:
    # `env` lets a caller hand in an environment the policy is already bound
    # to, which is how the rule-based baselines are measured through exactly
    # the same metric pipeline as a learned policy.
    owns_env = env is None
    if owns_env:
        env = make_env()
    rows: list[EpisodeMetrics] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            previous_lane = _lane_id(env)
            speeds: list[float] = []
            total_reward = 0.0
            lane_changes = 0
            collision = False
            steps = 0
            interventions = 0
            unsafe_requests = 0
            cooldown_requests = 0
            unavailable_requests = 0
            overtakes = 0
            overtake_attempts = 0
            outcomes = {name: 0 for name in OVERTAKE_OUTCOMES}
            superseded = 0
            action_counts = [0] * len(ACTION_NAMES)
            availability_counts = [0] * len(ACTION_NAMES)
            terminated = truncated = False

            while not (terminated or truncated):
                action_mask = np.asarray(observation[-len(ACTION_NAMES):])
                for index, available in enumerate(action_mask):
                    availability_counts[index] += int(available > 0.5)
                predict_kwargs = (
                    {"action_masks": action_mask.astype(bool)}
                    if use_action_masks else {}
                )
                action, _ = policy.predict(
                    observation, deterministic=True, **predict_kwargs
                )
                action = int(action)
                action_counts[action] += 1
                unavailable_requests += int(action_mask[action] <= 0.5)
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_reward += float(reward)
                speeds.append(float(env.unwrapped.vehicle.speed))
                collision = collision or bool(info.get("crashed", env.unwrapped.vehicle.crashed))
                interventions += int(bool(info.get("shield_intervened", False)))
                unsafe_requests += int(info.get("shield_mode", 0) == 1)
                cooldown_requests += int(info.get("shield_mode", 0) == 2)
                overtakes += int(float(info.get("overtake_bonus", 0.0)) > 0.0)
                overtake_attempts += int(bool(info.get("overtake_attempt_started", False)))
                outcome = info.get("overtake_outcome")
                if outcome is not None:
                    outcomes[outcome] += 1
                superseded += int(bool(info.get("overtake_superseded", False)))
                current_lane = _lane_id(env)
                lane_changes += int(current_lane != previous_lane)
                previous_lane = current_lane

            rows.append(
                EpisodeMetrics(
                    episode=episode + 1,
                    collision=collision,
                    average_speed_mps=fmean(speeds) if speeds else 0.0,
                    cumulative_reward=total_reward,
                    lane_changes=lane_changes,
                    steps=steps,
                    lane_changes_per_100_steps=100.0 * lane_changes / steps if steps else 0.0,
                    shield_interventions=interventions,
                    shield_unsafe_requests=unsafe_requests,
                    shield_cooldown_requests=cooldown_requests,
                    unavailable_action_requests=unavailable_requests,
                    overtakes=overtakes,
                    overtake_attempts=overtake_attempts,
                    overtake_outcomes=dict(outcomes),
                    overtake_superseded=superseded,
                    keep_lane_share=action_counts[KEEP_LANE] / steps if steps else 0.0,
                    crash_cause=_crash_cause(env) if collision else NO_CRASH,
                    action_mix={
                        name: count / steps if steps else 0.0
                        for name, count in zip(ACTION_NAMES, action_counts)
                    },
                    action_availability={
                        name: count / steps if steps else 0.0
                        for name, count in zip(ACTION_NAMES, availability_counts)
                    },
                )
            )
    finally:
        if owns_env:
            env.close()
    return rows


def summarize(rows: list[EpisodeMetrics]) -> dict[str, float | int]:
    rewards = [row.cumulative_reward for row in rows]
    steps = sum(row.steps for row in rows) or 1
    interventions = sum(row.shield_interventions for row in rows)
    unsafe_requests = sum(row.shield_unsafe_requests for row in rows)
    cooldown_requests = sum(row.shield_cooldown_requests for row in rows)
    unavailable_requests = sum(row.unavailable_action_requests for row in rows)
    overtake_attempts = sum(row.overtake_attempts for row in rows)
    overtakes = sum(row.overtakes for row in rows)
    lane_changes = sum(row.lane_changes for row in rows)
    summary: dict[str, object] = {
        "episodes": len(rows),
        "collision_rate": fmean(float(row.collision) for row in rows),
        "episode_completion_rate": fmean(
            float(
                row.steps
                >= int(ENV_CONFIG["duration"] * ENV_CONFIG["policy_frequency"])
            )
            for row in rows
        ),
        "average_speed_mps": fmean(row.average_speed_mps for row in rows),
        "mean_cumulative_reward": fmean(rewards),
        "std_cumulative_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_lane_changes_per_episode": fmean(row.lane_changes for row in rows),
        "mean_lane_changes_per_100_steps": fmean(row.lane_changes_per_100_steps for row in rows),
        # how much of the observed safety comes from the shield rather than the policy
        "shield_intervention_rate": interventions / steps,
        "mean_shield_interventions_per_episode": interventions / len(rows),
        "shield_unsafe_request_rate": unsafe_requests / steps,
        "shield_cooldown_request_rate": cooldown_requests / steps,
        "unavailable_action_request_rate": unavailable_requests / steps,
        # collapses to 1.0 when the policy has stopped making lateral decisions
        "mean_keep_lane_share": fmean(row.keep_lane_share for row in rows),
        "mean_overtakes_per_episode": fmean(row.overtakes for row in rows),
        "mean_overtake_attempts_per_episode": fmean(row.overtake_attempts for row in rows),
        "overtake_success_rate": overtakes / overtake_attempts if overtake_attempts else 0.0,
        # why the rest of the attempts did not convert
        "overtake_outcomes": {
            name: sum(row.overtake_outcomes.get(name, 0) for row in rows)
            for name in OVERTAKE_OUTCOMES
        },
        # attempts replaced by another manoeuvre before they could resolve;
        # without this the outcome counts do not add up to the attempt count
        "overtake_superseded": sum(row.overtake_superseded for row in rows),
        "overtake_attempts_resolved": sum(
            sum(row.overtake_outcomes.values()) for row in rows
        ) + sum(row.overtake_superseded for row in rows),
        # Accounting invariant: every attempt must end in exactly one outcome
        # or be superseded. If this is False the outcome distribution above is
        # not a partition of the attempts and must not be read as percentages.
        "overtake_accounting_balanced": (
            sum(sum(row.overtake_outcomes.values()) for row in rows)
            + sum(row.overtake_superseded for row in rows)
        ) == overtake_attempts,
        "shield_interventions_per_lane_change": (
            interventions / lane_changes if lane_changes else 0.0
        ),
        "crash_causes": {
            cause: sum(row.crash_cause == cause for row in rows)
            for cause in (FRONT_REAR_END, REAR_REAR_END, LANE_CHANGE_CONTACT)
        },
        "action_mix": {
            name: fmean(row.action_mix[name] for row in rows) for name in ACTION_NAMES
        },
        "action_availability": {
            name: fmean(row.action_availability[name] for row in rows)
            for name in ACTION_NAMES
        },
    }
    return summary


def save_results(
    rows: list[EpisodeMetrics],
    output_dir: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    if metadata:
        summary.update(metadata)
    excluded = {"action_mix", "action_availability", "overtake_outcomes"}
    columns = [key for key in asdict(rows[0]) if key not in excluded]
    columns += [f"share_{name}" for name in ACTION_NAMES]
    columns += [f"available_{name}" for name in ACTION_NAMES]
    columns += [f"outcome_{name}" for name in OVERTAKE_OUTCOMES]
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            record = {
                key: value for key, value in asdict(row).items()
                if key not in excluded
            }
            record.update({f"share_{name}": share for name, share in row.action_mix.items()})
            record.update({
                f"available_{name}": share
                for name, share in row.action_availability.items()
            })
            record.update({
                f"outcome_{name}": row.overtake_outcomes.get(name, 0)
                for name in OVERTAKE_OUTCOMES
            })
            writer.writerow(record)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_highway.zip"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--split",
        choices=tuple(SEED_SPLITS),
        help="use a held-out seed set instead of --seed/--episodes; "
             "dev is for debugging only and must not be reported as final",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "maskable-ppo"),
        default="ppo",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="override ENV_CONFIG entries for this run, e.g. "
             "--set lane_change_cooldown=5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split:
        seed, episodes = SEED_SPLITS[args.split]
    else:
        seed, episodes = args.seed, args.episodes
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    overrides = apply_overrides(args.set)
    algorithm = MaskablePPO if args.algorithm == "maskable-ppo" else PPO
    model = algorithm.load(args.model_path, device="cpu")
    rows = evaluate(
        model,
        episodes,
        seed,
        use_action_masks=args.algorithm == "maskable-ppo",
    )
    summary = save_results(
        rows,
        args.output_dir,
        metadata={
            "seed_split": args.split or "custom",
            "seed_start": seed,
            "config_overrides": overrides,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
