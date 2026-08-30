"""Evaluate a saved policy and write reproducible episode metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Protocol

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from highway_rl.config import ACTION_NAMES, SEED_SPLITS, apply_overrides
from highway_rl.environment import OVERTAKE_OUTCOMES, make_env


class Policy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


KEEP_LANE = ACTION_NAMES.index("KEEP_LANE")
NO_CRASH = "none"
FRONT_REAR_END = "ego rear-ended leader"
REAR_REAR_END = "rear-ended by follower"
LANE_CHANGE_CONTACT = "lane-change contact"
SERIALIZED_MAPPING_FIELDS = {
    "action_mix",
    "action_availability",
    "overtake_outcomes",
}


@dataclass
class EpisodeMetrics:
    episode: int
    collision: bool
    average_speed_mps: float
    cumulative_reward: float
    lane_changes: int
    steps: int
    expected_steps: int
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


@dataclass
class _EpisodeAccumulator:
    previous_lane: int
    expected_steps: int
    speeds: list[float] = field(default_factory=list)
    total_reward: float = 0.0
    lane_changes: int = 0
    collision: bool = False
    steps: int = 0
    interventions: int = 0
    unsafe_requests: int = 0
    cooldown_requests: int = 0
    unavailable_requests: int = 0
    overtakes: int = 0
    overtake_attempts: int = 0
    superseded: int = 0
    outcomes: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in OVERTAKE_OUTCOMES}
    )
    action_counts: list[int] = field(
        default_factory=lambda: [0] * len(ACTION_NAMES)
    )
    availability_counts: list[int] = field(
        default_factory=lambda: [0] * len(ACTION_NAMES)
    )

    def record_action_mask(self, action_mask: np.ndarray) -> None:
        for index, available in enumerate(action_mask):
            self.availability_counts[index] += int(available > 0.5)

    def record_requested_action(self, action: int, action_mask: np.ndarray) -> None:
        self.action_counts[action] += 1
        self.unavailable_requests += int(action_mask[action] <= 0.5)

    def record_transition(self, env, reward: float, info: dict) -> None:
        self.steps += 1
        self.total_reward += float(reward)
        self.speeds.append(float(env.unwrapped.vehicle.speed))
        self.collision = self.collision or bool(
            info.get("crashed", env.unwrapped.vehicle.crashed)
        )
        self.interventions += int(bool(info.get("shield_intervened", False)))
        self.unsafe_requests += int(info.get("shield_mode", 0) == 1)
        self.cooldown_requests += int(info.get("shield_mode", 0) == 2)
        self.overtakes += int(float(info.get("overtake_bonus", 0.0)) > 0.0)
        self.overtake_attempts += int(bool(info.get("overtake_attempt_started", False)))
        outcome = info.get("overtake_outcome")
        if outcome is not None:
            self.outcomes[outcome] += 1
        self.superseded += int(bool(info.get("overtake_superseded", False)))
        current_lane = _lane_id(env)
        self.lane_changes += int(current_lane != self.previous_lane)
        self.previous_lane = current_lane

    def to_metrics(self, episode: int, env) -> EpisodeMetrics:
        steps = self.steps
        return EpisodeMetrics(
            episode=episode + 1,
            collision=self.collision,
            average_speed_mps=fmean(self.speeds) if self.speeds else 0.0,
            cumulative_reward=self.total_reward,
            lane_changes=self.lane_changes,
            steps=steps,
            expected_steps=self.expected_steps,
            lane_changes_per_100_steps=(
                100.0 * self.lane_changes / steps if steps else 0.0
            ),
            shield_interventions=self.interventions,
            shield_unsafe_requests=self.unsafe_requests,
            shield_cooldown_requests=self.cooldown_requests,
            unavailable_action_requests=self.unavailable_requests,
            overtakes=self.overtakes,
            overtake_attempts=self.overtake_attempts,
            overtake_outcomes=dict(self.outcomes),
            overtake_superseded=self.superseded,
            keep_lane_share=(self.action_counts[KEEP_LANE] / steps if steps else 0.0),
            crash_cause=_crash_cause(env) if self.collision else NO_CRASH,
            action_mix={
                name: count / steps if steps else 0.0
                for name, count in zip(ACTION_NAMES, self.action_counts)
            },
            action_availability={
                name: count / steps if steps else 0.0
                for name, count in zip(ACTION_NAMES, self.availability_counts)
            },
        )


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


def _expected_episode_steps(env) -> int:
    return int(
        float(env.unwrapped.config["duration"])
        * float(env.unwrapped.config["policy_frequency"])
    )


def _select_action(
    policy: Policy,
    observation: np.ndarray,
    use_action_masks: bool,
    accumulator: _EpisodeAccumulator,
) -> int:
    action_mask = np.asarray(observation[-len(ACTION_NAMES) :])
    accumulator.record_action_mask(action_mask)
    predict_kwargs = (
        {"action_masks": action_mask.astype(bool)} if use_action_masks else {}
    )
    action, _ = policy.predict(observation, deterministic=True, **predict_kwargs)
    action = int(action)
    accumulator.record_requested_action(action, action_mask)
    return action


def _evaluate_episode(
    policy: Policy,
    env,
    episode: int,
    seed: int,
    use_action_masks: bool,
) -> EpisodeMetrics:
    observation, _ = env.reset(seed=seed + episode)
    accumulator = _EpisodeAccumulator(
        previous_lane=_lane_id(env),
        expected_steps=_expected_episode_steps(env),
    )
    terminated = truncated = False
    while not (terminated or truncated):
        action = _select_action(
            policy,
            observation,
            use_action_masks,
            accumulator,
        )
        observation, reward, terminated, truncated, info = env.step(action)
        accumulator.record_transition(env, reward, info)
    return accumulator.to_metrics(episode, env)


def evaluate(
    policy: Policy,
    episodes: int,
    seed: int,
    *,
    use_action_masks: bool = False,
    env=None,
) -> list[EpisodeMetrics]:
    # `env` lets a caller hand in an environment the policy is already bound
    # to, which is how rule-based baselines use the same metric pipeline.
    owns_env = env is None
    if owns_env:
        env = make_env()
    try:
        return [
            _evaluate_episode(policy, env, episode, seed, use_action_masks)
            for episode in range(episodes)
        ]
    finally:
        if owns_env:
            env.close()


def _sum_metric(rows: list[EpisodeMetrics], attribute: str) -> int:
    return sum(int(getattr(row, attribute)) for row in rows)


def _mean_metric(rows: list[EpisodeMetrics], attribute: str) -> float:
    return fmean(float(getattr(row, attribute)) for row in rows)


def _mean_mapping(
    rows: list[EpisodeMetrics],
    attribute: str,
    names: list[str],
) -> dict[str, float]:
    return {
        name: fmean(float(getattr(row, attribute)[name]) for row in rows)
        for name in names
    }


def _overtake_outcome_counts(rows: list[EpisodeMetrics]) -> dict[str, int]:
    return {
        name: sum(row.overtake_outcomes.get(name, 0) for row in rows)
        for name in OVERTAKE_OUTCOMES
    }


def _crash_cause_counts(rows: list[EpisodeMetrics]) -> dict[str, int]:
    causes = (FRONT_REAR_END, REAR_REAR_END, LANE_CHANGE_CONTACT)
    return {cause: sum(row.crash_cause == cause for row in rows) for cause in causes}


def _overtake_accounting(rows: list[EpisodeMetrics]) -> tuple[int, int, bool]:
    resolved = sum(sum(row.overtake_outcomes.values()) for row in rows)
    superseded = _sum_metric(rows, "overtake_superseded")
    attempts = _sum_metric(rows, "overtake_attempts")
    accounted = resolved + superseded
    return superseded, accounted, accounted == attempts


def summarize(rows: list[EpisodeMetrics]) -> dict[str, object]:
    rewards = [row.cumulative_reward for row in rows]
    steps = _sum_metric(rows, "steps") or 1
    episodes = len(rows)
    interventions = _sum_metric(rows, "shield_interventions")
    unsafe_requests = _sum_metric(rows, "shield_unsafe_requests")
    cooldown_requests = _sum_metric(rows, "shield_cooldown_requests")
    unavailable_requests = _sum_metric(rows, "unavailable_action_requests")
    overtake_attempts = _sum_metric(rows, "overtake_attempts")
    overtakes = _sum_metric(rows, "overtakes")
    lane_changes = _sum_metric(rows, "lane_changes")
    superseded, attempts_resolved, accounting_balanced = _overtake_accounting(rows)

    return {
        "episodes": episodes,
        "collision_rate": _mean_metric(rows, "collision"),
        "episode_completion_rate": fmean(
            float(row.steps >= row.expected_steps) for row in rows
        ),
        "average_speed_mps": _mean_metric(rows, "average_speed_mps"),
        "mean_cumulative_reward": fmean(rewards),
        "std_cumulative_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_lane_changes_per_episode": _mean_metric(rows, "lane_changes"),
        "mean_lane_changes_per_100_steps": _mean_metric(
            rows, "lane_changes_per_100_steps"
        ),
        "shield_intervention_rate": interventions / steps,
        "mean_shield_interventions_per_episode": interventions / episodes,
        "shield_unsafe_request_rate": unsafe_requests / steps,
        "shield_cooldown_request_rate": cooldown_requests / steps,
        "unavailable_action_request_rate": unavailable_requests / steps,
        "mean_keep_lane_share": _mean_metric(rows, "keep_lane_share"),
        "mean_overtakes_per_episode": _mean_metric(rows, "overtakes"),
        "mean_overtake_attempts_per_episode": _mean_metric(rows, "overtake_attempts"),
        "overtake_success_rate": (
            overtakes / overtake_attempts if overtake_attempts else 0.0
        ),
        "overtake_outcomes": _overtake_outcome_counts(rows),
        "overtake_superseded": superseded,
        "overtake_attempts_resolved": attempts_resolved,
        "overtake_accounting_balanced": accounting_balanced,
        "shield_interventions_per_lane_change": (
            interventions / lane_changes if lane_changes else 0.0
        ),
        "crash_causes": _crash_cause_counts(rows),
        "action_mix": _mean_mapping(rows, "action_mix", ACTION_NAMES),
        "action_availability": _mean_mapping(
            rows, "action_availability", ACTION_NAMES
        ),
    }


def _episode_columns(row: EpisodeMetrics) -> list[str]:
    columns = [
        key for key in asdict(row) if key not in SERIALIZED_MAPPING_FIELDS
    ]
    columns.extend(f"share_{name}" for name in ACTION_NAMES)
    columns.extend(f"available_{name}" for name in ACTION_NAMES)
    columns.extend(f"outcome_{name}" for name in OVERTAKE_OUTCOMES)
    return columns


def _episode_record(row: EpisodeMetrics) -> dict[str, object]:
    record = {
        key: value
        for key, value in asdict(row).items()
        if key not in SERIALIZED_MAPPING_FIELDS
    }
    record.update(
        {f"share_{name}": share for name, share in row.action_mix.items()}
    )
    record.update(
        {
            f"available_{name}": share
            for name, share in row.action_availability.items()
        }
    )
    record.update(
        {
            f"outcome_{name}": row.overtake_outcomes.get(name, 0)
            for name in OVERTAKE_OUTCOMES
        }
    )
    return record


def _write_episode_csv(rows: list[EpisodeMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_episode_columns(rows[0]))
        writer.writeheader()
        writer.writerows(_episode_record(row) for row in rows)


def _write_summary(summary: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


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
    _write_episode_csv(rows, output_dir / "episodes.csv")
    _write_summary(summary, output_dir / "summary.json")
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
        "--algorithm", choices=("ppo", "maskable-ppo"), default="ppo"
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
