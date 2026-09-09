"""Diagnose a saved policy: action mix, reward decomposition, traffic context.

Usage:
    python scripts/diagnose.py --algorithm maskable-ppo \
        --model-path models/ppo_highway_v1.0.0.zip
    python scripts/diagnose.py --random      # baseline without a model
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from highway_rl.config import ACTION_NAMES
from highway_rl.environment import make_env


def _current_action_mask(env) -> np.ndarray:
    """Read and validate action availability from the environment interface."""
    provider = getattr(env, "action_masks", None)
    if callable(provider):
        mask = provider()
    else:
        provider = getattr(env.unwrapped, "action_mask", None)
        if not callable(provider):
            raise AttributeError("environment does not expose an action mask")
        mask = provider()

    action_mask = np.asarray(mask, dtype=bool).reshape(-1)
    expected_shape = (len(ACTION_NAMES),)
    if action_mask.shape != expected_shape:
        raise ValueError(
            f"action mask has shape {action_mask.shape}, expected {expected_shape}"
        )
    if not np.any(action_mask):
        raise ValueError("action mask must leave at least one action available")
    return action_mask


def _validate_rollout_args(episodes: int, seed: int) -> None:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def rollout(policy, episodes: int, seed: int, use_masks: bool = False) -> dict:
    _validate_rollout_args(episodes, seed)
    env = make_env()
    unwrapped = env.unwrapped
    action_counts: Counter[str] = Counter()
    term_totals: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    speeds: list[float] = []
    lane_ids: list[int] = []
    headways: list[float] = []
    front_present = 0
    target_speeds: list[float] = []
    steps_total = 0
    collisions = 0

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = truncated = False
            crashed = False
            while not (terminated or truncated):
                predict_kwargs = (
                    {"action_masks": _current_action_mask(env)} if use_masks else {}
                )
                action, _ = policy.predict(
                    observation,
                    deterministic=True,
                    **predict_kwargs,
                )
                action_array = np.asarray(action)
                if action_array.size != 1:
                    raise ValueError(
                        f"policy returned {action_array.size} actions for one environment"
                    )
                action_index = int(action_array.item())
                if not 0 <= action_index < len(ACTION_NAMES):
                    raise ValueError(f"policy returned out-of-range action {action_index}")
                action_counts[ACTION_NAMES[action_index]] += 1

                front_vehicle, _ = unwrapped.road.neighbour_vehicles(
                    unwrapped.vehicle,
                    unwrapped.vehicle.lane_index,
                )
                if front_vehicle is not None:
                    front_present += 1
                    gap = float(unwrapped.vehicle.lane_distance_to(front_vehicle))
                    headways.append(gap / max(float(unwrapped.vehicle.speed), 1.0))

                observation, _reward, terminated, truncated, info = env.step(
                    action_index
                )

                outcome = info.get("overtake_outcome")
                if outcome is not None:
                    outcome_counts[outcome] += 1

                # Decompose the reward the environment just returned. This has to
                # run *after* the step: _rewards reads the decision-time snapshot
                # that step() captured, so calling it earlier would score the
                # previous decision against the current world.
                parts = unwrapped._rewards(int(info["applied_action"]))
                cfg = unwrapped.config
                for name, value in parts.items():
                    if name == "on_road_reward":
                        continue
                    term_totals[name] += float(cfg.get(name, 0.0)) * float(value)
                # The overtake bonus is added in step() rather than in _rewards,
                # so it is invisible to the decomposition above.
                term_totals["overtake_reward"] += float(
                    info.get("overtake_bonus", 0.0)
                )

                steps_total += 1
                speeds.append(float(unwrapped.vehicle.speed))
                lane_ids.append(int(unwrapped.vehicle.lane_index[2]))
                target_speeds.append(float(unwrapped.vehicle.target_speed))
                crashed = crashed or bool(
                    info.get("crashed", unwrapped.vehicle.crashed)
                )
            collisions += int(crashed)
    finally:
        env.close()

    total = max(steps_total, 1)
    return {
        "episodes": episodes,
        "steps_total": steps_total,
        "steps_per_episode": steps_total / episodes,
        "collision_rate": collisions / episodes,
        "action_share": {key: value / total for key, value in action_counts.most_common()},
        "action_count": dict(action_counts),
        "reward_per_step": {
            key: value / total for key, value in sorted(term_totals.items())
        },
        "reward_per_episode": {
            key: value / episodes for key, value in sorted(term_totals.items())
        },
        "mean_speed": float(np.mean(speeds)),
        "speed_p10_p90": (
            float(np.percentile(speeds, 10)),
            float(np.percentile(speeds, 90)),
        ),
        "mean_target_speed": float(np.mean(target_speeds)),
        "lane_distribution": {
            int(key): value / total
            for key, value in sorted(Counter(lane_ids).items())
        },
        "overtake_outcomes": dict(outcome_counts),
        "front_vehicle_share": front_present / total,
        "mean_time_headway_when_front": (
            float(np.mean(headways)) if headways else float("nan")
        ),
        "pct_headway_below_2s": (
            float(np.mean(np.array(headways) < 2.0)) if headways else float("nan")
        ),
    }


class RandomPolicy:
    """Reproducible uniform random policy over the lateral meta-actions."""

    def __init__(
        self,
        seed: int = 0,
        n_actions: int = len(ACTION_NAMES),
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions < 1:
            raise ValueError("n_actions must be a positive integer")
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def predict(self, observation, deterministic: bool = True):
        return int(self.rng.integers(self.n_actions)), None


def throughput(seconds: float = 20.0) -> float:
    """Measure environment interaction speed in steps/second (single env)."""
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be finite and > 0")

    env = make_env()
    _observation, _ = env.reset(seed=0)
    env.action_space.seed(0)
    steps = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            _observation, _reward, terminated, truncated, _info = env.step(
                env.action_space.sample()
            )
            steps += 1
            if terminated or truncated:
                _observation, _ = env.reset()
    finally:
        env.close()
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--random",
        action="store_true",
        help="use a uniform random policy",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument(
        "--throughput",
        type=float,
        default=0.0,
        help="seconds to spend measuring env steps/second",
    )
    parser.add_argument(
        "--algorithm",
        choices=("ppo", "maskable-ppo"),
        default="ppo",
        help="algorithm stored in --model-path",
    )
    args = parser.parse_args()

    try:
        _validate_rollout_args(args.episodes, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if not np.isfinite(args.throughput) or args.throughput < 0.0:
        parser.error("--throughput must be finite and >= 0")

    if args.random:
        policy = RandomPolicy(seed=args.seed)
        label = "RANDOM policy"
        use_masks = False
    else:
        if not args.model_path:
            parser.error("--model-path is required unless --random is used")
        if args.algorithm == "maskable-ppo":
            from sb3_contrib import MaskablePPO

            policy = MaskablePPO.load(args.model_path, device="cpu")
        else:
            policy = PPO.load(args.model_path, device="cpu")
        use_masks = args.algorithm == "maskable-ppo"
        label = f"model {args.model_path.name} ({args.algorithm})"

    print(f"=== {label} ===")
    report = rollout(policy, args.episodes, args.seed, use_masks=use_masks)
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, float):
                    print(f"    {inner_key:<32} {inner_value: .4f}")
                else:
                    print(f"    {inner_key:<32} {inner_value}")
        elif isinstance(value, float):
            print(f"{key:<28} {value: .4f}")
        else:
            print(f"{key:<28} {value}")

    if args.throughput > 0:
        rate = throughput(args.throughput)
        estimate = 200_000 / (rate * 4) / 60
        print(f"\nenv throughput: {rate:.1f} steps/s (single process)")
        print(
            f"  -> 200k steps on 4 parallel envs ~= {estimate:.1f} min "
            "(wall-clock, excluding gradient updates)"
        )


if __name__ == "__main__":
    main()
