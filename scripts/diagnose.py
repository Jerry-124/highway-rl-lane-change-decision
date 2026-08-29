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


def rollout(policy, episodes: int, seed: int, use_masks: bool = False) -> dict:
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
    crashes_per_episode = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            terminated = truncated = False
            crashed = False
            while not (terminated or truncated):
                predict_kwargs = (
                    {"action_masks": np.asarray(obs[-len(ACTION_NAMES):]).astype(bool)}
                    if use_masks else {}
                )
                action, _ = policy.predict(obs, deterministic=True, **predict_kwargs)
                action = int(action)
                action_counts[ACTION_NAMES[action]] += 1

                front_vehicle, _ = unwrapped.road.neighbour_vehicles(
                    unwrapped.vehicle, unwrapped.vehicle.lane_index
                )
                if front_vehicle is not None:
                    front_present += 1
                    gap = float(unwrapped.vehicle.lane_distance_to(front_vehicle))
                    headways.append(gap / max(float(unwrapped.vehicle.speed), 1.0))

                obs, _reward, terminated, truncated, info = env.step(action)

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
                # so it is invisible to the decomposition above. It is the
                # largest lateral incentive, which makes it the worst term to
                # be missing from this report.
                term_totals["overtake_reward"] += float(
                    info.get("overtake_bonus", 0.0)
                )

                steps_total += 1
                speeds.append(float(unwrapped.vehicle.speed))
                lane_ids.append(int(unwrapped.vehicle.lane_index[2]))
                target_speeds.append(float(unwrapped.vehicle.target_speed))
                crashed = crashed or bool(info.get("crashed", unwrapped.vehicle.crashed))
            collisions += int(crashed)
            crashes_per_episode.append(crashed)
    finally:
        env.close()

    total = max(steps_total, 1)
    return {
        "episodes": episodes,
        "steps_total": steps_total,
        "steps_per_episode": steps_total / episodes,
        "collision_rate": collisions / episodes,
        "action_share": {k: v / total for k, v in action_counts.most_common()},
        "action_count": dict(action_counts),
        "reward_per_step": {k: v / total for k, v in sorted(term_totals.items())},
        "reward_per_episode": {k: v / episodes for k, v in sorted(term_totals.items())},
        "mean_speed": float(np.mean(speeds)),
        "speed_p10_p90": (float(np.percentile(speeds, 10)), float(np.percentile(speeds, 90))),
        "mean_target_speed": float(np.mean(target_speeds)),
        "lane_distribution": {
            int(k): v / total for k, v in sorted(Counter(lane_ids).items())
        },
        "overtake_outcomes": dict(outcome_counts),
        "front_vehicle_share": front_present / total,
        "mean_time_headway_when_front": float(np.mean(headways)) if headways else float("nan"),
        "pct_headway_below_2s": float(np.mean(np.array(headways) < 2.0)) if headways else float("nan"),
    }


class RandomPolicy:
    """Uniform random policy over the lateral meta-actions."""

    def __init__(self, n_actions: int = len(ACTION_NAMES)) -> None:
        self.n_actions = n_actions

    def predict(self, observation, deterministic: bool = True):
        return np.random.randint(self.n_actions), None


def throughput(seconds: float = 20.0) -> float:
    """Measure environment interaction speed in steps/second (single env)."""
    env = make_env()
    obs, _ = env.reset(seed=0)
    steps = 0
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            obs, _r, terminated, truncated, _i = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--random", action="store_true", help="use a uniform random policy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--throughput", type=float, default=0.0,
                        help="seconds to spend measuring env steps/second")
    parser.add_argument("--algorithm", choices=("ppo", "maskable-ppo"), default="ppo",
                        help="algorithm stored in --model-path")
    args = parser.parse_args()

    if args.random:
        policy = RandomPolicy()
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
        print(f"\nenv throughput: {rate:.1f} steps/s (single process)")
        print(f"  -> 200k steps on 4 parallel envs ~= {200_000 / (rate * 4) / 60:.1f} min "
              f"(wall-clock, excluding gradient updates)")


if __name__ == "__main__":
    main()
