"""Attribute crashes and check the rule-based speed layer is actually safe.

With speed control moved out of the policy, "always keep lane" is the baseline
the learned policy has to beat: it must be at least as safe and faster.

Usage:
    python scripts/crash_audit.py --episodes 30
    python scripts/crash_audit.py --episodes 100 --split validation
"""

from __future__ import annotations

import argparse

import numpy as np

from highway_rl.config import ACTION_NAMES, ENV_CONFIG, SEED_SPLITS
from highway_rl.environment import make_env

KEEP_LANE = ACTION_NAMES.index("KEEP_LANE")
LANE_LEFT = ACTION_NAMES.index("LANE_LEFT")
LANE_RIGHT = ACTION_NAMES.index("LANE_RIGHT")


class ConstantPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, observation, deterministic: bool = True):
        return self.action, None


class RandomPolicy:
    def __init__(self, seed: int = 0, n_actions: int = len(ACTION_NAMES)) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def predict(self, observation, deterministic: bool = True):
        return int(self.rng.integers(self.n_actions)), None


class LateralHeuristic:
    """Change lanes when blocked and one side is clearly better, else hold."""

    def __init__(self, env) -> None:
        self.env = env

    def predict(self, observation, deterministic: bool = True):
        unwrapped = self.env.unwrapped
        if unwrapped._blocked() <= 0.0:
            return KEEP_LANE, None
        current = int(unwrapped.vehicle.lane_index[2])
        gap_here = unwrapped._front_gap()
        best_action, best_gain = KEEP_LANE, 0.0
        for action in (LANE_LEFT, LANE_RIGHT):
            candidate = unwrapped._target_lane_index(action)
            if not unwrapped._lane_change_safe(candidate):
                continue
            _front_v, front_gap, _rear_v, _rear_gap = unwrapped._lane_neighbours(candidate)
            gain = front_gap - gap_here
            if gain > 20.0 and gain > best_gain and int(candidate[2]) != current:
                best_gain, best_action = gain, action
        return best_action, None


def _nearest(road, vehicle, lane_index, ahead: bool):
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


def audit(policy_factory, episodes: int, seed: int) -> dict:
    env = make_env()
    crashes: list[str] = []
    speeds: list[float] = []
    steps_list: list[int] = []
    lane_changes: list[int] = []
    interventions = 0
    overtakes = 0

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            policy = policy_factory(env)
            steps = 0
            changes = 0
            previous_lane = int(env.unwrapped.vehicle.lane_index[2])
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = policy.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(int(action))
                steps += 1
                interventions += int(bool(info.get("shield_intervened", False)))
                overtakes += int(float(info.get("overtake_bonus", 0.0)) > 0.0)
                speeds.append(float(env.unwrapped.vehicle.speed))
                lane = int(env.unwrapped.vehicle.lane_index[2])
                changes += int(lane != previous_lane)
                previous_lane = lane
                if env.unwrapped.vehicle.crashed:
                    vehicle = env.unwrapped.vehicle
                    road = env.unwrapped.road
                    front_gap, front = _nearest(road, vehicle, vehicle.lane_index, True)
                    rear_gap, rear = _nearest(road, vehicle, vehicle.lane_index, False)
                    if front is not None and front_gap < 8.0:
                        crashes.append("ego rear-ended leader")
                    elif rear is not None and rear_gap < 8.0:
                        crashes.append("rear-ended by follower")
                    else:
                        crashes.append("lane-change contact")
                    break
            steps_list.append(steps)
            lane_changes.append(changes)
    finally:
        env.close()

    total_steps = max(sum(steps_list), 1)
    return {
        "collision_rate": len(crashes) / episodes,
        "mean_speed": float(np.mean(speeds)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_lane_changes": float(np.mean(lane_changes)),
        "mean_overtakes": overtakes / episodes,
        "shield_rate": interventions / total_steps,
        "causes": {c: crashes.count(c) for c in set(crashes)},
    }


def show(name: str, r: dict) -> None:
    causes = ", ".join(f"{k}={v}" for k, v in sorted(r["causes"].items())) or "none"
    print(f"{name:<26} coll={r['collision_rate'] * 100:5.1f}% | speed={r['mean_speed']:5.2f} | "
          f"lc={r['mean_lane_changes']:4.2f} | overtake={r['mean_overtakes']:4.2f} | "
          f"steps={r['mean_steps']:5.1f} | shield={r['shield_rate'] * 100:4.1f}%")
    print(f"{'':<26} causes: {causes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--split", choices=tuple(SEED_SPLITS))
    args = parser.parse_args()

    if args.split:
        seed, episodes = SEED_SPLITS[args.split]
    else:
        seed, episodes = args.seed, args.episodes

    print(f"actions: {ACTION_NAMES}  (speed is rule-based)")
    print(f"cruise={ENV_CONFIG['cruise_speed']} prepare={ENV_CONFIG['prepare_speed']} "
          f"decel<={ENV_CONFIG['ego_max_decel']} m/s^2")
    print(f"{episodes} episodes, seeds {seed}+\n")

    def const(action: int):
        return lambda _env: ConstantPolicy(action)

    show("Keep-lane baseline", audit(const(KEEP_LANE), episodes, seed))
    show("Lateral heuristic", audit(LateralHeuristic, episodes, seed))
    show("Random lateral (PPO start)",
         audit(lambda _e: RandomPolicy(args.seed), episodes, seed))
    show("Always change left", audit(const(LANE_LEFT), episodes, seed))


if __name__ == "__main__":
    main()
