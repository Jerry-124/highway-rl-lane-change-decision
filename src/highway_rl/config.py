"""Environment, reward, shield, and PPO settings for the v1.0 baseline.

Scope (matches the original project brief): PPO decides **whether and where to
change lanes**. Longitudinal control is a fixed rule-based layer underneath, so
"press SLOWER forever" is not an action the policy can even express. TTC-based
speed control, acceleration limits, action masking, and the safety shield form
the fixed execution layer beneath the learned lateral policy.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

ENV_ID = "highway-rl-v0"

# The learned action space: keep lane, or move one lane sideways.
ACTION_NAMES = ["LANE_LEFT", "KEEP_LANE", "LANE_RIGHT"]

ENV_CONFIG: dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        # lateral only: speed is handled by the rule-based controller below
        "longitudinal": False,
        "lateral": True,
    },
    "lanes_count": 4,
    "vehicles_count": 25,
    "vehicles_density": 1.0,
    "duration": 40,
    "ego_spacing": 2.0,
    # --- reward weights -------------------------------------------------
    "collision_reward": -25.0,
    "high_speed_reward": 1.1,
    "alive_reward": 0.0,
    "right_lane_reward": 0.05,
    "lane_change_reward": 0.0,
    "completed_lane_change_reward": 0.0,
    "overtake_reward": 2.0,
    "invalid_lane_change_penalty": -0.5,
    "unnecessary_lane_change_penalty": -0.5,
    "blocked_keep_penalty": -1.0,
    "shield_violation_penalty": -0.5,
    "headway_penalty": -0.5,
    # --- shaping thresholds ---------------------------------------------
    "safe_time_headway": 1.5,
    "desired_speed": 30.0,
    "block_horizon": 8.0,
    # --- rule-based longitudinal controller ------------------------------
    "cruise_speed": 30.0,
    "prepare_speed": 25.0,
    "min_cruise_speed": 10.0,
    "follow_ttc": 5.0,
    "emergency_ttc": 2.0,
    # --- overtake event --------------------------------------------------
    "overtake_margin": 5.0,
    "overtake_window": 8,
    "lane_change_cooldown": 8,
    # --- safety shield ---------------------------------------------------
    "shield_enabled": True,
    "shield_ttc": 3.0,
    "shield_gap_front": 20.0,
    "shield_rear_geometry": 8.0,
    "shield_lookahead": 60.0,
    "shield_min_lane_speed": 22.0,
    # --- observation -----------------------------------------------------
    "action_mask_observation": True,
    # --- ego dynamics ----------------------------------------------------
    "ego_kp_accel": 1.05,
    "ego_max_accel": 3.0,
    "ego_max_decel": 4.0,
    # --- simulation ------------------------------------------------------
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "offroad_terminal": True,
    "policy_frequency": 1,
    "simulation_frequency": 5,
}

PPO_CONFIG: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "policy_kwargs": {"net_arch": [256, 256]},
}

# Seed sets. 3000-3019 is reserved for tuning and must not be reported as a
# final result; validation and test seeds are held out.
SEED_SPLITS: dict[str, tuple[int, int]] = {
    "dev": (3000, 20),
    "validation": (5000, 100),
    "test": (9000, 100),
}


def _finite_number(config: dict[str, Any], key: str) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _require_positive(config: dict[str, Any], key: str) -> None:
    if _finite_number(config, key) <= 0.0:
        raise ValueError(f"{key} must be > 0")


def _require_nonnegative(config: dict[str, Any], key: str) -> None:
    if _finite_number(config, key) < 0.0:
        raise ValueError(f"{key} must be >= 0")


def _require_integer(config: dict[str, Any], key: str, *, minimum: int) -> None:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")


def validate_env_config(config: dict[str, Any]) -> None:
    """Validate execution-critical configuration without changing it."""
    for key in (
        "safe_time_headway",
        "desired_speed",
        "block_horizon",
        "cruise_speed",
        "prepare_speed",
        "follow_ttc",
        "emergency_ttc",
        "overtake_margin",
        "shield_ttc",
        "shield_gap_front",
        "shield_rear_geometry",
        "shield_lookahead",
        "shield_min_lane_speed",
        "ego_kp_accel",
        "ego_max_accel",
        "ego_max_decel",
        "vehicles_density",
        "duration",
    ):
        _require_positive(config, key)

    _require_nonnegative(config, "min_cruise_speed")
    _require_integer(config, "lanes_count", minimum=1)
    _require_integer(config, "vehicles_count", minimum=0)
    _require_integer(config, "overtake_window", minimum=1)
    _require_integer(config, "lane_change_cooldown", minimum=0)
    _require_integer(config, "policy_frequency", minimum=1)
    _require_integer(config, "simulation_frequency", minimum=1)

    for key in (
        "shield_enabled",
        "action_mask_observation",
        "normalize_reward",
        "offroad_terminal",
    ):
        if not isinstance(config[key], bool):
            raise TypeError(f"{key} must be a boolean")

    speed_range = config["reward_speed_range"]
    if not isinstance(speed_range, list) or len(speed_range) != 2:
        raise TypeError("reward_speed_range must be a two-element list")
    low, high = speed_range
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (low, high)
    ):
        raise ValueError("reward_speed_range values must be finite numbers")
    if float(low) >= float(high):
        raise ValueError("reward_speed_range must be strictly increasing")

    if float(config["min_cruise_speed"]) > float(config["cruise_speed"]):
        raise ValueError("min_cruise_speed must not exceed cruise_speed")
    if int(config["simulation_frequency"]) < int(config["policy_frequency"]):
        raise ValueError("simulation_frequency must be >= policy_frequency")


def _parse_override_value(raw: str, current: object) -> object:
    text = raw.strip()
    if isinstance(current, bool):
        normalized = text.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise ValueError(f"expected true or false, got {raw!r}")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"expected a finite float, got {raw!r}")
        return value
    if isinstance(current, str):
        return text
    if isinstance(current, list):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected a JSON list, got {raw!r}")
        return value
    raise TypeError(f"overrides are not supported for {type(current).__name__} values")


def apply_overrides(overrides: list[str]) -> dict[str, object]:
    """Apply validated ``KEY=VALUE`` pairs to ``ENV_CONFIG`` transactionally.

    Values are parsed according to the existing configuration type. In
    particular, boolean strings are converted to real booleans rather than
    truthy strings. All requested changes are validated on a copy first; if any
    item is invalid, ``ENV_CONFIG`` remains unchanged.
    """
    candidate = copy.deepcopy(ENV_CONFIG)
    applied: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("override key must not be empty")
        if key not in candidate:
            raise KeyError(f"{key!r} is not a known ENV_CONFIG entry")
        value = _parse_override_value(raw, candidate[key])
        candidate[key] = value
        applied[key] = value

    validate_env_config(candidate)
    ENV_CONFIG.clear()
    ENV_CONFIG.update(candidate)
    return applied


# Acceptance targets on the validation split (100 episodes).
ACCEPTANCE: dict[str, Any] = {
    "collision_rate_max": 0.05,
    "front_rear_end_rate_max": 0.02,
    "lane_change_contact_rate_max": 0.01,
    "episode_completion_min": 0.95,
    "average_speed_range": (22.0, 24.0),
    "lane_changes_per_episode_range": (1.0, 3.0),
    "weaving_events": 0,
    "overtake_success_rate_min": 0.70,
    "unnecessary_lane_changes_max": 0.2,
    "shield_intervention_rate_max": 0.05,
    "max_deceleration": 4.0,
}

validate_env_config(ENV_CONFIG)
