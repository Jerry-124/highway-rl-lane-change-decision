"""Environment, reward, shield, and PPO settings for the v1.0 baseline.

Scope (matches the original project brief): PPO decides **whether and where to
change lanes**. Longitudinal control is a fixed rule-based layer underneath, so
"press SLOWER forever" is not an action the policy can even express. TTC-based
speed control, acceleration limits, action masking, and the safety shield form
the fixed execution layer beneath the learned lateral policy.
"""

from __future__ import annotations

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
    # collision stays at -25 until the contracted scope is validated.
    "collision_reward": -25.0,
    # efficiency: a good lane choice is what lets the controller run at cruise
    "high_speed_reward": 1.1,
    # paying per step for merely existing is what made braking optimal
    "alive_reward": 0.0,
    "right_lane_reward": 0.05,
    # never punish a legal lane change: it is the behaviour we want
    "lane_change_reward": 0.0,
    # small nudge only; the real lateral signal is the overtake bonus, so that
    # weaving cannot out-earn overtaking
    # A safe lane change is not automatically useful.  Masking already makes
    # exploration safe; only a completed overtake should earn lateral credit.
    "completed_lane_change_reward": 0.0,
    "overtake_reward": 2.0,
    "invalid_lane_change_penalty": -0.5,
    "unnecessary_lane_change_penalty": -0.5,
    # held the lane while blocked and a safe lane change was available
    "blocked_keep_penalty": -1.0,
    # asking the shield for something it had to override
    # The 5k policy still requested unavailable lateral actions on 26.4% of
    # decisions.  The mask is visible in the observation, so make ignoring it
    # clearly worse than KEEP_LANE without relaxing the safety boundary.
    "shield_violation_penalty": -0.5,
    # mild: most following behaviour belongs to the controller, but a good
    # lane choice should still keep the ego out of tight following
    "headway_penalty": -0.5,
    # --- shaping thresholds ---------------------------------------------
    "safe_time_headway": 1.5,
    "desired_speed": 30.0,
    # seconds of catch-up time at the desired speed below which the leader is
    # considered to be holding us up; drives every "blocked" term
    "block_horizon": 8.0,
    # --- rule-based longitudinal controller ------------------------------
    "cruise_speed": 30.0,
    "prepare_speed": 25.0,
    # absolute standstill guard only; deliberately well below traffic speed so
    # it can never block the controller from backing off behind a slow leader
    "min_cruise_speed": 10.0,
    # ease down to the prepare speed while looking for a gap
    "follow_ttc": 5.0,
    # fall back to the leader's speed, then below it, as the gap tightens
    "emergency_ttc": 2.0,
    # --- overtake event --------------------------------------------------
    # metres the ego must be ahead of the vehicle it passed
    "overtake_margin": 5.0,
    # decision steps the pass may take after the manoeuvre was initiated
    "overtake_window": 8,
    # decision steps before another lane change is allowed
    # Match the overtake window so a pass cannot be abandoned by immediately
    # starting another lane change. This commitment constraint eliminated
    # superseded attempts on both held-out seed sets.
    "lane_change_cooldown": 8,
    # --- safety shield ---------------------------------------------------
    # Rear traffic is deliberately ignored when judging a merge: following
    # vehicles run IDM and are expected to brake for us. Only a minimum
    # geometric clearance is enforced, so we never merge onto a car alongside.
    "shield_enabled": True,
    "shield_ttc": 3.0,
    "shield_gap_front": 20.0,
    "shield_rear_geometry": 8.0,
    "shield_lookahead": 60.0,
    "shield_min_lane_speed": 22.0,
    # --- observation -----------------------------------------------------
    # the shield rewrites actions, so the policy must see what is allowed
    "action_mask_observation": True,
    # --- ego dynamics ----------------------------------------------------
    # ControlledVehicle has no acceleration clip at all, so a large speed
    # correction becomes an implausible deceleration that invites rear-ends.
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

def apply_overrides(overrides: list[str]) -> dict[str, object]:
    """Apply `KEY=VALUE` pairs to ENV_CONFIG and return what changed.

    Sweeping a single reward weight or controller constant from the command
    line keeps an experiment reproducible without editing this file. Numbers
    are coerced to int or float, anything else stays a string.
    """
    applied: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in ENV_CONFIG:
            raise KeyError(f"{key!r} is not a known ENV_CONFIG entry")
        value: object = raw.strip()
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                continue
        ENV_CONFIG[key] = value
        applied[key] = value
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
