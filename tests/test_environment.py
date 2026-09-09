import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from highway_rl.config import ACTION_NAMES, ENV_CONFIG
from highway_rl.environment import OVERTAKE_OUTCOMES, ActionMaskObservation, make_env

KEEP_LANE = ACTION_NAMES.index("KEEP_LANE")


def _place_leader(unwrapped, *, gap_m: float, speed_mps: float):
    """Place a traffic vehicle deterministically ahead of the ego in its lane."""
    unwrapped._cache.clear()
    leader = unwrapped._leader()
    if leader is None:
        candidates = [
            vehicle
            for vehicle in unwrapped.road.vehicles
            if vehicle is not unwrapped.vehicle
        ]
        assert candidates, "configured environment should contain traffic vehicles"
        leader = candidates[0]

    lane = unwrapped.road.network.get_lane(unwrapped.vehicle.lane_index)
    ego_s = float(lane.local_coordinates(unwrapped.vehicle.position)[0])
    leader_s = ego_s + gap_m
    leader.position = np.asarray(lane.position(leader_s, 0.0), dtype=float)
    leader.heading = float(lane.heading_at(leader_s))
    leader.speed = float(speed_mps)
    if hasattr(leader, "target_speed"):
        leader.target_speed = float(speed_mps)

    unwrapped._cache.clear()
    assert unwrapped._leader() is leader
    return leader


def test_environment_api() -> None:
    env = make_env(seed=0)
    try:
        check_env(env, warn=True)
        observation, info = env.reset(seed=0)
        assert observation.shape == env.observation_space.shape
        assert isinstance(info, dict)
        step = env.step(env.action_space.sample())
        assert len(step) == 5
        rewards = env.unwrapped._rewards(
            env.unwrapped.action_type.actions_indexes["LANE_LEFT"]
        )
        assert rewards["alive_reward"] == 1.0
        assert rewards["lane_change_reward"] in {0.0, 1.0}
        assert rewards["invalid_lane_change_penalty"] in {0.0, 1.0}
        assert 0.0 <= rewards["headway_penalty"] <= 1.0
        assert 0.0 <= rewards["blocked_keep_penalty"] <= 1.0
        assert 0.0 <= rewards["shield_violation_penalty"] <= 1.0
    finally:
        env.close()


def test_action_space_is_lateral_only() -> None:
    env = make_env(seed=0)
    try:
        assert env.action_space.n == len(ACTION_NAMES) == 3
        # highway-env names the keep-lane meta-action IDLE
        indexes = env.unwrapped.action_type.actions_indexes
        assert set(indexes) == {"LANE_LEFT", "IDLE", "LANE_RIGHT"}
        assert indexes["IDLE"] == KEEP_LANE
    finally:
        env.close()


def test_action_mask_in_observation() -> None:
    env = make_env(seed=0)
    try:
        base = (
            env.env.observation_space
            if isinstance(env, ActionMaskObservation)
            else None
        )
        assert base is not None, "observation should be wrapped with the action mask"
        observation, _ = env.reset(seed=0)
        assert observation.shape == (int(np.prod(base.shape)) + len(ACTION_NAMES),)
        mask = observation[-len(ACTION_NAMES) :]
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        # keeping the lane is never blocked: it is always the safe fallback
        assert mask[KEEP_LANE] == 1.0
    finally:
        env.close()


def test_masked_actions_are_executed_as_requested() -> None:
    """An action flagged available by the mask must reach the vehicle unchanged."""
    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        for _ in range(40):
            mask = env.env.unwrapped.action_mask()
            available = np.flatnonzero(mask > 0.5)
            for action in available:
                assert env.env.unwrapped._shield(int(action)) == action
            _obs, _reward, terminated, truncated, _info = env.step(int(available[0]))
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


def test_blocked_uses_catch_up_time() -> None:
    """The blocked signal must follow catch-up time, not a fixed speed gap."""
    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        for _ in range(30):
            env.step(env.action_space.sample())
            blocked = env.env.unwrapped._blocked()
            leader = env.env.unwrapped._leader()
            assert 0.0 <= blocked <= 1.0
            if leader is None or float(leader.speed) >= ENV_CONFIG["desired_speed"]:
                assert blocked == 0.0, "no leader, or one we cannot catch: not blocked"
            if env.env.unwrapped.vehicle.crashed:
                env.reset()
    finally:
        env.close()


def test_speed_is_rule_based_not_learned() -> None:
    """The controller must back off as the gap to the leader tightens."""
    env = make_env(seed=0)
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        for _ in range(60):
            env.step(KEEP_LANE)
            leader = unwrapped._leader()
            if leader is None:
                continue
            gap = unwrapped._front_gap()
            speed = unwrapped._forward_speed()
            if gap / max(speed, 1.0) < ENV_CONFIG["safe_time_headway"]:
                assert unwrapped._desired_speed() <= float(leader.speed), (
                    "too close to the leader: the controller must back off"
                )
            if unwrapped.vehicle.crashed:
                env.reset()
    finally:
        env.close()


def test_deceleration_is_clipped() -> None:
    """A large speed correction must not become an implausible brake command."""
    env = make_env(seed=0)
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        command = unwrapped.vehicle.speed_control(ENV_CONFIG["min_cruise_speed"])
        assert command >= -ENV_CONFIG["ego_max_decel"]
        assert (
            unwrapped.vehicle.speed_control(ENV_CONFIG["cruise_speed"])
            <= (ENV_CONFIG["ego_max_accel"])
        )
    finally:
        env.close()


def test_controller_can_follow_a_stopped_leader() -> None:
    """The cruise floor must not override the following terms."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        _place_leader(unwrapped, gap_m=6.0, speed_mps=0.0)
        desired = unwrapped._desired_speed()
        assert desired <= 1.0, (
            f"a stopped leader 6 m ahead must pull the target to a standstill, "
            f"got {desired:.2f} m/s (min_cruise_speed="
            f"{ENV_CONFIG['min_cruise_speed']})"
        )
    finally:
        env.close()


def test_controller_can_follow_a_slow_leader() -> None:
    """Following a leader below the cruise floor must not be clipped back up."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        slow = float(ENV_CONFIG["min_cruise_speed"]) - 4.0
        _place_leader(unwrapped, gap_m=10.0, speed_mps=slow)
        desired = unwrapped._desired_speed()
        assert desired <= slow, (
            f"following a {slow:.1f} m/s leader must target at most that speed, "
            f"got {desired:.2f} m/s"
        )
    finally:
        env.close()


def test_prepare_speed_does_not_hold_during_a_pass() -> None:
    """The prepare speed is for hunting a gap, not for completing a pass."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        leader = _place_leader(unwrapped, gap_m=60.0, speed_mps=18.0)
        unwrapped.vehicle.speed = 25.0
        unwrapped._cache.clear()
        assert unwrapped._blocked() > 0.0, "constructed scene must be blocked"

        unwrapped._cache.clear()
        hunting = unwrapped._desired_speed()
        unwrapped._overtake_leader = leader
        unwrapped._cache.clear()
        passing = unwrapped._desired_speed()

        assert passing > hunting, (
            f"arming an overtake must release the prepare-speed cap: "
            f"hunting={hunting:.2f} m/s, passing={passing:.2f} m/s"
        )
    finally:
        env.close()


def test_reward_terms_are_pinned_to_known_timepoints() -> None:
    """Decision terms read the snapshot; result terms read the post-update state."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        env.step(KEEP_LANE)
        ctx = unwrapped._decision_ctx
        rewards = unwrapped._rewards(KEEP_LANE)
        assert rewards["blocked_keep_penalty"] == pytest.approx(
            float(ctx["blocked"]) * float(ctx["can_escape"])
        )
        unwrapped._cache.clear()
        assert rewards["headway_penalty"] == pytest.approx(unwrapped._congestion())
    finally:
        env.close()


def test_overtake_outcome_is_reported() -> None:
    """Every armed attempt must deterministically resolve to a named outcome."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        leader = _place_leader(unwrapped, gap_m=60.0, speed_mps=18.0)
        source_lane = int(unwrapped.vehicle.lane_index[2])
        unwrapped._overtake_leader = leader
        unwrapped._overtake_source_lane = source_lane
        unwrapped._overtake_countdown = 1

        bonus, outcome = unwrapped._resolve_overtake(
            source_lane + 1,
            terminated=False,
            truncated=False,
        )

        assert bonus == 0.0
        assert outcome == "expired"
        assert outcome in OVERTAKE_OUTCOMES
    finally:
        env.close()


def test_overtake_outcomes_are_counted_per_attempt_not_per_step() -> None:
    """Superseded plus resolved attempts must balance exactly."""
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        leader = _place_leader(unwrapped, gap_m=60.0, speed_mps=18.0)
        source_lane = int(unwrapped.vehicle.lane_index[2])
        target_lane = source_lane + 1

        started_1, superseded_1 = unwrapped._register_overtake_attempt(
            target_before=source_lane,
            target_after=target_lane,
            blocked_before=1.0,
            leader_before=leader,
            lane_before=source_lane,
        )
        started_2, superseded_2 = unwrapped._register_overtake_attempt(
            target_before=source_lane,
            target_after=target_lane,
            blocked_before=1.0,
            leader_before=leader,
            lane_before=source_lane,
        )
        assert started_1 and not superseded_1
        assert started_2 and superseded_2

        unwrapped._overtake_countdown = 1
        _bonus, outcome = unwrapped._resolve_overtake(
            target_lane,
            terminated=False,
            truncated=False,
        )
        assert outcome in OVERTAKE_OUTCOMES
        assert outcome != "pending"

        attempts = int(started_1) + int(started_2)
        superseded = int(superseded_1) + int(superseded_2)
        resolved = int(outcome is not None)
        assert resolved + superseded == attempts
    finally:
        env.close()


@pytest.mark.slow
def test_keep_lane_baseline_is_safe() -> None:
    """The rule layer alone must already drive safely without any lane changes."""
    env = make_env()
    crashes = 0
    episodes = 20
    try:
        for episode in range(episodes):
            env.reset(seed=3000 + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                _obs, _reward, terminated, truncated, _info = env.step(KEEP_LANE)
            crashes += int(env.env.unwrapped.vehicle.crashed)
    finally:
        env.close()
    assert crashes / episodes <= 0.10, (
        f"keep-lane baseline crashed {crashes}/{episodes}"
    )


def test_overtake_bonus_is_recorded() -> None:
    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        for _ in range(40):
            _obs, _reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            assert "overtake_bonus" in info
            assert "requested_action" in info
            assert "applied_action" in info
            if info["shield_intervened"]:
                assert info["requested_action"] != info["applied_action"]
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


def test_lane_change_cooldown_blocks_immediate_return() -> None:
    """Cooldown must veto any immediate second lateral action deterministically."""
    env = make_env(seed=0)
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        left = unwrapped.action_type.actions_indexes["LANE_LEFT"]
        right = unwrapped.action_type.actions_indexes["LANE_RIGHT"]
        unwrapped._lane_change_cooldown = 1

        assert unwrapped._shield(left) == KEEP_LANE
        assert unwrapped._shield_mode == 2
        assert unwrapped._shield(right) == KEEP_LANE
        assert unwrapped._shield_mode == 2
    finally:
        env.close()
