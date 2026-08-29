import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from highway_rl.config import ACTION_NAMES, ENV_CONFIG
from highway_rl.environment import OVERTAKE_OUTCOMES, ActionMaskObservation, make_env

KEEP_LANE = ACTION_NAMES.index("KEEP_LANE")


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
        base = env.env.observation_space if isinstance(env, ActionMaskObservation) else None
        assert base is not None, "observation should be wrapped with the action mask"
        observation, _ = env.reset(seed=0)
        assert observation.shape == (int(np.prod(base.shape)) + len(ACTION_NAMES),)
        mask = observation[-len(ACTION_NAMES):]
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
            _obs, _r, terminated, truncated, _i = env.step(int(available[0]))
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
        assert unwrapped.vehicle.speed_control(ENV_CONFIG["cruise_speed"]) <= (
            ENV_CONFIG["ego_max_accel"]
        )
    finally:
        env.close()


def test_controller_can_follow_a_stopped_leader() -> None:
    """The cruise floor must not override the following terms.

    `min_cruise_speed` guards against stalling on an open road. Once a leader
    has pulled the target below it, clipping back up drives the ego into a slow
    or stopped vehicle instead of matching it - the exact opposite of what the
    floor is documented to do.
    """
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        leader = unwrapped._leader()
        if leader is None:
            pytest.skip("no leader directly ahead on this seed")
        # park a stopped vehicle a car length or so in front of the ego
        leader.speed = 0.0
        leader.position[0] = unwrapped.vehicle.position[0] + 6.0
        leader.position[1] = unwrapped.vehicle.position[1]
        unwrapped._cache.clear()
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
        leader = unwrapped._leader()
        if leader is None:
            pytest.skip("no leader directly ahead on this seed")
        slow = float(ENV_CONFIG["min_cruise_speed"]) - 4.0  # 6 m/s
        leader.speed = slow
        leader.position[0] = unwrapped.vehicle.position[0] + 10.0
        leader.position[1] = unwrapped.vehicle.position[1]
        unwrapped._cache.clear()
        desired = unwrapped._desired_speed()
        assert desired <= slow, (
            f"following a {slow:.1f} m/s leader must target at most that speed, "
            f"got {desired:.2f} m/s"
        )
    finally:
        env.close()


def test_prepare_speed_does_not_hold_during_a_pass() -> None:
    """The prepare speed is for hunting a gap, not for completing a pass.

    Easing off to 25 m/s while an overtake is under way caps the closing speed
    at roughly 4 m/s, so the ego never gets past the car it set out to overtake:
    measured over 37 attempts, blocked was active on 74% of the chase and the
    median attempt finished still 2.2 m behind.
    """
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        leader = unwrapped._leader()
        if leader is None:
            pytest.skip("no leader directly ahead on this seed")
        # Far enough that no following constraint applies, close enough that the
        # leader still holds us up (catch-up time well inside block_horizon).
        leader.speed = 18.0
        leader.position[0] = unwrapped.vehicle.position[0] + 60.0
        leader.position[1] = unwrapped.vehicle.position[1]
        unwrapped.vehicle.speed = 25.0
        unwrapped._cache.clear()
        if unwrapped._blocked() <= 0.0:
            pytest.skip("constructed scene is not blocked")

        # hunting for a gap: ease off to the prepare speed
        unwrapped._cache.clear()
        hunting = unwrapped._desired_speed()
        # a pass is armed: the same scene must no longer be held down
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
    """Decision terms read the snapshot; result terms read the post-update state.

    The bug this guards against: `_rewards` used to read `blocked` from a cache
    filled before `_simulate` while recomputing `can_escape` on the moved world,
    so the two halves of one penalty described different moments.
    """
    env = make_env()
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        env.step(KEEP_LANE)
        ctx = unwrapped._decision_ctx
        rewards = unwrapped._rewards(KEEP_LANE)
        # blocked_keep_penalty: both halves come from the decision-time snapshot
        assert rewards["blocked_keep_penalty"] == pytest.approx(
            float(ctx["blocked"]) * float(ctx["can_escape"])
        )
        # headway_penalty: a result term, measured after the world moved
        unwrapped._cache.clear()
        assert rewards["headway_penalty"] == pytest.approx(unwrapped._congestion())
    finally:
        env.close()


def test_overtake_outcome_is_reported() -> None:
    """Every armed attempt must resolve to a named outcome, not just vanish."""
    env = make_env()
    unwrapped = env.env.unwrapped
    left = unwrapped.action_type.actions_indexes["LANE_LEFT"]
    right = unwrapped.action_type.actions_indexes["LANE_RIGHT"]
    seen: set[str] = set()
    try:
        for episode in range(10):
            env.reset(seed=3000 + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                # change lane whenever held up: this is what arms an attempt
                if unwrapped._blocked() > 0.0:
                    lane = int(unwrapped.vehicle.lane_index[2])
                    action = left if lane < 3 else right
                else:
                    action = KEEP_LANE
                _obs, _r, terminated, truncated, info = env.step(action)
                outcome = info.get("overtake_outcome")
                if outcome is not None:
                    assert outcome in OVERTAKE_OUTCOMES, f"unknown outcome {outcome}"
                    seen.add(outcome)
        if not seen:
            pytest.skip("no overtake attempt armed within the sampled episodes")
    finally:
        env.close()


def test_overtake_outcomes_are_counted_per_attempt_not_per_step() -> None:
    """One outcome per attempt, never per step.

    Emitting an outcome on every step an attempt stays armed makes the
    distribution a count of steps: a 100-episode run reported 1736 "pending"
    against 247 attempts, so the outcome shares could not be read as
    percentages at all.
    """
    env = make_env()
    unwrapped = env.env.unwrapped
    left = unwrapped.action_type.actions_indexes["LANE_LEFT"]
    right = unwrapped.action_type.actions_indexes["LANE_RIGHT"]
    attempts = 0
    superseded = 0
    outcomes: dict[str, int] = {}
    try:
        for episode in range(10):
            env.reset(seed=3000 + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                if unwrapped._blocked() > 0.0:
                    lane = int(unwrapped.vehicle.lane_index[2])
                    action = left if lane < 3 else right
                else:
                    action = KEEP_LANE
                _obs, _r, terminated, truncated, info = env.step(action)
                attempts += int(bool(info.get("overtake_attempt_started", False)))
                superseded += int(bool(info.get("overtake_superseded", False)))
                outcome = info.get("overtake_outcome")
                if outcome is not None:
                    assert outcome in OVERTAKE_OUTCOMES, f"unknown outcome {outcome}"
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if not attempts:
            pytest.skip("no overtake attempt armed within the sampled episodes")
        assert "pending" not in outcomes, "pending must not be reported as an outcome"
        assert sum(outcomes.values()) + superseded == attempts, (
            f"accounting does not balance: {sum(outcomes.values())} outcomes + "
            f"{superseded} superseded != {attempts} attempts"
        )
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
                _obs, _r, terminated, truncated, _i = env.step(KEEP_LANE)
            crashes += int(env.env.unwrapped.vehicle.crashed)
    finally:
        env.close()
    # the whole point of moving speed control out of the policy: this has to
    # be dramatically better than the ~95% crash rate of unconstrained driving
    assert crashes / episodes <= 0.10, f"keep-lane baseline crashed {crashes}/{episodes}"


def test_overtake_bonus_is_recorded() -> None:
    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        for _ in range(40):
            _obs, _r, terminated, truncated, info = env.step(
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
    """Changing back right after a change must be flagged as unnecessary."""
    env = make_env(seed=0)
    unwrapped = env.env.unwrapped
    try:
        env.reset(seed=0)
        for _ in range(60):
            left = unwrapped.action_type.actions_indexes["LANE_LEFT"]
            right = unwrapped.action_type.actions_indexes["LANE_RIGHT"]
            previous_lane = int(unwrapped.vehicle.lane_index[2])
            _obs, _r, terminated, truncated, _i = env.step(left)
            if int(unwrapped.vehicle.lane_index[2]) != previous_lane:
                assert unwrapped._shield(right) == KEEP_LANE, (
                    "the shield must block an immediate change back"
                )
                return
            if terminated or truncated:
                env.reset()
        pytest.skip("no lane change completed within the sampled steps")
    finally:
        env.close()
