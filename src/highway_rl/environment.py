"""Highway lane-change decision environment: PPO decides lateral, rules drive speed.

Scope
-----
The policy's action space is lateral only - keep lane, move left, move right.
Speed is produced by a fixed rule-based controller, so the degenerate
"press SLOWER forever" policy of earlier iterations is not expressible at all;
the safety problem it was covering for is handled below the policy instead.

Three layers, from the policy downwards:

1. **Safety shield** - vetoes lane changes into occupied lanes and blocks a
   second change while one is still settling (no weaving).
2. **Rule-based longitudinal controller** - cruises at 30 m/s, eases to 25 m/s
   while looking for a gap, matches and then backs off from a slow leader. Its
   commanded acceleration is clipped, because the stock ``ControlledVehicle``
   applies none and would otherwise brake far harder than the IDM traffic.
3. **Rewards** - safety, efficiency, and credit for *completed* overtakes, all
   keyed off one shared ``blocked`` signal so the terms cannot disagree.

Action availability is exposed in the observation: the shield rewrites actions,
and PPO must not keep crediting requests that were never executed.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import highway_env  # noqa: F401 - importing registers highway-env environments
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnvFast
from highway_env.vehicle.controller import ControlledVehicle

from highway_rl.config import ACTION_NAMES, ENV_CONFIG, ENV_ID

INF = float("inf")
N_ACTIONS = len(ACTION_NAMES)

# Everything the reward needs from *before* the physics update. Populated at
# decision time so the reward never has to guess which side of the step a
# cached value came from.
# How an armed overtake attempt can resolve. Emitted on the single step where
# the attempt terminates, so a caller counting outcomes counts *attempts* and
# not steps. The invariant is
#     attempts == sum(outcomes) + superseded
# which evaluate.py asserts on. An attempt that is still armed simply reports
# nothing this step; it resolves later as one of the outcomes below, or is
# superseded by a newer manoeuvre.
OVERTAKE_OUTCOMES = (
    "success",                  # cleared the leader by the required margin
    "expired",                  # the window ran out
    "returned_to_source_lane",  # merged back before completing the pass
    "crashed",
    "lost_leader",              # the leader left the road mid-attempt
    "episode_ended",            # still armed when the episode was cut off
)

EMPTY_DECISION_CTX: dict[str, object] = {
    "blocked": 0.0,
    "can_escape": False,
    "lane_change_requested": False,
    "valid_lane_change": False,
    "cooldown_active": False,
    "shield_intervened": False,
}


class HighwayLaneChangeEnv(HighwayEnvFast):
    """HighwayEnvFast with a lateral-only action space and a rule-based speed layer."""

    def __init__(self, *args, **kwargs) -> None:
        # AbstractEnv.__init__ calls reset(), so these must exist before super().
        self._previous_lane = 0
        self._shield_intervened = False
        self._shield_mode = 0
        self._lane_change_cooldown = 0
        self._overtake_leader = None
        self._overtake_blocked = 0.0
        self._overtake_source_lane = 0
        self._overtake_countdown = 0
        self._cache: dict = {}
        self._decision_ctx: dict = dict(EMPTY_DECISION_CTX)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # traffic context
    # ------------------------------------------------------------------

    def _leader(self):
        if "leader" in self._cache:
            return self._cache["leader"]
        front, _ = self.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        self._cache["leader"] = front
        return front

    def _forward_speed(self) -> float:
        return float(self.vehicle.speed * np.cos(self.vehicle.heading))

    def _front_gap(self) -> float:
        leader = self._leader()
        if leader is None:
            return INF
        return max(float(self.vehicle.lane_distance_to(leader)), 0.0)

    @staticmethod
    def _ttc(gap: float, closing_speed: float) -> float:
        """Time to collision; infinite when the gap is not closing."""
        if gap == INF or closing_speed <= 0.1:
            return INF
        return gap / closing_speed

    def _congestion(self) -> float:
        """0 when the road ahead is open, ramps to 1 at zero time headway."""
        if "congestion" in self._cache:
            return self._cache["congestion"]
        gap = self._front_gap()
        if gap == INF:
            value = 0.0
        else:
            time_headway = gap / max(self._forward_speed(), 1.0)
            safe = float(self.config["safe_time_headway"])
            value = float(np.clip((safe - time_headway) / safe, 0.0, 1.0))
        self._cache["congestion"] = value
        return value

    def _blocked(self) -> float:
        """How strongly a slow leader is holding the ego up, in [0, 1].

        Measured as catch-up time at the desired speed rather than by comparing
        speeds against a fixed threshold: a leader 80 m ahead going 25 does not
        stop us from reaching 30, a leader 30 m ahead going 25 does.

        Boundaries: no leader -> 0; leader at or above the desired speed -> 0;
        a negligible closing speed -> 0 (treated as an infinite catch-up time).
        """
        if "blocked" in self._cache:
            return self._cache["blocked"]
        leader = self._leader()
        value = 0.0
        if leader is not None:
            closing = float(self.config["desired_speed"]) - float(leader.speed)
            gap = self._front_gap()
            if closing > 0.1 and gap != INF:
                horizon = float(self.config["block_horizon"])
                catch_time = gap / closing
                if catch_time < horizon:
                    value = float(np.clip(1.0 - catch_time / horizon, 0.0, 1.0))
        self._cache["blocked"] = value
        return value

    def _lane_neighbours(self, lane_index):
        """(front vehicle, front gap, rear vehicle, rear gap) on `lane_index`."""
        key = ("lane", lane_index)
        if key in self._cache:
            return self._cache[key]
        lane = self.road.network.get_lane(lane_index)
        s_self = lane.local_coordinates(self.vehicle.position)[0]
        # inlined AbstractLane.on_lane: it would recompute local_coordinates
        # for every vehicle, and this loop runs several times per step
        half_width = float(lane.width_at(0.0)) / 2.0 + 1.0
        lane_length = float(getattr(lane, "length", INF))
        vehicle_length = float(getattr(lane, "VEHICLE_LENGTH", 5.0))
        front_vehicle = rear_vehicle = None
        front_gap = rear_gap = INF
        for other in self.road.vehicles:
            if other is self.vehicle:
                continue
            s_other, lat_other = lane.local_coordinates(other.position)
            if abs(lat_other) > half_width:
                continue
            if not (-vehicle_length <= s_other < lane_length + vehicle_length):
                continue
            delta = s_other - s_self
            if delta >= 0:
                if delta < front_gap:
                    front_gap, front_vehicle = delta, other
            elif -delta < rear_gap:
                rear_gap, rear_vehicle = -delta, other
        self._cache[key] = (front_vehicle, front_gap, rear_vehicle, rear_gap)
        return self._cache[key]

    # ------------------------------------------------------------------
    # rule-based longitudinal controller
    # ------------------------------------------------------------------

    def _desired_speed(self) -> float:
        """Cruise, ease off while looking for a gap, then follow safely.

        This is the layer that makes "just brake forever" unnecessary: the ego
        always drives as fast as the leader allows, and the policy only has to
        decide whether to go around.
        """
        cruise = float(self.config["cruise_speed"])
        leader = self._leader()
        if leader is None:
            return cruise

        gap = self._front_gap()
        speed = self._forward_speed()
        leader_speed = float(leader.speed)
        ttc = self._ttc(gap, speed - leader_speed)
        time_headway = gap / max(speed, 1.0)

        desired = cruise
        # Ease off to the prepare speed while *hunting* for a gap - but not once
        # a pass is under way. Holding 25 m/s through the attempt caps the
        # closing speed at roughly 4 m/s, so the ego never actually gets past:
        # measured over 37 attempts, blocked was active on 74% of the chase and
        # the median attempt finished 2.2 m still behind the car it set out to
        # pass. The prepare speed is for looking, not for completing.
        if self._blocked() > 0.0 and self._overtake_leader is None:
            desired = min(desired, float(self.config["prepare_speed"]))
        # safe following: match the leader, then drop below it as we close in
        if ttc < float(self.config["follow_ttc"]):
            desired = min(desired, leader_speed)
        if time_headway < float(self.config["safe_time_headway"]):
            desired = min(desired, leader_speed - 3.0)
        if ttc < float(self.config["emergency_ttc"]):
            desired = min(desired, leader_speed - 6.0)
        # The floor is a stall guard for an open road, not a cruise floor. If the
        # following terms above have already pulled the target below it, clipping
        # back up would drive the ego into a slow or stopped leader instead of
        # matching it, so in that case the floor is dropped entirely. Without
        # this the ego cannot come to a stop or follow anything under
        # `min_cruise_speed`, no matter how small the gap gets.
        floor = float(self.config["min_cruise_speed"])
        if desired < floor:
            floor = 0.0
        return float(np.clip(desired, floor, cruise))

    # ------------------------------------------------------------------
    # safety shield
    # ------------------------------------------------------------------

    def _target_lane_index(self, action: int):
        """Lane index the given lateral action would select, or None if invalid."""
        indexes = self.action_type.actions_indexes
        if action not in (indexes["LANE_LEFT"], indexes["LANE_RIGHT"]):
            return None
        offset = -1 if action == indexes["LANE_LEFT"] else 1
        _from, _to, lane_id = self.vehicle.target_lane_index
        lane_count = len(self.road.network.graph[_from][_to])
        candidate = (_from, _to, int(np.clip(lane_id + offset, 0, lane_count - 1)))
        if candidate[2] == lane_id:
            return None
        if not self.road.network.get_lane(candidate).is_reachable_from(
            self.vehicle.position
        ):
            return None
        return candidate

    def _lane_change_safe(self, candidate) -> bool:
        """Target lane must be clear ahead, with no slow vehicle to merge behind.

        Rear traffic is judged geometrically only - a minimum clearance so we do
        not merge onto a car sitting beside us. Rear speed and rear TTC are
        deliberately ignored: following vehicles run IDM and brake for us.
        """
        if candidate is None:
            return False
        # never start a second manoeuvre while one is still in progress
        if int(self.vehicle.target_lane_index[2]) != int(self.vehicle.lane_index[2]):
            return False
        front_vehicle, front_gap, _rear_vehicle, rear_gap = self._lane_neighbours(candidate)
        if front_gap < float(self.config["shield_gap_front"]):
            return False
        # pure geometry: never move into a vehicle alongside or just behind
        if rear_gap < float(self.config["shield_rear_geometry"]):
            return False
        if front_vehicle is None:
            return True
        ego_speed = self._forward_speed()
        if self._ttc(front_gap, ego_speed - float(front_vehicle.speed)) < float(
            self.config["shield_ttc"]
        ):
            return False
        # merging behind a slow vehicle defeats the purpose of the manoeuvre
        if front_gap < float(self.config["shield_lookahead"]):
            if float(front_vehicle.speed) < float(self.config["shield_min_lane_speed"]):
                return False
        return True

    def _shield(self, action: int) -> int:
        """Veto unsafe lateral actions, recording why in `self._shield_mode`.

        Modes: 0 = untouched, 1 = target lane not clear, 2 = still cooling down
        from the previous change.
        """
        self._shield_mode = 0
        if not self.config.get("shield_enabled", True):
            return action

        keep = self.action_type.actions_indexes["IDLE"]
        if action == keep:
            return action
        if self._lane_change_cooldown > 0:
            self._shield_mode = 2
            return keep
        if not self._lane_change_safe(self._target_lane_index(action)):
            self._shield_mode = 1
            return keep
        return action

    def action_mask(self) -> np.ndarray:
        """Per-action availability under the shield, for the observation."""
        saved_mode = self._shield_mode
        mask = np.array(
            [1.0 if self._shield(action) == action else 0.0 for action in range(N_ACTIONS)],
            dtype=np.float32,
        )
        self._shield_mode = saved_mode
        return mask

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _capture_decision_context(self, action: int) -> None:
        """Snapshot the decision-time facts the reward needs, before _simulate.

        Called once the shield has settled the action, so `shield_intervened` is
        already final. `can_escape` lives here rather than in `_rewards` so that
        both halves of `blocked_keep_penalty` describe the same moment - the one
        at which the policy chose to hold its lane - instead of one half being
        read from a cache and the other recomputed after the world moved on.
        """
        indexes = self.action_type.actions_indexes
        blocked = self._blocked()
        self._decision_ctx = {
            "blocked": blocked,
            # each check scans the whole traffic, so only when it can matter
            "can_escape": blocked > 0.0
            and any(
                self._lane_change_safe(self._target_lane_index(indexes[direction]))
                for direction in ("LANE_LEFT", "LANE_RIGHT")
            ),
            "lane_change_requested": action in (
                indexes["LANE_LEFT"], indexes["LANE_RIGHT"]
            ),
            "valid_lane_change": self._target_lane_index(action) is not None,
            "cooldown_active": self._lane_change_cooldown > 0,
            "shield_intervened": self._shield_intervened,
        }

    def _rewards(self, action: int) -> dict[str, float]:
        """Reward terms, each pinned to an explicit side of the physics update.

        A reward of the form r(s, a, s') legitimately reads both states, but the
        choice has to be deliberate rather than an artefact of what happens to
        sit in the cache. The split used here:

        * **decision time (s)** - was the action legal, was the ego blocked with
          a safe escape available, did the shield have to step in. These judge
          the *choice*, so they must not see the outcome. Read from
          `_decision_ctx`.
        * **post-update (s')** - collision, speed, headway, whether the lane
          actually changed, which lane we ended up in. These judge the *result*,
          so they are read live, after the cache is cleared.
        """
        # The cache was filled before _simulate ran. Drop it so the result terms
        # below are genuinely measured on the state we ended up in, rather than
        # quietly served pre-update leaders and gaps.
        self._cache.clear()
        ctx = self._decision_ctx

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        forward_speed = self._forward_speed()
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        # result term: how tight the following situation became
        congestion = self._congestion()
        # decision term: how strongly we were being held up when we chose
        blocked = float(ctx["blocked"])

        indexes = self.action_type.actions_indexes
        keep = indexes["IDLE"]
        lane_change_requested = bool(ctx["lane_change_requested"])
        valid_lane_change = bool(ctx["valid_lane_change"])
        lane_change_completed = int(self.vehicle.lane_index[2]) != self._previous_lane
        can_escape = bool(ctx["can_escape"])

        return {
            "collision_reward": float(self.vehicle.crashed),
            "alive_reward": 1.0,
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "lane_change_reward": float(valid_lane_change),
            "completed_lane_change_reward": float(lane_change_completed),
            "invalid_lane_change_penalty": float(
                lane_change_requested and not valid_lane_change
            ),
            "unnecessary_lane_change_penalty": float(
                lane_change_requested and valid_lane_change and ctx["cooldown_active"]
            ),
            # held the lane while blocked and a safe lane change was available
            "blocked_keep_penalty": blocked * float(can_escape and action == keep),
            # asking the shield for something it had to override
            "shield_violation_penalty": float(ctx["shield_intervened"]),
            "headway_penalty": congestion,
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(
            float(self.config.get(name, 0.0)) * value
            for name, value in rewards.items()
            if name != "on_road_reward"
        )
        return reward * rewards["on_road_reward"]

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._cache.clear()
        observation, info = super().reset(**kwargs)
        self._install_ego_dynamics()
        self._previous_lane = int(self.vehicle.lane_index[2])
        self._shield_intervened = False
        self._shield_mode = 0
        self._decision_ctx = dict(EMPTY_DECISION_CTX)
        self._lane_change_cooldown = 0
        self._overtake_leader = None
        self._overtake_blocked = 0.0
        self._overtake_source_lane = int(self.vehicle.lane_index[2])
        self._overtake_countdown = 0
        return observation, info

    def _install_ego_dynamics(self) -> None:
        """Soften the speed tracker and, crucially, clip its acceleration.

        ControlledVehicle applies no acceleration limit at all, so a large speed
        correction turns into an implausible deceleration - far harder than the
        3.0 m/s^2 the IDM traffic brakes at, which is what invites rear-ends.
        """
        vehicle = self.vehicle
        vehicle.KP_A = float(self.config.get("ego_kp_accel", ControlledVehicle.KP_A))
        max_accel = float(self.config["ego_max_accel"])
        max_decel = float(self.config["ego_max_decel"])
        unclipped = type(vehicle).speed_control

        def clipped_speed_control(target_speed):
            return float(
                np.clip(unclipped(vehicle, target_speed), -max_decel, max_accel)
            )

        # instance attribute shadows the bound method for this vehicle only
        vehicle.speed_control = clipped_speed_control

    def step(self, action: int):
        self._cache.clear()
        # context captured before the physics update: the overtake event needs
        # to know who the leader was when the manoeuvre started
        blocked_before = self._blocked()
        leader_before = self._leader()
        lane_before = int(self.vehicle.lane_index[2])
        target_before = int(self.vehicle.target_lane_index[2])

        # longitudinal control is not learned: ask the rule layer for a speed
        self.vehicle.target_speed = self._desired_speed()

        requested = int(action)
        applied = self._shield(requested)
        self._shield_intervened = applied != requested
        # freeze the decision-time facts before the physics update moves the world
        self._capture_decision_context(applied)

        observation, reward, terminated, truncated, info = super().step(applied)

        lane_after = int(self.vehicle.lane_index[2])
        target_after = int(self.vehicle.target_lane_index[2])
        overtake_attempt_started = False
        overtake_superseded = False

        if target_after != target_before:
            # a manoeuvre was initiated: remember what we are trying to pass.
            # The pass itself only happens a second or two after the lane
            # change completes, so the attempt stays armed for a few steps.
            # An attempt still armed at this point never resolved - this new
            # manoeuvre replaced it, which is its own outcome and not a success.
            if self._overtake_leader is not None:
                overtake_superseded = True
                self._overtake_leader = None
            # Only a manoeuvre that set out to pass someone counts as an
            # attempt. Arming on an unblocked lane change would let such a
            # manoeuvre resolve as "success" with no matching attempt on the
            # denominator, and with a zero bonus because _overtake_blocked is 0.
            if blocked_before > 0.0 and leader_before is not None:
                self._overtake_leader = leader_before
                self._overtake_blocked = blocked_before
                self._overtake_source_lane = lane_before
                self._overtake_countdown = int(self.config["overtake_window"])
                overtake_attempt_started = True

        if lane_after != lane_before:
            self._lane_change_cooldown = int(self.config["lane_change_cooldown"])
        elif self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1

        overtake_bonus = 0.0
        overtake_outcome = None
        if self._overtake_leader is not None:
            leader = self._overtake_leader
            # the leader can leave the road while we are passing it; the attempt
            # is then unresolvable rather than failed
            leader_gone = not any(other is leader for other in self.road.vehicles)
            passed = (
                not leader_gone
                and lane_after != self._overtake_source_lane
                and float(self.vehicle.position[0]) - float(leader.position[0])
                > float(self.config["overtake_margin"])
            )
            if passed and not self.vehicle.crashed:
                overtake_bonus = (
                    float(self.config["overtake_reward"]) * self._overtake_blocked
                )
                overtake_outcome = "success"
                self._overtake_leader = None
            elif self.vehicle.crashed:
                overtake_outcome = "crashed"
                self._overtake_leader = None
            elif lane_after == self._overtake_source_lane:
                # merged back before completing the pass
                overtake_outcome = "returned_to_source_lane"
                self._overtake_leader = None
            elif leader_gone:
                overtake_outcome = "lost_leader"
                self._overtake_leader = None
            else:
                self._overtake_countdown -= 1
                if self._overtake_countdown <= 0:
                    overtake_outcome = "expired"
                    self._overtake_leader = None
                elif terminated or truncated:
                    # ran out of episode, not out of time
                    overtake_outcome = "episode_ended"
                    self._overtake_leader = None
                # still armed: deliberately report nothing. Emitting a
                # "pending" outcome here would make a caller that counts
                # outcomes once per step count steps, not attempts.

        # Applied here rather than inside _rewards so the credit survives even
        # when the episode ends on the completing step.
        reward += overtake_bonus

        self._previous_lane = lane_after
        info["shield_intervened"] = self._shield_intervened
        info["shield_mode"] = self._shield_mode
        info["requested_action"] = requested
        info["applied_action"] = applied
        info["overtake_bonus"] = overtake_bonus
        info["overtake_outcome"] = overtake_outcome
        info["overtake_superseded"] = overtake_superseded
        info["overtake_attempt_started"] = overtake_attempt_started
        info["blocked"] = blocked_before
        info["target_speed"] = float(self.vehicle.target_speed)
        return observation, reward, terminated, truncated, info


class ActionMaskObservation(gym.ObservationWrapper):
    """Append the shield's action-availability mask to the observation.

    Without this the policy cannot tell which actions the shield would actually
    let through, so it keeps taking credit for actions that were never executed.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.observation_space
        size = int(np.prod(base.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(size + N_ACTIONS,), dtype=np.float32
        )

    def observation(self, observation):
        flat = np.asarray(observation, dtype=np.float32).ravel()
        return np.concatenate([flat, self.env.unwrapped.action_mask()]).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        """Boolean mask consumed by sb3-contrib's MaskablePPO."""
        return self.env.unwrapped.action_mask().astype(bool)


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=HighwayLaneChangeEnv)


def make_env(*, render_mode: str | None = None, seed: int | None = None) -> gym.Env:
    """Create the configured custom highway environment."""
    env = gym.make(ENV_ID, config=ENV_CONFIG, render_mode=render_mode)
    if ENV_CONFIG.get("action_mask_observation", True):
        env = ActionMaskObservation(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
