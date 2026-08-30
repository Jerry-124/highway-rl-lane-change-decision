import csv
import json

import pytest

from highway_rl.config import ACTION_NAMES, ENV_CONFIG
from highway_rl.environment import OVERTAKE_OUTCOMES
from highway_rl.evaluate import EpisodeMetrics, save_results, summarize


def _row(*, steps: int, expected_steps: int) -> EpisodeMetrics:
    zeros = {name: 0.0 for name in ACTION_NAMES}
    return EpisodeMetrics(
        episode=1,
        collision=False,
        average_speed_mps=20.0,
        cumulative_reward=1.0,
        lane_changes=0,
        steps=steps,
        expected_steps=expected_steps,
        lane_changes_per_100_steps=0.0,
        shield_interventions=0,
        shield_unsafe_requests=0,
        shield_cooldown_requests=0,
        unavailable_action_requests=0,
        overtakes=0,
        overtake_attempts=0,
        overtake_outcomes={},
        overtake_superseded=0,
        keep_lane_share=1.0,
        crash_cause="none",
        action_mix=zeros.copy(),
        action_availability=zeros.copy(),
    )


def test_completion_uses_runtime_expected_steps_not_global_config(monkeypatch) -> None:
    """A custom environment must not be judged using stale module globals."""
    monkeypatch.setitem(ENV_CONFIG, "duration", 999)
    monkeypatch.setitem(ENV_CONFIG, "policy_frequency", 999)

    summary = summarize([_row(steps=10, expected_steps=10)])

    assert summary["episode_completion_rate"] == 1.0


def test_incomplete_episode_stays_incomplete() -> None:
    summary = summarize([_row(steps=9, expected_steps=10)])

    assert summary["episode_completion_rate"] == 0.0


def test_mixed_completion_rate_uses_each_episode_provenance() -> None:
    rows = [
        _row(steps=10, expected_steps=10),
        _row(steps=20, expected_steps=20),
        _row(steps=19, expected_steps=20),
    ]

    summary = summarize(rows)

    assert summary["episode_completion_rate"] == pytest.approx(2 / 3)


def test_summary_refactor_preserves_rates_and_overtake_accounting() -> None:
    first = _row(steps=10, expected_steps=10)
    second = _row(steps=20, expected_steps=20)
    outcome = OVERTAKE_OUTCOMES[0]

    first.shield_interventions = 3
    first.shield_unsafe_requests = 2
    first.shield_cooldown_requests = 1
    first.unavailable_action_requests = 1
    first.lane_changes = 2
    first.lane_changes_per_100_steps = 20.0
    first.overtakes = 1
    first.overtake_attempts = 2
    first.overtake_outcomes = {outcome: 1}
    first.overtake_superseded = 1
    first.keep_lane_share = 0.4

    second.shield_interventions = 1
    second.lane_changes = 1
    second.lane_changes_per_100_steps = 5.0
    second.overtake_attempts = 1
    second.overtake_outcomes = {outcome: 1}
    second.keep_lane_share = 0.8

    summary = summarize([first, second])

    assert summary["shield_intervention_rate"] == pytest.approx(4 / 30)
    assert summary["shield_unsafe_request_rate"] == pytest.approx(2 / 30)
    assert summary["shield_cooldown_request_rate"] == pytest.approx(1 / 30)
    assert summary["unavailable_action_request_rate"] == pytest.approx(1 / 30)
    assert summary["shield_interventions_per_lane_change"] == pytest.approx(4 / 3)
    assert summary["mean_keep_lane_share"] == pytest.approx(0.6)
    assert summary["overtake_success_rate"] == pytest.approx(1 / 3)
    assert summary["overtake_outcomes"][outcome] == 2
    assert summary["overtake_superseded"] == 1
    assert summary["overtake_attempts_resolved"] == 3
    assert summary["overtake_accounting_balanced"] is True


def test_save_results_refactor_preserves_csv_and_summary_schema(tmp_path) -> None:
    row = _row(steps=10, expected_steps=10)
    row.action_mix[ACTION_NAMES[0]] = 0.25
    row.action_availability[ACTION_NAMES[0]] = 0.75
    row.overtake_outcomes[OVERTAKE_OUTCOMES[0]] = 2

    summary = save_results([row], tmp_path, metadata={"audit_marker": "scenario2"})

    assert summary["episode_completion_rate"] == 1.0
    assert summary["audit_marker"] == "scenario2"
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary == summary

    with (tmp_path / "episodes.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert records[0]["steps"] == "10"
    assert records[0][f"share_{ACTION_NAMES[0]}"] == "0.25"
    assert records[0][f"available_{ACTION_NAMES[0]}"] == "0.75"
    assert records[0][f"outcome_{OVERTAKE_OUTCOMES[0]}"] == "2"
