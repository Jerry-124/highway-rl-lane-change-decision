from __future__ import annotations

import copy

import pytest

from highway_rl.config import ENV_CONFIG, apply_overrides


@pytest.fixture(autouse=True)
def restore_env_config():
    original = copy.deepcopy(ENV_CONFIG)
    try:
        yield
    finally:
        ENV_CONFIG.clear()
        ENV_CONFIG.update(original)


def test_boolean_override_is_parsed_as_boolean() -> None:
    applied = apply_overrides(["shield_enabled=false"])

    assert applied == {"shield_enabled": False}
    assert ENV_CONFIG["shield_enabled"] is False


def test_invalid_boolean_does_not_mutate_config() -> None:
    before = copy.deepcopy(ENV_CONFIG)

    with pytest.raises(ValueError, match="true or false"):
        apply_overrides(["shield_enabled=disabled"])

    assert ENV_CONFIG == before


def test_multiple_overrides_are_transactional() -> None:
    before = copy.deepcopy(ENV_CONFIG)

    with pytest.raises(ValueError, match="ego_max_decel"):
        apply_overrides(["overtake_reward=3.0", "ego_max_decel=-1"])

    assert ENV_CONFIG == before


def test_list_override_uses_json_and_is_validated() -> None:
    applied = apply_overrides(["reward_speed_range=[18, 32]"])

    assert applied["reward_speed_range"] == [18, 32]
    assert ENV_CONFIG["reward_speed_range"] == [18, 32]


def test_nonfinite_override_is_rejected() -> None:
    before = copy.deepcopy(ENV_CONFIG)

    with pytest.raises(ValueError, match="finite"):
        apply_overrides(["shield_ttc=nan"])

    assert ENV_CONFIG == before
