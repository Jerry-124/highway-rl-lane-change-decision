import numpy as np
import pytest

from scripts.crash_audit import RandomPolicy as AuditRandomPolicy
from scripts.crash_audit import _validate_audit_args
from scripts.diagnose import RandomPolicy as DiagnoseRandomPolicy
from scripts.diagnose import _current_action_mask, _validate_rollout_args


def test_diagnose_mask_comes_from_environment_interface() -> None:
    class Unwrapped:
        @staticmethod
        def action_mask():
            return np.array([True, False, True])

    class Env:
        unwrapped = Unwrapped()

    assert _current_action_mask(Env()).tolist() == [True, False, True]


def test_diagnose_prefers_wrapper_mask_interface() -> None:
    class Unwrapped:
        @staticmethod
        def action_mask():
            return np.array([True, True, True])

    class Env:
        unwrapped = Unwrapped()

        @staticmethod
        def action_masks():
            return np.array([False, True, False])

    assert _current_action_mask(Env()).tolist() == [False, True, False]


def test_diagnostic_random_policy_is_reproducible() -> None:
    expected_rng = np.random.default_rng(17)
    expected = [int(expected_rng.integers(3)) for _ in range(12)]

    policy = DiagnoseRandomPolicy(seed=17)
    actual = [policy.predict(None)[0] for _ in range(12)]

    assert actual == expected


def test_crash_audit_random_policy_is_reproducible_and_advances() -> None:
    expected_rng = np.random.default_rng(23)
    expected = [int(expected_rng.integers(3)) for _ in range(12)]

    policy = AuditRandomPolicy(seed=23)
    actual = [policy.predict(None)[0] for _ in range(12)]

    assert actual == expected


def test_diagnostic_argument_validation_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="episodes"):
        _validate_rollout_args(0, 1)
    with pytest.raises(ValueError, match="seed"):
        _validate_rollout_args(1, -1)
    with pytest.raises(ValueError, match="episodes"):
        _validate_audit_args(0, 1)
    with pytest.raises(ValueError, match="seed"):
        _validate_audit_args(1, -1)
