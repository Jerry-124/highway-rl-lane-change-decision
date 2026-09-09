from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from highway_rl.train import _validate_training_args


def _args(**overrides) -> Namespace:
    values = {
        "total_timesteps": 10_000,
        "n_envs": 4,
        "n_steps": 256,
        "batch_size": 256,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "learning_rate": 1e-4,
        "n_epochs": 10,
        "checkpoint_every": 0,
        "seed": 42,
        "model_path": Path("models/test"),
        "resume_from": None,
        "mask_actions": True,
        "resume_algorithm": "ppo",
        "reset_critic": False,
        "log_dir": Path("logs/test"),
        "device": "cpu",
        "vec_env": "dummy",
        "set": [],
    }
    values.update(overrides)
    return Namespace(**values)


def test_valid_training_arguments_pass() -> None:
    _validate_training_args(_args())


def test_zero_environment_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n-envs"):
        _validate_training_args(_args(n_envs=0))


def test_nonfinite_learning_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="learning-rate"):
        _validate_training_args(_args(learning_rate=float("nan")))


def test_reset_critic_requires_resume_model() -> None:
    with pytest.raises(ValueError, match="resume-from"):
        _validate_training_args(_args(reset_critic=True))


def test_missing_resume_model_is_rejected(tmp_path) -> None:
    missing = tmp_path / "missing_model"

    with pytest.raises(FileNotFoundError, match="resume model not found"):
        _validate_training_args(_args(resume_from=missing))
