"""Train and save the PPO baseline."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from highway_rl.config import PPO_CONFIG, apply_overrides
from highway_rl.environment import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=PPO_CONFIG["gamma"])
    parser.add_argument("--ent-coef", type=float, default=PPO_CONFIG["ent_coef"])
    parser.add_argument("--learning-rate", type=float, default=PPO_CONFIG["learning_rate"])
    parser.add_argument("--n-epochs", type=int, default=PPO_CONFIG["n_epochs"])
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save an intermediate model every N steps (0 disables checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=Path("models/ppo_highway"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--mask-actions",
        action="store_true",
        help="use MaskablePPO so unavailable lateral actions cannot be sampled",
    )
    parser.add_argument(
        "--resume-algorithm",
        choices=("ppo", "maskable-ppo"),
        default="ppo",
        help="algorithm stored in --resume-from; PPO weights can be transferred "
             "into a new MaskablePPO optimizer",
    )
    parser.add_argument(
        "--reset-critic",
        action="store_true",
        help="when transferring weights, reinitialise the value network and keep "
             "the learned actor. Use this after a reward-semantics change: the "
             "old critic encodes returns that can no longer occur, so every "
             "advantage estimate would start out biased.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="subproc",
        help="subproc runs environments in separate processes (needs this to "
             "actually use multiple cores); dummy is easier to debug.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="override ENV_CONFIG entries for this run, e.g. "
             "--set overtake_reward=5.0 blocked_keep_penalty=-0.25",
    )
    return parser.parse_args()


def reset_critic(model) -> None:
    """Reinitialise the value path and its optimizer state, keeping the actor.

    The actor is a reasonable initialisation regardless of how rewards were
    defined. The critic is not: it predicts returns under the *previous* reward,
    and after a semantics change those returns cannot occur any more, which
    biases every advantage estimate from the first update onwards. Carrying the
    Adam moments for parameters that were just re randomised would be worse
    than having no moments at all, so those are dropped too.
    """
    init = partial(type(model.policy).init_weights, gain=1.0)
    model.policy.mlp_extractor.value_net.apply(init)
    model.policy.value_net.apply(init)
    optimizer = getattr(model.policy, "optimizer", None)
    if optimizer is not None:
        optimizer.state.clear()
    print("reset critic: value network reinitialised, optimizer moments dropped, "
          "actor weights kept")


def main() -> None:
    args = parse_args()
    overrides = apply_overrides(args.set)
    if overrides:
        print(f"config overrides: {overrides}")
    # Stable-Baselines3 resolves "auto" to CUDA when available, otherwise CPU.
    device = args.device
    if (args.n_steps * args.n_envs) % args.batch_size != 0:
        raise ValueError("batch-size must divide n-steps * n-envs")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    # make_vec_env defaults to DummyVecEnv, which steps environments in the
    # same process: adding environments then adds no parallelism at all.
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(args.log_dir),
        vec_env_cls=vec_env_cls,
    )
    try:
        if args.resume_from and args.mask_actions and args.resume_algorithm == "ppo":
            # Keep the source representation and action/value heads, but reset
            # the optimizer under MaskablePPO. The two policy classes use
            # the same network parameter layout; only action sampling differs.
            source = PPO.load(args.resume_from, device=device)
            model = MaskablePPO(
                "MlpPolicy",
                vec_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=PPO_CONFIG["gae_lambda"],
                clip_range=PPO_CONFIG["clip_range"],
                ent_coef=args.ent_coef,
                policy_kwargs=PPO_CONFIG["policy_kwargs"],
                seed=args.seed,
                device=device,
                verbose=1,
            )
            model.policy.load_state_dict(source.policy.state_dict(), strict=True)
            if args.reset_critic:
                reset_critic(model)
            model.num_timesteps = source.num_timesteps
        elif args.resume_from:
            algorithm = MaskablePPO if args.resume_algorithm == "maskable-ppo" else PPO
            model = algorithm.load(args.resume_from, env=vec_env, device=device)
            model.gamma = args.gamma
            model.ent_coef = args.ent_coef
            model.learning_rate = args.learning_rate
            model.lr_schedule = FloatSchedule(args.learning_rate)
            for parameter_group in model.policy.optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
            model.verbose = 1
            if args.reset_critic:
                reset_critic(model)
        else:
            algorithm = MaskablePPO if args.mask_actions else PPO
            model = algorithm(
                "MlpPolicy",
                vec_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=PPO_CONFIG["gae_lambda"],
                clip_range=PPO_CONFIG["clip_range"],
                ent_coef=args.ent_coef,
                policy_kwargs=PPO_CONFIG["policy_kwargs"],
                seed=args.seed,
                device=device,
                verbose=1,
            )
        callbacks = []
        if args.checkpoint_every > 0:
            # CheckpointCallback counts VecEnv steps, not individual env steps,
            # so the requested interval has to be divided by the env count.
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(args.checkpoint_every // args.n_envs, 1),
                    save_path=str(args.log_dir / "checkpoints"),
                    name_prefix="ppo_step",
                    save_vecnormalize=False,
                )
            )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks or None,
            progress_bar=False,
            reset_num_timesteps=not bool(args.resume_from),
        )
        model.save(args.model_path)
        print(f"Saved PPO model to {args.model_path.with_suffix('.zip')}")
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
