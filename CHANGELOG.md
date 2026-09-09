# Changelog

## [1.0.1] - 2026-09-09

### Changed

- Consolidated the post-v1.0.0 architecture cleanup that decomposed environment, training, and evaluation flows while preserving runtime evaluation provenance.
- Retained the existing regression coverage added during the architecture-hardening pass.
- Added MIT license/package metadata and repository-hygiene updates that landed after the v1.0.0 tag.
- Standardized the public README, model-card wording, package description, and version presentation.
- Aligned project and runtime package versions to 1.0.1.
- Made `--set KEY=VALUE` configuration overrides type-aware and transactional; boolean values are parsed explicitly and invalid multi-key updates leave the global configuration unchanged.
- Decoupled evaluation action masks from observation-vector layout by querying the environment mask interface directly.
- Added explicit validation for training counts, hyperparameter finiteness/ranges, batch geometry, resume artifacts, and critic-reset usage.
- Added evaluation guards for empty episode sets, invalid action-mask shapes, out-of-range actions, unknown overtake outcomes, and metadata collisions with computed metrics.
- Added automatic software/model provenance to newly generated evaluation summaries, including software version, algorithm, resolved model path, model SHA-256, seed provenance, and configuration overrides.
- Hardened `scripts/diagnose.py` so masked diagnostics query the environment mask interface rather than slicing observation tails, and made its random baseline locally seeded and reproducible.
- Kept a single seeded random-policy instance across `crash_audit.py` episodes so the baseline RNG advances continuously instead of restarting the same action sequence every episode.
- Added explicit diagnostic/audit argument validation and deterministic regression coverage for diagnostic RNG and action-mask semantics.
- Replaced scenario-dependent test skips with deterministic constructed traffic states for slow/stopped-leader, overtake-state, and cooldown regressions.
- Upgraded GitHub Actions to the current Node 24-compatible `actions/checkout@v7` and `actions/setup-python@v7` lines.
- Added source/test/script compilation, Ruff formatting checks, and CPU-only PyTorch installation to the CI regression matrix.

### Validation / Key Results

- The automated suite contains 40 pytest tests covering environment behavior, action masks, longitudinal control, reward timing, overtake accounting, configuration overrides, training-argument validation, evaluation provenance, diagnostic reproducibility, and result serialization.
- Scenario-sensitive regressions execute deterministically rather than being conditionally skipped when a sampled traffic scene is unsuitable.
- CI targets Python 3.10 and 3.12 and checks dependency consistency, compilation, pytest, Ruff linting, and Ruff formatting.
- The bundled `ppo_highway_v1.0.0.zip` model and its held-out validation/test artifacts are unchanged.
- No policy retraining, reward retuning, benchmark rerun, or historical-result rewrite is part of this patch.

### Scope

Version 1.0.1 is a software-maintenance, reproducibility, and presentation-consistency patch. The historical v1.0.0 model release remains the source of the reported 100-episode validation and 100-episode test results.

## [1.0.0] - 2026-08-29

### Added

- Released the lateral-only Maskable PPO highway lane-change baseline.
- Added deterministic longitudinal cruise, preparation, and safe-following control with explicit acceleration/deceleration limits.
- Added action masking, deterministic safety-shield constraints, and an 8-decision lane-change cooldown.
- Matched the lane-change cooldown to the 8-step overtake window to prevent active manoeuvres from being silently superseded.
- Added balanced overtake-outcome accounting and held-out validation/test result artifacts.

### Fixed

- Corrected decision-time versus post-update reward-state handling.
- Corrected emergency speed control when following traffic below the open-road cruise floor.
- Removed obsolete five-action experiments and failed intermediate checkpoints from the stable public project layout.

### Validation / Key Results

- Validation seeds 5000–5099: 1.0% collision rate, 99% episode completion, and 23.29 m/s average speed.
- Test seeds 9000–9099: 2.0% collision rate, 98% episode completion, and 23.13 m/s average speed.
- Strict overtake success was 51.6% on validation and 50.6% on test, below the 70% stretch target.
- Shield intervention and unavailable-action request rates were 0.0% on both held-out splits.

### Scope

The v1.0.0 tag records the evaluated model baseline and its original software state. Historical model weights and result artifacts are preserved unchanged.
