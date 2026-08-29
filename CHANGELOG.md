# Changelog

## v1.0.0 — 2026-08-29

- Released the lateral-only Maskable PPO baseline.
- Added rule-based longitudinal control and explicit acceleration limits.
- Added safety action masks and deterministic shield constraints.
- Matched the lane-change cooldown to the overtake window to prevent abandoned
  manoeuvres from being replaced by another lane change.
- Corrected decision-time versus post-update reward state handling.
- Corrected emergency speed control below the open-road cruise floor.
- Added balanced overtake outcome accounting and held-out validation/test
  reports.
- Removed obsolete five-action experiments and failed intermediate checkpoints
  from the public project layout.
