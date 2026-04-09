# Cartesian Validation Report

- Timestamp (UTC): 2026-04-09T13:11:58.761488+00:00
- Expected combos: 1536
- Expected fold evals: 4608
- Metrics rows: 4608
- Completed ok: 4392
- Skipped or failed: 216
- Completed ok (%): 95.31%
- Figures found: 11
- Mode: strict full run
- Runtime (sec): 5294.26
- Runtime (HH:MM:SS): 01:28:14
- Execution device: cpu
- Acceleration backend: none
- Checkpoint percent: 5
- Run label: run2_20260409T131110Z
- Platform profile: mac
- Started (UTC): 2026-04-09T13:11:14.222657+00:00
- Finished (UTC): 2026-04-09T13:11:14.328287+00:00
- Validation runtime (sec): 0.021

Validation result: PASS

## Top Skip Reasons
- selection_failed: ValueError: Found array with 1 feature(s) (shape=(700, 1)) while a minimum of 2 is required by SequentialFeatureSelector.: 72
- selection_failed: ValueError: n_features_to_select must be < n_features.: 72
- selection_failed: ValueError: Found array with 1 feature(s) (shape=(7667, 1)) while a minimum of 2 is required by RFE.: 48
- selection_failed: ValueError: Found array with 1 feature(s) (shape=(7666, 1)) while a minimum of 2 is required by RFE.: 24