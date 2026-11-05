# eval/ — Workstream H

Scores the pipeline's output against hand labels (`labels.csv`, contract v1.1).

```bash
python -m eval --labels fixtures/labels.csv --runs out/
python -m eval --labels contract/examples/labels.example.csv --runs contract/examples --no-history   # smoke test
```

`--runs` is searched recursively for `pose`, `ball`, `reps` and `metrics` JSON. Files are recognised by
name (the contract validator's rule: `reps.json`, `clip1_reps.json`, `reps.example.json`, ...) and
grouped by the `video` field inside them (by base name), so any folder layout works. If a video has two
files of the same type, the newest is used and a warning is printed. Videos in `labels.csv` with no run
count every labeled rep as missed. Runs with no labels are listed and skipped.

## What it reports

One row per labeled video, plus a pooled `ALL` row (pooled over reps and frames, not averaged over videos):

| Column | Meaning |
| --- | --- |
| lab / det / hit | Labeled reps, detected reps, matched pairs. A detected rep matches a labeled one when their contact frames are within `--match-tol` seconds (default 0.25 s). Pairing is one-to-one and minimises total contact error. |
| prec / recall | hit / det and hit / lab. |
| contact MAE, bias | Mean absolute and mean signed contact-frame error over matched reps, in frames (positive = detected late). `report.json` also has the median, the max and the same numbers for the plant frame. |
| foot | Share of matched reps where the kicking foot agrees with the label. |
| coach ρ (n) | Spearman correlation between `coach_score_1to5` and a metric-based score, over matched reps with a coach score. The metric-based score is the share of the rep's measured metrics that are `in_range` in `metrics.json` (`not_measured` is left out). Needs at least 3 pairs and some variation on both sides; `—` otherwise. |
| ball recall | Share of frames within `--ball-window` seconds (default 0.25 s) of each labeled contact where `ball.json` has a ball. Windows that overlap are counted once. This measures the detector against ground truth, independent of rep detection. |

Missed labels and false positives are printed by rep_id under the table.

## Files written

- `eval/report.json` (`--report`): the settings, the git commit, every number above per video and
  pooled, and per-rep detail (`matches`, `missed_labels`, `false_positives`). Git-ignored.
- `eval/history.jsonl` (`--history`): one line per run with the commit, the time, pooled and per-video
  ball recall, and rep precision, recall and contact MAE. Each run prints how ball recall changed since
  the last entry made with the same `--match-tol` and `--ball-window`. Commit this file if you want the
  record shared; `--no-history` skips it.

## CI

`.github/workflows/ci.yml` runs on every push and pull request (Python 3.11 and 3.13). It installs
every `requirements.txt` in the repo except `mediapipe`, `ultralytics` and `opencv-python`
(`opencv-python-headless` instead), runs `python -m pytest -m "not models"` and then
`python contract/tools/validate.py contract/examples/`. Tests that download model weights must be
marked `models` (see `pipeline/pose/tests`) or skip themselves when those packages are missing.

## Tests

```bash
python -m pytest eval -v
```
