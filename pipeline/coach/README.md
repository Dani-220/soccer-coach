# pipeline/coach (Workstream F)

```bash
python -m pipeline.coach --in metrics.json player.json --out out/cards.json   # also writes out/plan.json
python contract/tools/validate.py metrics.json player.json out/
python -m pytest pipeline/coach                                               # no API key needed
```

- **Cards**: the 1-2 worst `out_of_range` metrics per rep (by distance outside the target, in target widths;
  `ball_speed` only when no technique metric is out). Reps with nothing out of range get `cards: []`.
  `measured_value`, `target`, `unit` and `frame` are copied from metrics.json; `compare_to_strong_foot` is the
  strong-foot reps' mean for that metric. Headlines come from a fixed table.
- **Wording**: with `ANTHROPIC_API_KEY` in the environment or in `.env` at the repo root, `why` and `fix` come
  from Claude (`COACH_MODEL`, default `claude-opus-5-5`), given only the metric name, value, unit, target and
  the "why" text from `pipeline/metrics/targets.yaml`. A reply with any digit, over the length limit, or a failed
  call falls back to the template. `--no-llm` forces the template.
- **Plan**: up to 3 focus metrics by total severity, drills from `drills.yaml` for the player's level, days from
  `sessions_per_week` (default 3), re-test filmed on the last day. `weak_foot_share` is 0.5, rising to 0.8 as the
  weak foot's share of out-of-range metrics exceeds the strong foot's.
- **Drills**: `drills.yaml` (35 drills). `fixes` lists contract metric names, most-targeted first.
