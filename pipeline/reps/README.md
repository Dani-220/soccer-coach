# pipeline/reps (Workstream C)

`pose.json` + `ball.json` (+ optional `player.json`) → `reps.json`, per `contract/CONTRACT.md`.

```bash
python -m pipeline.reps --in pose.json ball.json player.json --out reps.json [--config reps.yaml]
python -m unittest pipeline.reps.tests.test_reps -v     # run from the repo root
```

## How each field is found

| Field | Rule |
| --- | --- |
| `contact` | Last frame the ball is at rest (all detections in `rest_window_s` within `rest_radius` H) before its speed jumps above `jump_speed` H/s, while a foot touches the ball within `touch_window_s` of contact. "Touches" is measured to the ball's **surface** (radius from ball.json): the ankle→toe segment within `foot_ball_max_gap` (0.08) H when `foot_index` confidence ≥ `min_toe_conf`, else the ankle alone within `ankle_ball_max_gap` (0.15) H, because the ball is struck a foot-length ahead of the ankle. If the foot hides the ball (up to `max_ball_gap_s`), contact is back-projected from the ball's flight speed. If the detector loses the ball for good, contact is the first frame a foot moving at ≥ `vanish_min_foot_speed` H/s touches it, unless the ball reappears at its resting spot within `vanish_confirm_s` (a dropout). |
| `foot` | Combines "nearest the ball at contact" (toe segments when both toes are detected, else ankles) and "fastest ankle over `foot_speed_window_s` before contact" into a signed score. `strong_foot` from player.json only adds a small `strong_foot_prior` to break near-ties. Foot confidence = 0.5 + 0.5·\|score\|. |
| `plant` | Walking back from contact, the first frame of the support ankle's final stationary run, tolerating `still_max_outliers` noisy samples. "Stationary" = within a radius of its median position over contact − `plant_ref_window_s` … contact + `plant_jitter_window_s`; the radius is the larger of `plant_still_radius` H and `plant_jitter_mult` × the jitter measured in that window (capped at `plant_still_radius_max`), so it widens on small, jittery players. Omitted if the support ankle isn't detected. |
| `end` | First frame after contact from which the kicking ankle stays within `end_still_radius` H for `end_hold_s`. Capped at `max_follow_through_s` or the end of the player's visible segment. |
| `start` | `contact - approach_s`, clamped to the player's visible segment and to after the previous rep's end. |
| `confidence` | Foot confidence × `missing_phase_penalty` for each of plant/end not found × `vanish_penalty` if the ball was never seen leaving. |

All thresholds are fields of `RepsConfig` in [config.py](config.py) (distances in body heights, speeds in H/s,
times in seconds, so they don't depend on resolution or fps). A YAML file passed with `--config` overrides any subset;
unknown keys are an error.

## Accuracy on test data

- Example fixtures: contact, start, end and foot exact on both reps; plant 2 frames (17 ms) early, because the
  synthetic support ankle settles within pose noise before its labeled plant frame.
- Synthetic sweep (tests/synthetic.py, real strike geometry with the ankle ~0.17 H from the ball's center, 1.8 px noise,
  30/60/120 fps, both directions, ball occluded at contact, toe confident or not):
  foot and contact always right, plant within 25 ms. End comes out 30–80 ms early when the foot slows smoothly to a
  stop, because the last few frames move less than the stillness radius.

Defaults were set on synthetic data only; retune them on real labeled footage (`labels.csv`).
- Known limit: the ball's at-rest check (`rest_radius` 0.03 H) is fixed, like plant used to be. Ball jitter above
  ~2 px on a 200 px player starts to drop whole reps; raise `rest_radius` if that shows up on real footage.
