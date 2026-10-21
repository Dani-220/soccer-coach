# Soccer Coach Pipeline Contract v1.0

Paste this file into every parallel chat. It replaces the "Shared contract" section of the build plan.
**Chats must not change this contract.** If a chat needs a new field or move, it asks the integrator,
who bumps `schema_version`, updates `schemas/`, and pastes the change into every affected chat.

## Pipeline

```
video.mp4 ──B──> pose.json + ball.json ──C──> reps.json ──D──> metrics.json ──F──> cards.json + plan.json
                                                    └────────E (two reps) ──> overlay.mp4 / overlay.png
player.json (from the app, G) feeds C (strong foot), D (height) and F (level, goals).
```

Each stage is a CLI: `python -m pipeline.<stage> --in <file(s)> --out <file>`.
Formal JSON Schemas live in `schemas/`; `python tools/validate.py <folder>` checks a set of files.

## Conventions

| Topic | Rule |
| --- | --- |
| Coordinates | Pixels in the original video, origin top-left, y grows downward. |
| Time | Integer frame index `f`, 0..N-1 with no gaps; seconds = f / fps. |
| Keypoint names | MediaPipe Pose names. 17 core points are required whenever a player is detected: `nose`, then `left_`/`right_` + `shoulder, elbow, wrist, hip, knee, ankle, heel, foot_index`. Other MediaPipe points are optional. |
| Keypoint value | `[x, y, confidence]`. Keep low-confidence points; never omit a core point. |
| No player | `"keypoints": {}`. No ball: `"ball": null`. Never invent or interpolate in B's output. |
| Feet | Always `"left"` or `"right"` in data. Strong/weak is derived from `player.json` at display time. |
| Body height (H) | Vertical px from `nose` to the lower ankle, median over the first 30 frames with a player. All distances are divided by H (`unit: body_heights`). Convert to cm with `H_cm = height_cm × 0.93`. |
| Angles | Degrees. `trunk_lean`: shoulder-midpoint vs. hip-midpoint from vertical, **positive = leaning forward**. |
| Facing | The player moves in +x (left to right) in fixture videos. Code must not assume this; detect direction from hip motion. |
| Unmeasurable metrics | Metrics needing a front camera (`plant_foot_angle`, `hip_shoulder_separation`) are output with `value: null`, `status: "not_measured"` from side-on video. |
| Moves | `shot_instep` for v1. Reserved: `shot_inside`, `pass_inside`, `dribble`, `stepover`, `body_feint`, `tackle_block`. |
| IDs | `rep_id` = `r001`, `r002`, ... in time order within a video. `player_id` is a short string. |

## Files

| File | Writer | Key contents |
| --- | --- | --- |
| `pose.json` | B | `video, fps, width, height, frames[{f, keypoints}]` |
| `ball.json` | B | `frames[{f, ball: [x, y, radius, conf] or null}]` |
| `reps.json` | C | `reps[{rep_id, move, foot, frames{start, plant, contact, end}, confidence}]` |
| `metrics.json` | D | `reps[{rep_id, foot, metrics[{name, value, unit, target, status, frame}]}]` |
| `targets.yaml` | D (config) | Target range, unit, camera views and "why" per metric. Placeholder values until calibrated. |
| `cards.json` | F | 0 to 3 cards per rep: `metric, headline, measured_value, target, why, fix, drill_ids, frame` |
| `plan.json` | F | One week: `focus, weak_foot_share, days[{day, drills[{drill_id, sets, reps, foot}]}], retest_on` |
| `drills.yaml` | F (config) | Drill library: `id, name, fixes[metrics], levels, equipment, setup, coaching_points` |
| `player.json` | G | `player_id, strong_foot, level, height_cm, goals` |
| `labels.csv` | A (you) | Ground truth: `video, rep_id, move, foot, plant_frame, contact_frame, coach_score_1to5, notes` |

## Rules that the validator enforces beyond the schemas

- `start <= plant <= contact <= end`, and every frame index exists in `pose.json`.
- `pose.json` and `ball.json` have the same video, fps and frame count.
- A metric's `status` agrees with its `value` and `target`.
- Every card's `measured_value` and `target` are **copied exactly** from `metrics.json`, and cards are only written for `out_of_range` metrics. The LLM writes words, never numbers.

## Fixture data

`examples/` holds a synthetic side-on video's data (396 frames, 120 fps):

| Rep | Foot | Frames (start, plant, contact, end) | What's wrong |
| --- | --- | --- | --- |
| r001 | right (strong) | 0, 96, 103, 143 | Nothing: every measurable metric in range. Use as the "good" reference rep. |
| r002 | left (weak) | 228, 324, 331, 371 | Plant foot far from ball, knee behind ball, leaning back, loose ankle, short follow-through, slow ball. |

Frames 168 to 227 have no player (walking back). The ball is `null` at r002's contact (hidden by the foot)
and after it leaves frame. Pose has ~1.8 px noise; feet have lower confidence, as in real detectors.
`metrics.example.json` holds the **true** values measured on the noise-free skeleton, so Workstream D can
test its computed metrics against them.
