# pipeline/pose — Workstream B

Turns a training video into `pose.json` and `ball.json` in the format set by contract v1.0
(`contract/CONTRACT.md`, `contract/schemas/`).

```bash
pip install -r pipeline/pose/requirements.txt          # Python 3.11
python -m pipeline.pose --in clip.mp4 --out out/pose.json --debug
python contract/tools/validate.py out/                  # OK for pose, ball, cross-file
```

This writes `out/pose.json`, `out/ball.json` (next to `--out`; `clip_pose.json` → `clip_ball.json`;
override with `--ball-out`) and, with `--debug`, `out/clip_debug.mp4` (or `--debug path.mp4`).
Run it from the repo root. Model weights download on first use into `~/.cache/soccer-coach/models`
(override with `SOCCER_COACH_MODELS`).

## What it does

| Step | How |
| --- | --- |
| Frames | OpenCV decodes every frame, so `f` runs 0..N-1 with no gaps. Size comes from the decoded frame, so rotated phone video gets the right width and height. `--fps` overrides a bad container fps. |
| Pose | MediaPipe Tasks `PoseLandmarker`, VIDEO mode, 33 landmarks (heel and foot_index included). The `heavy` model is the default; `--pose-model lite/full/<path.task>` picks another. Confidence = landmark visibility. |
| One player | MediaPipe detects up to `--max-poses` people. The tracker keeps the one whose hip midpoint is nearest the last frame's. A jump bigger than `--max-jump` body heights counts as a different person, so that frame gets `{}` instead of a switch to the wrong player. After 15 frames without the player, the tracker takes the largest person again. |
| Smoothing | One Euro filter on every keypoint's x and y (`--min-cutoff 1.5 --beta 0.01 --d-cutoff 1.0`, in px; `--no-smooth` turns it off). The filter **resets whenever the player is missing**, so it never fills a gap. Confidence is not smoothed. |
| Ball | Ultralytics YOLO, COCO class `sports ball` (`--ball-model yolo11m.pt --ball-conf 0.25 --ball-imgsz 1280`). The highest-confidence box becomes `[cx, cy, (w+h)/4, conf]`. |
| Missing data | No player → `"keypoints": {}`. No ball → `"ball": null`. Nothing is interpolated or guessed; smoothing the ball path is a downstream job. Low-confidence and off-screen landmarks are kept with their confidence, as the contract requires. |
| Debug video | Skeleton coloured by confidence (green high, red low), a yellow ball circle with its confidence, and the frame number with "no player" / "no ball" flags. |

## Tests

```bash
python -m pytest pipeline/pose -v
POSE_TEST_VIDEO=/path/to/real_clip.mp4 python -m pytest pipeline/pose -v   # also run on real footage
```

- `test_one_euro.py`: the filter smooths, follows fast motion, and resets.
- `test_extract.py`: the full pipeline with scripted fake detectors (gaps, missing ball, a second person,
  out-of-range confidence, the CLI and `--debug`). Every output goes through `contract/tools/validate.py`.
- `test_models.py`: the real MediaPipe and YOLO on a generated video with and without a person
  (marker `models`; skip it with `-m "not models"`).

## Known limits

- Ball detection with stock COCO weights is weak for small or motion-blurred balls, especially at
  contact. Expect `null` there, which is correct output. Raise `--ball-imgsz` or use fine-tuned
  weights (`--ball-model path.pt`) if recall is too low on real footage.
- Smoothing defaults were set on synthetic signals, not on labelled footage. Tune
  `--min-cutoff`/`--beta` against real clips; check contact-frame lag with `--debug`.
- `player_track_id` is always `1`: one tracked player per video.
