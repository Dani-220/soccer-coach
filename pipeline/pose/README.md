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
| Ball detection | Ultralytics YOLO, COCO class `sports ball` (`--ball-model yolo11m.pt --ball-conf 0.1 --ball-imgsz 1280`). Each box becomes `[cx, cy, (w+h)/4, conf]`. |
| Ball tracking | `ball_tracker.py`: one track, using a constant-velocity Kalman filter. Each frame takes the detection nearest the predicted position, if it falls inside a gate of `3r + n·(0.25·speed + kick·r/fps)` px. Here r is the ball radius, n the frames since the last hit, and kick `--ball-kick-speed` (120 radii/s ≈ 13 m/s). The gate scales with distance from the camera and grows with speed and misses, so a second ball elsewhere on the pitch isn't picked. With no track, the detection nearest the player's feet starts one (the most confident if no player is in frame). The track is dropped after `--ball-max-missing` (0.5 s). The Kalman filter only chooses; **the output is always the raw detection, never the prediction**. |
| Crop re-detect | When the tracked ball is missed, YOLO runs again on a window about 4 ball diameters across (min 64 px) around the prediction, upscaled to `--ball-imgsz`. Hits are mapped back to full-frame pixels and must still pass the gate. `--no-ball-redetect` turns it off. |
| Missing data | No player → `"keypoints": {}`. No ball → `"ball": null`. Nothing is interpolated or guessed; smoothing the ball path is a downstream job. Low-confidence and off-screen landmarks are kept with their confidence, as the contract requires. |
| Debug video | Skeleton coloured by confidence (green high, red low), a yellow ball circle with its confidence, a grey cross and circle for the tracker's predicted position and gate (debug only, never written to JSON), and the frame number with "no player" / "no ball" flags. |

## Tests

```bash
python -m pytest pipeline/pose -v
POSE_TEST_VIDEO=/path/to/real_clip.mp4 python -m pytest pipeline/pose -v   # also run on real footage
```

- `test_one_euro.py`: the filter smooths, follows fast motion, and resets.
- `test_extract.py`: the full pipeline with scripted fake detectors (gaps, missing ball, a second person,
  out-of-range confidence, the CLI and `--debug`). Every output goes through `contract/tools/validate.py`.
- `test_ball_tracker.py`: scripted ball detections. A stationary, more confident ball far away is never
  picked while the kicked ball is tracked, a 3-frame miss mid-flight keeps the track, every output is a
  real detection, the track drops after 0.5 s, and the crop window maps back to full-frame pixels.
- `test_models.py`: the real MediaPipe and YOLO on a generated video with and without a person
  (marker `models`; skip it with `-m "not models"`).

## Known limits

- Ball detection with stock COCO weights is weak for small or motion-blurred balls, especially in
  flight against trees and at contact. Expect `null` there, which is correct output. On
  `shots_clip1.mp4` (210 frames, ball visible in all), recall rose from 67.6% to 77.6%, and wrong-ball
  picks fell from 12 to 0. Almost every remaining miss is a frame where YOLO finds nothing ≥ 0.1, even
  on the upscaled crop (crops added 4 frames). Fine-tuned weights (`--ball-model path.pt`) are the next
  step.
- A hard cut between shots in an edited clip costs about 3 null frames. The old track's gate has to grow
  until it reaches the new ball, or the track drops. Continuous training footage doesn't have this.
- Smoothing defaults were set on synthetic signals, not on labelled footage. Tune
  `--min-cutoff`/`--beta` against real clips; check contact-frame lag with `--debug`.
- `player_track_id` is always `1`: one tracked player per video.
