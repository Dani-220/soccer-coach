# pipeline/overlay (Workstream E)

Compares two reps: the **new** rep is drawn solid, the **old** rep as a faint ghost. Each joint is
colored green to red by its distance from the ghost, and the 3 most deviant joints are labeled.
Writes `overlay.mp4` plus `overlay.png` at the new rep's contact frame.

```bash
pip install -r contract/requirements.txt -r pipeline/overlay/requirements.txt

# run from the repo root
python -m pipeline.overlay --in contract/examples/pose.example.json contract/examples/reps.example.json \
    --new r002 --old r001 --out out/overlay.mp4                       # plain background
python -m pipeline.overlay --in pose.json reps.json --background video --video clip.mp4 --out out/overlay.mp4
python -m pipeline.overlay --in new_pose.json new_reps.json old_pose.json old_reps.json --out out/overlay.mp4

python -m pytest pipeline/overlay/tests
```

`--in` takes `POSE REPS` (both reps from one video) or `POSE REPS OLD_POSE OLD_REPS` (the first pair
holds the new rep). If `--out` ends in `.png`, only the contact PNG is written.
See `python -m pipeline.overlay -h` for all options.

## How it works

1. **Normalize** (`skeleton.py`). Each frame is centered on the hip midpoint and divided by the rep's
   median torso length (shoulder midpoint to hip midpoint). Facing is detected from hip motion, and x
   is flipped so both reps move in +x. Frames with no player (`{}`) are interpolated with confidence 0,
   so they carry no weight and are not drawn.
2. **Foot match.** If the reps use different feet, the ghost is relabeled left<->right, so the kicking
   leg is compared with the kicking leg (shown as `[mirrored L/R]`; turn off with `--no-swap`).
3. **Align** (`align.py`). The contact frames are pinned together. `[start, contact]` and
   `[contact, end]` are each aligned with DTW, using confidence-weighted mean joint distance as the
   cost. The ghost for each new frame is the mean of its matched old frames.
4. **Render** (`render.py`, OpenCV). Deviation = joint distance in torso lengths; it is fully red at
   `--max-dev` (default 0.5). On `--background video`, the ghost is placed on the new player's hips
   and torso scale in the new rep's own video (pose coords are rescaled if the video resolution
   differs). Output defaults to 30 fps (slow motion for 120 fps sources) with a 0.5 s hold at contact.
