# app/ (Workstream G)

`run.py` (repo root) runs the whole pipeline on one video; the Streamlit app wraps it for players.

```bash
pip install -r app/requirements.txt -r pipeline/pose/requirements.txt -r pipeline/overlay/requirements.txt

python run.py video.mp4 --player player.json --out out/<name>/
python run.py --reuse-pose --player player.json --out out/<name>/   # skip B, reuse pose/ball.json in out/<name>/
streamlit run app/streamlit_app.py
python -m pytest app/tests
```

## run.py

Calls B → C → D → E → F through their CLIs (`python -m pipeline.<stage>`). After each stage it runs
`contract/tools/validate.py` on every file written so far, so cross-file rules are checked as files are added.
It stops at the first failing stage or validation with `run.py: FAILED: stage X (...)` and exit code 1.

Writes into `--out`: `player.json` (copy), `pose.json`, `ball.json`, `reps.json`, `metrics.json`,
`overlays/<new>_vs_<old>.mp4|.png` for each rep vs. the one before it, and `cards.json` (+ `plan.json`) from F.
F is skipped while `pipeline/coach/__main__.py` doesn't exist; it will be called as
`python -m pipeline.coach --in metrics.json player.json --out cards.json`.

## The app

- **Sidebar:** player profile (`player.json` fields), validated against `contract/schemas/player.schema.json`.
- **New session:** upload a video and the date it was filmed; the app saves it to `out/<player>_<time>_<video>/`,
  runs `run.py` with live output, and stores the reps and metrics in `out/history.sqlite`.
  "Run on fixture" does the same with the contract's synthetic pose/ball data (no video needed).
- **Reps:** each rep tagged left/right and strong/weak; its metrics; the ghost overlay vs. the previous rep
  (the rep before it in the session, or the player's last rep from the previous session, rendered on demand);
  and the why-it-matters cards. Until F exists, cards come from `contract/examples/cards.example.json`,
  with a banner saying so.
- **Weak-foot dashboard:** per metric, the session mean for each foot over time against the target band,
  plus a summary table and a table view.

### Not-measured metrics

`targets.yaml` lists the camera views each metric can be measured from. The video's view is inferred from
which single-view metrics got values. A not_measured metric that needs the other view is shown as
"Film side-on to measure: …" or "Film front-on to measure: …"; one that should have been measurable from this
view (player not detected, no height for ball speed) is listed as "Couldn't measure on this clip".

### Overlay video

E writes `mp4v`, which browsers don't play. The app transcodes to H.264 (`*.web.mp4`) with ffmpeg
(system or `imageio-ffmpeg`) and always shows the contact-frame PNG.

`SOCCER_COACH_OUT` overrides the `out/` folder (the tests use it).
