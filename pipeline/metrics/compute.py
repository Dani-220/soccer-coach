"""Build metrics.json from pose.json, ball.json, reps.json (+ player.json) and targets.yaml."""
import sys
from pathlib import Path

import yaml

from . import instep
from .context import RepContext, Video

SCHEMA_VERSION = "1.0"
DEFAULT_TARGETS = Path(__file__).resolve().parent / "targets.yaml"
MOVE_METRICS = {"shot_instep": instep.METRICS}


def load_targets(path=DEFAULT_TARGETS):
    return yaml.safe_load(Path(path).read_text())


def round_value(value, unit):
    """Body-height metrics keep 3 decimals, everything else 1 (status is judged on the rounded value)."""
    return round(float(value), 3 if unit == "body_heights" else 1)


def metric_entry(name, value, frame, target_cfg, view):
    """One metrics.json entry. Metrics whose `views` exclude this camera view are not_measured."""
    unit, rng = target_cfg["unit"], [target_cfg["range"][0], target_cfg["range"][1]]
    if value is None or view not in target_cfg.get("views", ["side", "front"]):
        return {"name": name, "value": None, "unit": unit, "target": rng, "status": "not_measured", "frame": int(frame)}
    v = round_value(value, unit)
    status = "in_range" if rng[0] <= v <= rng[1] else "out_of_range"
    return {"name": name, "value": v, "unit": unit, "target": rng, "status": status, "frame": int(frame)}


def compute_metrics(pose, ball, reps, player=None, targets=None, view=None):
    """Return the metrics.json document. `view` ('side'|'front') overrides auto-detection."""
    targets = targets or load_targets()
    height_cm = (player or {}).get("height_cm")
    if height_cm is None:
        print("metrics: no player height; ball_speed will be not_measured", file=sys.stderr)

    video = Video.load(pose, ball, view=view)

    out_reps = []
    for rep in reps["reps"]:
        fns = MOVE_METRICS.get(rep["move"])
        if fns is None or rep["move"] not in targets:
            print(f"metrics: skipping {rep['rep_id']} ({rep['move']} has no metrics yet)", file=sys.stderr)
            continue
        ctx = RepContext.build(video, rep, height_cm=height_cm)
        move_targets = targets[rep["move"]]
        mets = []
        for name, fn in fns.items():
            if name not in move_targets:
                continue
            value, frame = fn(ctx)
            mets.append(metric_entry(name, value, frame, move_targets[name], video.view))
        out_reps.append({"rep_id": rep["rep_id"], "move": rep["move"], "foot": rep["foot"], "metrics": mets})

    return {"schema_version": SCHEMA_VERSION, "video": reps["video"], "player_id": reps["player_id"], "reps": out_reps}
