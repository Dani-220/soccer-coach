"""Workstream D CLI: python -m pipeline.metrics --in pose.json ball.json reps.json [player.json] --out metrics.json

Input files can be given in any order; each one's type is recognised from its contents.
player.json supplies height_cm for ball_speed. Without it, ball_speed is not_measured.
"""
import argparse
import json
import sys
from pathlib import Path

from .compute import DEFAULT_TARGETS, compute_metrics, load_targets


def kind_of(doc):
    """'pose' | 'ball' | 'reps' | 'player' | None, from the document's own fields."""
    if "reps" in doc:
        return "reps"
    if "height_cm" in doc or "strong_foot" in doc:
        return "player"
    frames = doc.get("frames")
    if isinstance(frames, list):
        if "width" in doc and "height" in doc and (not frames or "keypoints" in frames[0]):
            return "pose"
        if frames and "ball" in frames[0]:
            return "ball"
        if frames and "keypoints" in frames[0]:
            return "pose"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.metrics", description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="pose.json ball.json reps.json [player.json]")
    ap.add_argument("--out", required=True, help="metrics.json to write")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS), help="target ranges YAML (default: %(default)s)")
    ap.add_argument("--view", choices=["side", "front"], help="camera view; auto-detected when omitted")
    a = ap.parse_args(argv)

    docs = {}
    for p in a.inputs:
        doc = json.loads(Path(p).read_text())
        kind = kind_of(doc)
        if kind is None:
            ap.error(f"can't tell what {p} is (expected pose, ball, reps or player JSON)")
        if kind in docs:
            ap.error(f"two {kind} files given")
        docs[kind] = doc
    missing = [k for k in ("pose", "ball", "reps") if k not in docs]
    if missing:
        ap.error(f"missing input(s): {', '.join(missing)}")

    out = compute_metrics(docs["pose"], docs["ball"], docs["reps"], docs.get("player"),
                          targets=load_targets(a.targets), view=a.view)
    Path(a.out).write_text(json.dumps(out, indent=2))
    n = sum(len(r["metrics"]) for r in out["reps"])
    print(f"metrics: wrote {n} metrics for {len(out['reps'])} reps to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
