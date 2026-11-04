"""CLI:  python -m pipeline.reps --in pose.json ball.json [player.json] --out reps.json [--config reps.yaml]

Input files are recognised by content, so their order doesn't matter.
"""
import argparse
import json
import sys
from pathlib import Path

from .config import RepsConfig
from .detect import detect_reps


def classify(doc: dict) -> str:
    if "strong_foot" in doc:
        return "player"
    frames = doc.get("frames") or [{}]
    if "keypoints" in frames[0]:
        return "pose"
    if "ball" in frames[0]:
        return "ball"
    raise ValueError("can't tell whether this is pose.json, ball.json or player.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline.reps", description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="pose.json, ball.json and optionally player.json")
    ap.add_argument("--out", required=True, help="reps.json to write")
    ap.add_argument("--config", help="YAML file overriding RepsConfig defaults")
    a = ap.parse_args(argv)

    docs = {}
    for path in a.inputs:
        doc = json.loads(Path(path).read_text())
        docs[classify(doc)] = doc
    missing = {"pose", "ball"} - set(docs)
    if missing:
        ap.error(f"missing input: {', '.join(sorted(missing))}.json")

    cfg = RepsConfig.from_yaml(a.config) if a.config else RepsConfig()
    out = detect_reps(docs["pose"], docs["ball"], docs.get("player"), cfg)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(out['reps'])} reps written to {a.out}")
    for r in out["reps"]:
        print(f"  {r['rep_id']} {r['foot']:<5} {r['frames']} confidence {r['confidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
