"""Validate pipeline files against the contract.

  python tools/validate.py examples/                 # every *.json in a folder (type from file name)
  python tools/validate.py out/reps.json             # one file
  python tools/validate.py --type reps my_output.json

Checks (1) the JSON Schema in schemas/ and (2) cross-file rules a schema can't express:
frame ordering, frame indices that exist in pose.json, rep_ids that match across files,
metric statuses that agree with their targets, and cards that copy values from metrics.json.
Exit code is 0 only when everything passes, so CI (Workstream H) can run it on every push.
"""
import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["pose", "ball", "reps", "metrics", "cards", "player", "plan"]


def file_type(path):
    stem = Path(path).name.split(".")[0]
    for t in TYPES:
        if stem == t or stem.startswith(t + "_") or stem.endswith("_" + t):
            return t
    return None


def schema_errors(kind, data):
    schema = json.loads((ROOT / "schemas" / f"{kind}.schema.json").read_text())
    v = Draft202012Validator(schema, format_checker=FormatChecker())
    out = []
    for e in sorted(v.iter_errors(data), key=lambda e: list(e.absolute_path)):
        loc = "/".join(str(p) for p in e.absolute_path) or "(root)"
        out.append(f"{loc}: {e.message[:200]}")
    return out


def cross_checks(docs):
    errs = []
    pose, ball, reps, metrics, cards = (docs.get(k) for k in ("pose", "ball", "reps", "metrics", "cards"))

    for k in ("pose", "ball"):
        d = docs.get(k)
        if d:
            fs = [fr["f"] for fr in d["frames"]]
            if fs != list(range(len(fs))):
                errs.append(f"{k}: frame indices must be 0..N-1 with no gaps or repeats")
    if pose and ball:
        if len(pose["frames"]) != len(ball["frames"]):
            errs.append("pose/ball: different frame counts")
        if pose["fps"] != ball["fps"] or pose["video"] != ball["video"]:
            errs.append("pose/ball: video or fps differ")

    n_frames = len(pose["frames"]) if pose else None
    rep_ids = set()
    if reps:
        for r in reps["reps"]:
            rep_ids.add(r["rep_id"])
            fr = r["frames"]
            order = [fr[k] for k in ("start", "plant", "contact", "end") if k in fr]
            if order != sorted(order):
                errs.append(f"reps/{r['rep_id']}: frames must satisfy start <= plant <= contact <= end, got {fr}")
            if n_frames is not None and fr["end"] >= n_frames:
                errs.append(f"reps/{r['rep_id']}: end frame {fr['end']} is past the last pose frame {n_frames - 1}")
        if len(rep_ids) != len(reps["reps"]):
            errs.append("reps: duplicate rep_id")

    metric_lookup = {}
    if metrics:
        for r in metrics["reps"]:
            if reps and r["rep_id"] not in rep_ids:
                errs.append(f"metrics: rep_id {r['rep_id']} is not in reps.json")
            for m in r["metrics"]:
                metric_lookup[(r["rep_id"], m["name"])] = m
                lo, hi = m["target"]
                if lo > hi:
                    errs.append(f"metrics/{r['rep_id']}/{m['name']}: target low > high")
                if m["value"] is not None:
                    inside = lo <= m["value"] <= hi
                    if inside != (m["status"] == "in_range"):
                        errs.append(f"metrics/{r['rep_id']}/{m['name']}: status '{m['status']}' disagrees with value {m['value']} vs target {m['target']}")
                if n_frames is not None and m["frame"] >= n_frames:
                    errs.append(f"metrics/{r['rep_id']}/{m['name']}: frame {m['frame']} out of range")

    if cards:
        for r in cards["reps"]:
            for c in r["cards"]:
                m = metric_lookup.get((r["rep_id"], c["metric"]))
                if metrics and m is None:
                    errs.append(f"cards/{r['rep_id']}: card for {c['metric']} has no matching metric")
                elif m is not None:
                    if m["value"] != c["measured_value"]:
                        errs.append(f"cards/{r['rep_id']}/{c['metric']}: measured_value {c['measured_value']} != metrics.json {m['value']} (values must be copied, not generated)")
                    if m["target"] != c["target"]:
                        errs.append(f"cards/{r['rep_id']}/{c['metric']}: target differs from metrics.json")
                    if m["status"] != "out_of_range":
                        errs.append(f"cards/{r['rep_id']}/{c['metric']}: card written for a metric that is {m['status']}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--type", choices=TYPES, help="force the file type (single file only)")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]

    docs, failed = {}, False
    for f in files:
        kind = a.type or file_type(f)
        if kind is None:
            print(f"SKIP  {f} (can't tell the type from its name; use --type)")
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"FAIL  {f}: not valid JSON ({e})"); failed = True; continue
        errs = schema_errors(kind, data)
        docs[kind] = data
        print(f"{'FAIL' if errs else 'OK  '}  {f.name:<28} [{kind}]")
        for e in errs[:15]:
            print(f"        - {e}")
        failed |= bool(errs)

    xerrs = cross_checks(docs)
    if len(docs) > 1:
        print(f"{'FAIL' if xerrs else 'OK  '}  cross-file checks ({', '.join(sorted(docs))})")
        for e in xerrs:
            print(f"        - {e}")
    failed |= bool(xerrs)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
