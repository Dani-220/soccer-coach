"""Run the whole pipeline on one video: B -> C -> D -> E -> F, validating every output.

  python run.py video.mp4 --player player.json --out out/<name>/
  python run.py --reuse-pose --player player.json --out out/<name>/   # skip B, use the pose/ball.json already there

Each stage is called through its own CLI (python -m pipeline.<stage>). After each stage, every file written
so far is checked with contract/tools/validate.py, so cross-file rules are re-checked as files are added.
The run stops at the first failing stage or validation, with a message naming the stage.

Writes into --out: player.json (copy), pose.json, ball.json, reps.json, metrics.json,
overlays/<new>_vs_<old>.mp4 + .png for each rep vs. the one before it, and cards.json (+ plan.json) once
Workstream F exists. Until then F is skipped and the app shows contract/examples/cards.example.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATE = ROOT / "contract" / "tools" / "validate.py"
STAGE_NAMES = {"B": "pose", "C": "reps", "D": "metrics", "E": "overlay", "F": "coach"}


class StageFailed(Exception):
    pass


def coach_available():
    return (ROOT / "pipeline" / "coach" / "__main__.py").exists()


def overlay_path(out_dir, new_id, old_id):
    return Path(out_dir) / "overlays" / f"{new_id}_vs_{old_id}.mp4"


def _run(stage, cmd):
    print(f"\n== {stage} ({STAGE_NAMES[stage]}): {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode
    if rc != 0:
        raise StageFailed(f"stage {stage} ({STAGE_NAMES[stage]}) exited with code {rc}; its output is above")


def _validate(stage, files):
    files = [f for f in files if Path(f).exists()]
    print(f"-- validating {', '.join(Path(f).name for f in files)}", flush=True)
    rc = subprocess.run([sys.executable, VALIDATE, *files], cwd=ROOT).returncode
    if rc != 0:
        raise StageFailed(f"stage {stage} ({STAGE_NAMES[stage]}) wrote files that break the contract; "
                          "see the FAIL lines above")


def _expect(stage, *paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise StageFailed(f"stage {stage} ({STAGE_NAMES[stage]}) finished but did not write {', '.join(missing)}")


def run(video, player, out_dir, reuse_pose=False, pose_args=()):
    py = sys.executable
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = {k: out / f"{k}.json" for k in ("player", "pose", "ball", "reps", "metrics", "cards", "plan")}

    print(f"-- validating {Path(player).name}", flush=True)
    if subprocess.run([py, VALIDATE, "--type", "player", player], cwd=ROOT).returncode != 0:
        raise StageFailed(f"{player} is not a valid player.json; see the FAIL lines above")
    if Path(player).resolve() != f["player"].resolve():
        shutil.copyfile(player, f["player"])
    checked = [f["player"]]

    if reuse_pose:
        _expect("B", f["pose"], f["ball"])
        print(f"\n== B (pose): reusing {f['pose']} and {f['ball']}")
    else:
        _run("B", [py, "-m", "pipeline.pose", "--in", video, "--out", f["pose"], "--ball-out", f["ball"], *pose_args])
        _expect("B", f["pose"], f["ball"])
    checked += [f["pose"], f["ball"]]
    _validate("B", checked)

    _run("C", [py, "-m", "pipeline.reps", "--in", f["pose"], f["ball"], f["player"], "--out", f["reps"]])
    _expect("C", f["reps"])
    checked.append(f["reps"])
    _validate("C", checked)

    _run("D", [py, "-m", "pipeline.metrics", "--in", f["pose"], f["ball"], f["reps"], f["player"], "--out", f["metrics"]])
    _expect("D", f["metrics"])
    checked.append(f["metrics"])
    _validate("D", checked)

    rep_ids = [r["rep_id"] for r in json.loads(f["reps"].read_text())["reps"]]
    if len(rep_ids) < 2:
        print(f"\n== E (overlay): skipped, {len(rep_ids)} rep(s) found and an overlay needs two")
    for old_id, new_id in zip(rep_ids, rep_ids[1:]):
        dst = overlay_path(out, new_id, old_id)
        dst.parent.mkdir(exist_ok=True)
        _run("E", [py, "-m", "pipeline.overlay", "--in", f["pose"], f["reps"],
                   "--new", new_id, "--old", old_id, "--out", dst])
        _expect("E", dst, dst.with_suffix(".png"))

    if coach_available():
        _run("F", [py, "-m", "pipeline.coach", "--in", f["metrics"], f["player"], "--out", f["cards"]])
        _expect("F", f["cards"])
        checked += [f["cards"], f["plan"]]
        _validate("F", checked)
    else:
        print("\n== F (coach): not built yet, skipped; the app shows contract/examples/cards.example.json")

    print(f"\nrun.py: all stages passed; outputs in {out}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="training video (not needed with --reuse-pose)")
    ap.add_argument("--player", required=True, help="player.json")
    ap.add_argument("--out", required=True, help="output folder, e.g. out/<name>/")
    ap.add_argument("--reuse-pose", action="store_true",
                    help="skip B and use the pose.json and ball.json already in --out")
    ap.add_argument("--pose-arg", action="append", default=[], metavar="ARG",
                    help="extra argument passed to pipeline.pose, e.g. --pose-arg=--pose-model --pose-arg=lite")
    a = ap.parse_args(argv)
    if not a.reuse_pose and not a.video:
        ap.error("a video is required unless --reuse-pose is given")
    if a.video and not a.reuse_pose and not Path(a.video).exists():
        ap.error(f"video not found: {a.video}")
    if not Path(a.player).exists():
        ap.error(f"player file not found: {a.player}")
    try:
        run(a.video, a.player, a.out, a.reuse_pose, a.pose_arg)
    except StageFailed as e:
        print(f"\nrun.py: FAILED: {e}. Outputs written so far are in {a.out}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
