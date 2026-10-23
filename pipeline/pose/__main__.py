"""Workstream B: training video -> pose.json + ball.json (contract v1.0).

  python -m pipeline.pose --in clip.mp4 --out out/pose.json [--debug out/debug.mp4]
"""
import argparse
import json
import sys
from pathlib import Path

from .extract import extract
from .one_euro import KeypointSmoother
from .tracking import PlayerTracker


def default_ball_path(pose_path):
    """out/pose.json -> out/ball.json, out/clip_pose.json -> out/clip_ball.json."""
    p = Path(pose_path)
    name = p.name
    if name.startswith("pose."):
        name = "ball." + name[len("pose."):]
    elif "_pose." in name:
        name = name.replace("_pose.", "_ball.", 1)
    else:
        name = p.stem + "_ball" + p.suffix
    return p.with_name(name)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.pose", description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="video", required=True, help="input training video")
    ap.add_argument("--out", required=True, help="pose.json path")
    ap.add_argument("--ball-out", help="ball.json path (default: next to --out, e.g. pose.json -> ball.json)")
    ap.add_argument("--debug", nargs="?", const="", metavar="MP4",
                    help="also write a video with skeleton and ball drawn (default: <out dir>/<video>_debug.mp4)")
    ap.add_argument("--fps", type=float, help="override the fps the container reports")

    g = ap.add_argument_group("pose (MediaPipe)")
    g.add_argument("--pose-model", default="heavy", help="lite | full | heavy, or a .task path (default: heavy)")
    g.add_argument("--max-poses", type=int, default=2, help="people MediaPipe may detect; one is tracked (default: 2)")
    g.add_argument("--pose-conf", type=float, default=0.5, help="MediaPipe detection/presence/tracking threshold")
    g.add_argument("--max-jump", type=float, default=0.75,
                   help="max hip movement per frame, in body heights, before a pose counts as a different person")

    g = ap.add_argument_group("smoothing (One Euro)")
    g.add_argument("--min-cutoff", type=float, default=1.5, help="Hz, cutoff when still (lower = smoother)")
    g.add_argument("--beta", type=float, default=0.01, help="cutoff increase per px/s (higher = less lag)")
    g.add_argument("--d-cutoff", type=float, default=1.0, help="Hz, cutoff for the speed estimate")
    g.add_argument("--no-smooth", action="store_true", help="write raw MediaPipe keypoints")

    g = ap.add_argument_group("ball (Ultralytics YOLO)")
    g.add_argument("--ball-model", default="yolo11m.pt", help="YOLO weights with a 'sports ball' class (default: yolo11m.pt)")
    g.add_argument("--ball-conf", type=float, default=0.1,
                   help="minimum ball confidence (default: 0.1; the tracker's gate rejects stray detections)")
    g.add_argument("--ball-imgsz", type=int, default=1280, help="YOLO inference size; larger finds smaller balls")
    g.add_argument("--ball-max-missing", type=float, default=0.5, help="seconds without a hit before the ball track is dropped")
    g.add_argument("--ball-kick-speed", type=float, default=120.0,
                   help="fastest launch the gate allows for, in ball radii per second (120 r/s is about 13 m/s)")
    g.add_argument("--no-ball-redetect", action="store_true",
                   help="don't re-run YOLO on an upscaled crop around the predicted ball when it is missed")
    g.add_argument("--device", help="torch device for YOLO, e.g. cpu, mps, cuda:0 (default: auto)")
    return ap.parse_args(argv)


class _Passthrough:
    def reset(self):
        pass

    def __call__(self, keypoints, t):
        return keypoints


def _progress(f, total):
    if f % 50 == 0 or f == total:
        print(f"\r  frame {f}/{total or '?'}", end="", file=sys.stderr, flush=True)


def main(argv=None):
    a = parse_args(argv)
    from .detectors import MediaPipePoseDetector, YoloBallDetector

    pose_out = Path(a.out)
    ball_out = Path(a.ball_out) if a.ball_out else default_ball_path(pose_out)
    debug_out = None
    if a.debug is not None:
        debug_out = Path(a.debug) if a.debug else pose_out.with_name(Path(a.video).stem + "_debug.mp4")
    for p in (pose_out, ball_out, debug_out):
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)

    smoother = _Passthrough() if a.no_smooth else KeypointSmoother(a.min_cutoff, a.beta, a.d_cutoff)
    pose_det = MediaPipePoseDetector(a.pose_model, a.max_poses, a.pose_conf, a.pose_conf, a.pose_conf)
    ball_det = YoloBallDetector(a.ball_model, a.ball_conf, a.ball_imgsz, a.device)
    try:
        pose_doc, ball_doc = extract(a.video, pose_det, ball_det, smoother=smoother,
                                     tracker=PlayerTracker(max_jump=a.max_jump),
                                     ball_opts=dict(max_missing_s=a.ball_max_missing, kick_speed=a.ball_kick_speed),
                                     redetect=not a.no_ball_redetect,
                                     debug_path=debug_out, fps_override=a.fps, progress=_progress)
    finally:
        pose_det.close()
        ball_det.close()
    print(file=sys.stderr)

    for path, doc in ((pose_out, pose_doc), (ball_out, ball_doc)):
        path.write_text(json.dumps(doc, separators=(",", ":")))
    n = len(pose_doc["frames"])
    n_player = sum(1 for fr in pose_doc["frames"] if fr["keypoints"])
    n_ball = sum(1 for fr in ball_doc["frames"] if fr["ball"] is not None)
    print(f"{n} frames @ {pose_doc['fps']} fps, {pose_doc['width']}x{pose_doc['height']}: "
          f"player in {n_player}, ball in {n_ball}")
    print(f"wrote {pose_out}\nwrote {ball_out}" + (f"\nwrote {debug_out}" if debug_out else ""))


if __name__ == "__main__":
    main()
