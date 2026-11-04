"""Overlay two reps (Workstream E).

  # both reps from one video: new rep drawn solid, old rep as a ghost
  python -m pipeline.overlay --in pose.json reps.json --new r002 --old r001 --out out/overlay.mp4

  # reps from two videos: first pair holds the NEW rep, second pair the OLD rep
  python -m pipeline.overlay --in new_pose.json new_reps.json old_pose.json old_reps.json --out out/overlay.mp4

  # draw on the new rep's original video instead of a plain background
  python -m pipeline.overlay --in pose.json reps.json --background video --video clip.mp4 --out out/overlay.mp4

Writes the video to --out and a PNG of the contact frame next to it (same name, .png),
or only the PNG if --out ends in .png.
"""
import argparse
import json
import sys
from pathlib import Path

from .overlay import Canvas, compare, contact_summary, write_png, write_video
from .skeleton import load_rep


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m pipeline.overlay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inputs", nargs="+", required=True, metavar="FILE",
                   help="POSE REPS [OLD_POSE OLD_REPS]; the first pair holds the new rep")
    p.add_argument("--out", required=True, help="overlay .mp4 (PNG written alongside), or a .png")
    p.add_argument("--new", help="new rep_id (default: last rep in the first reps.json)")
    p.add_argument("--old", help="old/ghost rep_id (default: first rep in the old reps.json)")
    p.add_argument("--png", help="contact-frame PNG path (default: --out with .png)")
    p.add_argument("--background", choices=["plain", "video"], default="plain")
    p.add_argument("--video", help="new rep's video file (default: pose.json 'video', relative to pose.json)")
    p.add_argument("--size", type=_size, default=(960, 960), help="plain background WxH (default 960x960)")
    p.add_argument("--fps", type=float, help="output fps (default: min(source fps, 30), i.e. slow motion)")
    p.add_argument("--hold-contact", type=float, default=0.5, help="seconds to freeze on contact (default 0.5)")
    p.add_argument("--max-dev", type=float, default=0.5,
                   help="deviation (torso lengths) shown fully red (default 0.5)")
    p.add_argument("--min-conf", type=float, default=0.3, help="hide joints below this confidence")
    p.add_argument("--no-swap", action="store_true",
                   help="do not relabel the ghost left<->right when the two reps use different feet")
    a = p.parse_args(argv)
    if len(a.inputs) not in (2, 4):
        p.error("--in takes POSE REPS or POSE REPS OLD_POSE OLD_REPS")
    return a


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    paths = [Path(x) for x in a.inputs]
    docs = [json.loads(x.read_text()) for x in paths]
    pose, reps = docs[0], docs[1]
    old_pose, old_reps = (docs[2], docs[3]) if len(docs) == 4 else (pose, reps)

    new_id = a.new or reps["reps"][-1]["rep_id"]
    old_id = a.old or next((r["rep_id"] for r in old_reps["reps"]
                            if len(docs) == 4 or r["rep_id"] != new_id), new_id)
    new, old = load_rep(pose, reps, new_id), load_rep(old_pose, old_reps, old_id)
    if new.move != old.move:
        print(f"warning: comparing different moves ({new.move} vs {old.move})", file=sys.stderr)
    cmp = compare(new, old, swap=not a.no_swap, min_conf=a.min_conf)

    video = None
    if a.background == "video":
        video = Path(a.video) if a.video else paths[0].parent / pose["video"]
        if not video.exists():
            sys.exit(f"error: video not found: {video} (pass --video or use --background plain)")
    canvas = Canvas(cmp, a.background, video, a.size, (pose["width"], pose["height"]))
    out = Path(a.out)
    try:
        if out.suffix.lower() == ".png":
            write_png(canvas, out, a.max_dev)
            written = [out]
        else:
            png = Path(a.png) if a.png else out.with_suffix(".png")
            fps = a.fps or min(new.fps, 30.0)
            write_video(canvas, out, a.max_dev, fps, a.hold_contact)
            write_png(canvas, png, a.max_dev)
            written = [out, png]
    finally:
        canvas.close()

    for w in written:
        print("wrote", w)
    top = ", ".join(f"{n} {d:.2f}" for n, d in contact_summary(cmp))
    print(f"{new.rep_id} vs {old.rep_id} at contact, largest deviations (torso lengths): {top}")


if __name__ == "__main__":
    main()
