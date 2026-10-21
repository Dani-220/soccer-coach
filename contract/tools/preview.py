"""Draw skeleton + ball at a rep's key frames, so you can eyeball any pose.json.

  python tools/preview.py examples/pose.example.json examples/ball.example.json examples/reps.example.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"),
         ("knee", "ankle"), ("ankle", "heel"), ("ankle", "foot_index"), ("heel", "foot_index")]


def main(pose_p, ball_p, reps_p, out_p):
    pose, ball, reps = (json.load(open(p)) for p in (pose_p, ball_p, reps_p))
    fr = {f["f"]: f["keypoints"] for f in pose["frames"]}
    bl = {f["f"]: f["ball"] for f in ball["frames"]}
    phases = ["start", "plant", "contact", "end"]
    fig, axes = plt.subplots(len(reps["reps"]), 4, figsize=(16, 4.2 * len(reps["reps"])), squeeze=False)
    for row, rep in zip(axes, reps["reps"]):
        for ax, ph in zip(row, phases):
            f = rep["frames"][ph]
            kp = fr[f]
            for side, color in (("left", "tab:blue"), ("right", "tab:red")):
                for a, b in BONES:
                    pa, pb = kp.get(f"{side}_{a}"), kp.get(f"{side}_{b}")
                    if pa and pb:
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=2.5)
            if "nose" in kp:
                ax.plot(kp["nose"][0], kp["nose"][1], "ko", ms=6)
            if bl.get(f):
                ax.add_patch(plt.Circle(bl[f][:2], bl[f][2], fill=False, color="black", lw=1.5))
            ax.axhline(955, color="green", lw=1)
            ax.set_xlim(500, 1500); ax.set_ylim(1000, 250); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{rep['rep_id']} ({rep['foot']} foot) {ph}  f={f}")
    fig.suptitle("blue = left side, red = right side, green = ground", y=0.995)
    fig.tight_layout(); fig.savefig(out_p, dpi=70)
    print("wrote", out_p)


if __name__ == "__main__":
    main(*sys.argv[1:5])
