"""Tiny synthetic skeletons with exactly known geometry, for unit tests.

The default skeleton stands upright with the nose at y=300 and both ankles at y=900, so body
height H = 600 px exactly. Tests move individual keypoints to set up a known answer.
"""
import copy

NOSE_Y, HIP_Y, SHOULDER_Y, KNEE_Y, ANKLE_Y = 300.0, 560.0, 360.0, 730.0, 900.0
H = ANKLE_Y - NOSE_Y      # 600 px
FPS = 120
# Rep phases used by most tests.
START, PLANT, CONTACT, END, N = 0, 20, 25, 40, 50


def skeleton(x, view="side", conf=0.9):
    """All 17 core keypoints for a player whose hip midpoint is at x. Side view: left/right
    almost overlap. Front view: shoulders 160 px wide, hips 100 px wide (left on the image right)."""
    sh_half, hip_half, leg_half = (4, 3, 3) if view == "side" else (80, 50, 40)
    kp = {"nose": [x, NOSE_Y]}
    for side, s in (("left", 1), ("right", -1)):
        kp[f"{side}_shoulder"] = [x + s * sh_half, SHOULDER_Y]
        kp[f"{side}_elbow"] = [x + s * sh_half, 460.0]
        kp[f"{side}_wrist"] = [x + s * sh_half, 550.0]
        kp[f"{side}_hip"] = [x + s * hip_half, HIP_Y]
        kp[f"{side}_knee"] = [x + s * leg_half, KNEE_Y]
        kp[f"{side}_ankle"] = [x + s * leg_half, ANKLE_Y]
        kp[f"{side}_heel"] = [x + s * leg_half - 15, ANKLE_Y + 10]
        kp[f"{side}_foot_index"] = [x + s * leg_half + 40, ANKLE_Y + 15]
    return {k: [float(v[0]), float(v[1]), conf] for k, v in kp.items()}


class Clip:
    """A mutable synthetic video: per-frame keypoints and ball, convertible to pose/ball/reps JSON."""

    def __init__(self, n=N, x0=600.0, speed=4.0, direction=1, view="side", width=1920, height=1080):
        self.width, self.height, self.view = width, height, view
        self.kps = [skeleton(x0 + direction * speed * f, view) for f in range(n)]
        self.balls = [None] * n

    def set(self, f, name, x=None, y=None):
        """Move keypoint `name` at frame f (None keeps that coordinate)."""
        p = self.kps[f][name]
        self.kps[f][name] = [p[0] if x is None else float(x), p[1] if y is None else float(y), p[2]]

    def get(self, f, name):
        return self.kps[f][name][:2]

    def shift(self, f, names, dx=0.0, dy=0.0):
        for n in names:
            x, y = self.get(f, n)
            self.set(f, n, x + dx, y + dy)

    def hide_player(self, f):
        self.kps[f] = {}

    def ball(self, f, x, y, r=20.0):
        self.balls[f] = [float(x), float(y), r, 0.9]

    def mirrored(self):
        """Left-right flipped copy (x -> width - x), e.g. a player running toward -x."""
        m = copy.deepcopy(self)
        for kp in m.kps:
            for k, v in kp.items():
                kp[k] = [self.width - v[0], v[1], v[2]]
        m.balls = [None if b is None else [self.width - b[0], *b[1:]] for b in m.balls]
        return m

    def pose(self):
        return {"schema_version": "1.0", "video": "synth.mp4", "fps": FPS, "width": self.width,
                "height": self.height, "frames": [{"f": f, "keypoints": kp} for f, kp in enumerate(self.kps)]}

    def ball_doc(self):
        return {"schema_version": "1.0", "video": "synth.mp4", "fps": FPS,
                "frames": [{"f": f, "ball": b} for f, b in enumerate(self.balls)]}


def rep(foot="right", start=START, plant=PLANT, contact=CONTACT, end=END, rep_id="r001"):
    return {"rep_id": rep_id, "move": "shot_instep", "foot": foot,
            "frames": {"start": start, "plant": plant, "contact": contact, "end": end}, "confidence": 1.0}


def reps_doc(*reps):
    return {"schema_version": "1.0", "video": "synth.mp4", "player_id": "p1", "reps": list(reps)}
