"""Shared per-rep measurement context: keypoint access, body height, direction, camera view, ball position.

Everything here follows CONTRACT.md conventions: pixel coordinates with y growing downward, integer
frame indices, MediaPipe keypoint names, and body height H = nose-to-lower-ankle vertical distance.
"""
from dataclasses import dataclass

import numpy as np

# Nose-to-ankle is ~93% of standing height (CONTRACT.md: H_cm = height_cm * 0.93).
NOSE_TO_ANKLE_OF_STATURE = 0.93
BODY_HEIGHT_FRAMES = 30
# Shoulder-width / torso-length ratio above which the camera is treated as facing the player.
# Side-on this ratio is ~0.05 (both shoulders overlap); facing the camera it is ~0.8.
FRONT_VIEW_WIDTH_RATIO = 0.35


def _parse_frames(pose):
    """pose.json frames -> list indexed by f of {name: np.array([x, y, conf])}, {} when no player."""
    out = [{} for _ in pose["frames"]]
    for fr in pose["frames"]:
        out[fr["f"]] = {k: np.asarray(v, dtype=float) for k, v in fr["keypoints"].items()}
    return out


def _parse_balls(ball, n):
    """ball.json frames -> list indexed by f of np.array([x, y, radius, conf]) or None."""
    out = [None] * n
    for fr in ball["frames"]:
        if fr["f"] < n and fr["ball"] is not None:
            out[fr["f"]] = np.asarray(fr["ball"], dtype=float)
    return out


def body_height_px(frames):
    """Body height H in pixels, as defined in CONTRACT.md.

    H = median over the first 30 frames that contain a player of
        max(left_ankle.y, right_ankle.y) - nose.y
    (vertical distance from the nose to the lower ankle; y grows downward).
    Returns None if no frame contains a player.
    """
    hs = []
    for kp in frames:
        if kp:
            hs.append(max(kp["left_ankle"][1], kp["right_ankle"][1]) - kp["nose"][1])
            if len(hs) == BODY_HEIGHT_FRAMES:
                break
    return float(np.median(hs)) if hs else None


def hip_mid(kp):
    return (kp["left_hip"][:2] + kp["right_hip"][:2]) / 2


def shoulder_mid(kp):
    return (kp["left_shoulder"][:2] + kp["right_shoulder"][:2]) / 2


def travel_direction(frames, start, end):
    """+1 if the player moves toward +x during [start, end], -1 toward -x.

    Sign of the least-squares slope of hip-midpoint x over the frames that contain a player.
    Defaults to +1 when there are fewer than two such frames or the slope is exactly zero.
    """
    ts, xs = [], []
    for f in range(start, end + 1):
        if frames[f]:
            ts.append(f)
            xs.append(hip_mid(frames[f])[0])
    if len(ts) < 2:
        return 1
    slope = np.polyfit(ts, xs, 1)[0]
    return -1 if slope < 0 else 1


def camera_view(frames):
    """'side' or 'front', from how wide the shoulders look relative to the torso.

    ratio = median over player frames of |left_shoulder.x - right_shoulder.x| / |shoulder_mid - hip_mid|
    Side-on, the shoulders overlap (ratio ~0.05); facing the camera they are about as wide as the
    torso is long (ratio ~0.8). 'front' when ratio > FRONT_VIEW_WIDTH_RATIO. Behind-the-player
    footage also counts as 'front' (same geometry, mirrored).
    """
    ratios = []
    for kp in frames:
        if kp:
            torso = np.linalg.norm(shoulder_mid(kp) - hip_mid(kp))
            if torso > 0:
                ratios.append(abs(kp["left_shoulder"][0] - kp["right_shoulder"][0]) / torso)
    if not ratios:
        return "side"
    return "front" if np.median(ratios) > FRONT_VIEW_WIDTH_RATIO else "side"


@dataclass
class RepContext:
    """Everything a metric function needs for one rep."""
    frames: list          # per-frame {keypoint: [x, y, conf]} ({} = no player)
    balls: list           # per-frame [x, y, r, conf] or None
    fps: float
    H: float              # body height, px
    px_per_m: float       # None when the player's height is unknown
    direction: int        # +1 = moving toward +x, -1 = toward -x
    view: str             # 'side' or 'front'
    foot: str             # kicking foot, 'left' | 'right'
    start: int
    plant: int
    contact: int
    end: int

    @property
    def support(self):
        return "left" if self.foot == "right" else "right"

    @property
    def n_frames(self):
        return len(self.frames)

    def point(self, f, name):
        """(x, y) of keypoint `name` at frame f, or None if no player was detected there."""
        if not 0 <= f < len(self.frames) or not self.frames[f]:
            return None
        return self.frames[f][name][:2]

    def ball_at_rest(self, f, window_s=0.05):
        """Ball (x, y) at frame f (at or before contact), robust to dropouts and occlusion.

        Median of the ball detections in [f - w, min(f + w, contact)], with w = 0.05 s and the
        window clipped to the rep. Before contact the ball is still (dead-ball shot), so nearby
        detections stand in for a frame where the ball is hidden, e.g. behind the foot at contact.
        Returns None if no detection falls in the window.
        """
        w = max(2, round(window_s * self.fps))
        lo, hi = max(self.start, f - w), min(f + w, self.contact)
        pts = [self.balls[g][:2] for g in range(lo, hi + 1) if self.balls[g] is not None]
        return np.median(pts, axis=0) if pts else None

    @classmethod
    def build(cls, video, rep, height_cm=None):
        """Context for one reps.json entry. A missing `plant` frame falls back to contact."""
        fr = rep["frames"]
        px_per_m = video.H / (height_cm / 100 * NOSE_TO_ANKLE_OF_STATURE) if height_cm else None
        return cls(
            frames=video.frames, balls=video.balls, fps=video.fps, H=video.H, px_per_m=px_per_m,
            direction=travel_direction(video.frames, fr["start"], fr["contact"]),
            view=video.view, foot=rep["foot"],
            start=fr["start"], plant=fr.get("plant", fr["contact"]), contact=fr["contact"], end=fr["end"],
        )


@dataclass
class Video:
    """Parsed pose + ball data and the per-video quantities (H, camera view)."""
    frames: list
    balls: list
    fps: float
    H: float
    view: str

    @classmethod
    def load(cls, pose, ball, view=None):
        """`view` ('side' | 'front') overrides auto-detection."""
        frames = _parse_frames(pose)
        H = body_height_px(frames)
        if H is None or H <= 0:
            raise ValueError("pose.json has no usable frame with a player; cannot compute body height")
        return cls(frames=frames, balls=_parse_balls(ball, len(frames)), fps=float(pose["fps"]),
                   H=H, view=view or camera_view(frames))
