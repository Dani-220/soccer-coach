"""Load one rep from pose.json + reps.json and normalize its skeleton.

Normalized space: origin at the hip midpoint of each frame, unit = the rep's median
torso length (shoulder midpoint to hip midpoint), x flipped so the player always moves
in +x, y down (same as image space).
"""
from dataclasses import dataclass, field

import numpy as np

CORE = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
IDX = {name: i for i, name in enumerate(CORE)}
# Index permutation that swaps every left_* joint with its right_* twin.
LR_SWAP = [IDX[n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")] for n in CORE]


@dataclass
class Rep:
    rep_id: str
    move: str
    foot: str
    fps: float
    frames: list            # video frame index per sample, start..end
    contact: int            # sample index of the contact frame
    pts: np.ndarray         # (T, J, 2) image px, gaps filled
    conf: np.ndarray        # (T, J), 0 where a frame had no player
    hip: np.ndarray         # (T, 2) hip midpoint, image px
    torso: float            # median torso length, px
    direction: int          # +1 moves in +x, -1 moves in -x
    norm: np.ndarray = field(default=None)  # (T, J, 2) normalized


def _mid(pts, a, b):
    return (pts[..., IDX[a], :] + pts[..., IDX[b], :]) / 2.0


def load_rep(pose, reps, rep_id):
    """Pull rep `rep_id` out of parsed pose.json / reps.json dicts and normalize it."""
    rep = next((r for r in reps["reps"] if r["rep_id"] == rep_id), None)
    if rep is None:
        raise ValueError(f"rep {rep_id!r} not in reps.json ({[r['rep_id'] for r in reps['reps']]})")
    fr = rep["frames"]
    by_f = {f["f"]: f["keypoints"] for f in pose["frames"]}
    frames = list(range(fr["start"], fr["end"] + 1))
    missing = [f for f in frames if f not in by_f]
    if missing:
        raise ValueError(f"rep {rep_id}: frames {missing[:5]}... not in pose.json")

    T, J = len(frames), len(CORE)
    pts = np.full((T, J, 2), np.nan)
    conf = np.zeros((T, J))
    for t, f in enumerate(frames):
        kp = by_f[f]
        if not kp:
            continue
        for j, name in enumerate(CORE):
            x, y, c = kp[name]
            pts[t, j] = (x, y)
            conf[t, j] = c
    have = ~np.isnan(pts[:, 0, 0])
    if not have.any():
        raise ValueError(f"rep {rep_id}: no player detected in any frame")
    if not have.all():
        # Fill frames with no player by linear interpolation (clamped at the ends).
        # conf stays 0 there, so those samples carry no weight and are not drawn.
        t_have = np.flatnonzero(have)
        for j in range(J):
            for k in range(2):
                pts[:, j, k] = np.interp(np.arange(T), t_have, pts[t_have, j, k])

    hip = _mid(pts, "left_hip", "right_hip")
    sh = _mid(pts, "left_shoulder", "right_shoulder")
    torso = float(np.median(np.linalg.norm(sh[have] - hip[have], axis=1)))
    if torso <= 1e-6:
        raise ValueError(f"rep {rep_id}: torso length is zero")
    t_have = np.flatnonzero(have)
    dx = hip[t_have[-1], 0] - hip[t_have[0], 0]
    direction = 1 if dx >= 0 else -1

    r = Rep(rep_id=rep_id, move=rep["move"], foot=rep["foot"], fps=float(pose["fps"]),
            frames=frames, contact=fr["contact"] - fr["start"], pts=pts, conf=conf,
            hip=hip, torso=torso, direction=direction)
    r.norm = normalize(pts, hip, torso, direction)
    return r


def normalize(pts, hip, torso, direction):
    n = (pts - hip[:, None, :]) / torso
    n[..., 0] *= direction
    return n


def denormalize(norm, hip, torso, direction):
    """Inverse of normalize(). norm is (J, 2) with hip (2,), or (T, J, 2) with hip (T, 2)."""
    p = norm.copy()
    p[..., 0] *= direction
    return p * torso + hip[..., None, :]


def swap_sides(norm, conf):
    """Relabel left<->right, so a left-foot rep can be compared limb-for-limb with a right-foot rep."""
    return norm[:, LR_SWAP], conf[:, LR_SWAP]
