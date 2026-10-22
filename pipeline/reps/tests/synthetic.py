"""A small, controllable synthetic shot sequence in contract format, with known answers.

Simpler than contract/tools/generate_synthetic.py: a rigid stick figure whose support ankle steps
in and stops at `plant`, whose kicking ankle accelerates into the ball at `contact` and decelerates
to a stop at `end`. Lets tests vary fps, direction, foot, occlusion and noise independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

W, H_IMG = 1920, 1080
GROUND = 940.0
BODY_H = 580.0           # nose-to-ankle px
BALL_R = 39.0
BALL_Y = GROUND - 24.0

CORE = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


@dataclass
class RepSpec:
    foot: str = "right"
    approach_s: float = 1.0      # time from rep start to contact
    plant_to_contact_s: float = 0.06
    follow_s: float = 0.33       # contact to end of follow-through
    tail_s: float = 0.2          # standing still after end
    ball_speed_hs: float = 10.0  # H/s after contact
    plant_gap: float = 0.13      # H, support ankle behind the ball
    hide_ball_at_contact: int = 0  # frames hidden on each side of contact
    foot_reaches_ball: bool = True


@dataclass
class Scenario:
    reps: list[RepSpec] = field(default_factory=lambda: [RepSpec()])
    fps: float = 120.0
    direction: int = 1           # +1 moves left->right, -1 right->left
    gap_s: float = 0.5           # no-player frames between reps
    noise_px: float = 0.0
    seed: int = 0


def _ease_in(s):   # fastest at s=1 (strike)
    return np.clip(s, 0, 1) ** 2


def _ease_out(s):  # zero speed at s=1 (follow-through stops)
    s = np.clip(s, 0, 1)
    return 1 - (1 - s) ** 2


def _rep(spec: RepSpec, fps: float):
    """Rep-local frames in a +x world, ball at x=0. Returns (keypoint dicts, ball positions, truth)."""
    f = lambda s: int(round(s * fps))
    C = f(spec.approach_s)
    P = C - f(spec.plant_to_contact_s)
    E = C + f(spec.follow_s)
    N = E + f(spec.tail_s) + 1
    step = f(0.25)
    sup_land_x = -spec.plant_gap * BODY_H
    sup_from_x = sup_land_x - 0.35 * BODY_H
    contact_pt = np.array([-0.05 * BODY_H, BALL_Y + 6])
    back_pt = np.array([sup_land_x - 0.3 * BODY_H, GROUND - 0.3 * BODY_H])
    follow_pt = np.array([0.45 * BODY_H, GROUND - 0.5 * BODY_H])
    kick_rest = np.array([sup_land_x - 0.2 * BODY_H, GROUND])
    swing_start = P - f(0.12)
    if not spec.foot_reaches_ball:
        contact_pt = contact_pt + [-0.25 * BODY_H, -0.2 * BODY_H]

    frames, balls = [], []
    for t in range(N):
        # support ankle: walks in at constant speed, lands at P with a small lift arc
        s = 1 - (P - t) / step
        if s >= 1:
            sup = np.array([sup_land_x, GROUND])
        elif s < 0:
            sup = np.array([sup_from_x, GROUND])
        else:
            x = sup_from_x + (sup_land_x - sup_from_x) * s
            sup = np.array([x, GROUND - 0.06 * BODY_H * np.sin(np.pi * s)])
        # kicking ankle
        if t <= swing_start:
            kick = kick_rest
        elif t <= P:
            kick = kick_rest + (back_pt - kick_rest) * _ease_out((t - swing_start) / (P - swing_start))
        elif t <= C:
            kick = back_pt + (contact_pt - back_pt) * _ease_in((t - P) / (C - P))
        else:
            kick = contact_pt + (follow_pt - contact_pt) * _ease_out((t - C) / (E - C))
        hip = np.array([sup_land_x - 0.1 * BODY_H + 0.05 * BODY_H * _ease_out((t - P) / (E - P)), GROUND - 0.5 * BODY_H])
        frames.append(_skeleton(hip, spec.foot, kick, sup))

        if t <= C:
            b = np.array([0.0, BALL_Y])
        else:
            v = spec.ball_speed_hs * BODY_H / fps
            b = np.array([v * 0.97 * (t - C), BALL_Y - v * 0.25 * (t - C)])
        if abs(t - C) <= spec.hide_ball_at_contact and spec.hide_ball_at_contact > 0:
            b = None
        balls.append(b)
    truth = {"foot": spec.foot, "start": 0, "plant": P, "contact": C, "end": E}
    return frames, balls, truth, N


def _skeleton(hip, kick_foot, kick_ank, sup_ank):
    sup_foot = "left" if kick_foot == "right" else "right"
    sh = hip + [0, -0.33 * BODY_H]
    nose = np.array([hip[0], GROUND - BODY_H])
    kp = {"nose": nose}
    for side, ank in ((kick_foot, kick_ank), (sup_foot, sup_ank)):
        kp[f"{side}_shoulder"] = sh
        kp[f"{side}_elbow"] = sh + [0, 0.18 * BODY_H]
        kp[f"{side}_wrist"] = sh + [0, 0.33 * BODY_H]
        kp[f"{side}_hip"] = hip
        kp[f"{side}_knee"] = (hip + ank) / 2 + [0.03 * BODY_H, 0]
        kp[f"{side}_ankle"] = ank
        kp[f"{side}_heel"] = ank + [-0.03 * BODY_H, 0.02 * BODY_H]
        kp[f"{side}_foot_index"] = ank + [0.07 * BODY_H, 0.02 * BODY_H]
    return kp


def generate(sc: Scenario):
    """Return (pose dict, ball dict, list of truth dicts in global frame indices)."""
    rng = np.random.default_rng(sc.seed)
    ball_x0 = W * 0.6 if sc.direction > 0 else W * 0.4
    pose_frames, ball_frames, truths = [], [], []
    mirror = lambda x: ball_x0 + sc.direction * x

    for i, spec in enumerate(sc.reps):
        frames, balls, tr, N = _rep(spec, sc.fps)
        off = len(pose_frames)
        for t in range(N):
            kp = {}
            for name in CORE:
                x, y = frames[t][name] + rng.normal(0, sc.noise_px, 2) if sc.noise_px else frames[t][name]
                kp[name] = [round(float(mirror(x)), 2), round(float(y), 2), 0.9]
            pose_frames.append({"f": off + t, "keypoints": kp})
            b = balls[t]
            if b is not None:
                bx, by = mirror(b[0]), b[1]
                if sc.noise_px:
                    bx, by = np.array([bx, by]) + rng.normal(0, sc.noise_px * 0.6, 2)
                b = [round(float(bx), 2), round(float(by), 2), BALL_R, 0.9] if 0 <= bx < W and 0 <= by < H_IMG else None
            ball_frames.append({"f": off + t, "ball": b})
        truths.append({k: (v + off if k != "foot" else v) for k, v in tr.items()})
        if i < len(sc.reps) - 1:
            for _ in range(int(round(sc.gap_s * sc.fps))):
                f = len(pose_frames)
                pose_frames.append({"f": f, "keypoints": {}})
                ball_frames.append({"f": f, "ball": None})

    meta = {"schema_version": "1.0", "video": "synthetic_test.mp4", "fps": sc.fps, "width": W, "height": H_IMG}
    return {**meta, "frames": pose_frames}, {**meta, "frames": ball_frames}, truths
