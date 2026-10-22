"""Compare two reps: align them, measure per-joint deviation, render video + contact PNG."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import render
from .align import ghost_sequence
from .skeleton import CORE, Rep, denormalize, swap_sides


@dataclass
class Comparison:
    new: Rep
    old: Rep
    swapped: bool           # ghost relabeled left<->right because the kicking feet differ
    ghost: np.ndarray       # (T, J, 2) normalized ghost pose per new frame
    ghost_conf: np.ndarray  # (T, J)
    old_idx: list           # matched old sample index per new frame
    dev: np.ndarray         # (T, J) joint distance new vs ghost, torso lengths
    min_conf: float

    def top_joints(self, i, k=3):
        ok = (self.new.conf[i] >= self.min_conf) & (self.ghost_conf[i] >= self.min_conf)
        order = [j for j in np.argsort(-self.dev[i]) if ok[j]]
        return order[:k]


def compare(new, old, swap=True, min_conf=0.3):
    old_norm, old_conf = old.norm, old.conf
    swapped = swap and new.foot != old.foot
    if swapped:
        old_norm, old_conf = swap_sides(old_norm, old_conf)
    ghost, gconf, old_idx = ghost_sequence(new, old_norm, old_conf, old.contact)
    dev = np.linalg.norm(new.norm - ghost, axis=-1)
    return Comparison(new, old, swapped, ghost, gconf, old_idx, dev, min_conf)


class Canvas:
    """Where to draw: the new rep's own video frames, or a plain background."""

    def __init__(self, cmp, background="plain", video=None, size=None, pose_size=None):
        self.cmp, self.background = cmp, background
        if background == "video":
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                raise FileNotFoundError(f"cannot open video {video}")
            self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pw, ph = pose_size
            self.sx, self.sy = self.W / pw, self.H / ph   # pose px -> video px, if resized
            self._next = None
        else:
            self.W, self.H = size or (960, 960)

    def frame(self, i):
        """Background image + (new_px, ghost_px, px_per_torso, hip_px) for new sample i."""
        c, new = self.cmp, self.cmp.new
        if self.background == "video":
            f = new.frames[i]
            if self._next != f:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = self.cap.read()
            if not ok:
                raise RuntimeError(f"could not read frame {f} from video")
            self._next = f + 1
            s = np.array([self.sx, self.sy])
            new_px = new.pts[i] * s
            ghost_px = denormalize(c.ghost[i], new.hip[i], new.torso, new.direction) * s
            return img, new_px, ghost_px, new.torso * (self.sx + self.sy) / 2, new.hip[i] * s
        img = np.full((self.H, self.W, 3), render.PLAIN_BGR, np.uint8)
        scale = 0.2 * self.H
        center = np.array([self.W / 2, 0.48 * self.H])
        return img, new.norm[i] * scale + center, c.ghost[i] * scale + center, scale, center

    def close(self):
        if self.background == "video":
            self.cap.release()


def draw(canvas, i, max_dev):
    c = canvas.cmp
    new, old = c.new, c.old
    img, new_px, ghost_px, scale, hip_px = canvas.frame(i)
    fs = 0.7 * canvas.H / 1080 if canvas.background == "video" else 0.6 * canvas.H / 960
    render.draw_ghost(img, ghost_px, c.ghost_conf[i], scale, c.min_conf)
    render.draw_new(img, new_px, new.conf[i], c.dev[i], scale, c.min_conf, max_dev)
    render.draw_labels(img, new_px, c.dev[i], c.top_joints(i), hip_px, fs, max_dev)
    j = c.old_idx[i]
    when = "CONTACT" if i == new.contact else f"{(i - new.contact) / new.fps:+.3f} s from contact"
    lines = [
        f"NEW {new.rep_id} ({new.foot} foot)  f={new.frames[i]}",
        f"GHOST {old.rep_id} ({old.foot} foot)  f={old.frames[j]}"
        + ("  [mirrored L/R]" if c.swapped else ""),
        when,
    ]
    render.draw_hud(img, lines, max_dev, fs)
    return img


def write_png(canvas, path, max_dev):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), draw(canvas, canvas.cmp.new.contact, max_dev)):
        raise RuntimeError(f"could not write {path}")


def write_video(canvas, path, max_dev, fps, hold_contact=0.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.W, canvas.H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for i in range(len(canvas.cmp.new.frames)):
            img = draw(canvas, i, max_dev)
            repeat = 1 + (int(round(hold_contact * fps)) if i == canvas.cmp.new.contact else 0)
            for _ in range(repeat):
                writer.write(img)
    finally:
        writer.release()


def contact_summary(cmp):
    i = cmp.new.contact
    return [(CORE[j], float(cmp.dev[i, j])) for j in cmp.top_joints(i)]
