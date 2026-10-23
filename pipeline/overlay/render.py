"""OpenCV drawing: faint ghost skeleton, solid new skeleton colored by deviation, labels."""
import cv2
import numpy as np

from .skeleton import CORE, IDX

BONES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("nose", "neck"),
]
for _side in ("left", "right"):
    BONES += [(f"{_side}_{a}", f"{_side}_{b}") for a, b in (
        ("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"), ("hip", "knee"),
        ("knee", "ankle"), ("ankle", "heel"), ("ankle", "foot_index"), ("heel", "foot_index"))]

# BGR stops for the deviation color scale: green -> yellow -> red.
_STOPS = np.array([[80, 200, 60], [0, 215, 255], [40, 40, 230]], dtype=float)
GHOST_BGR = (235, 235, 235)
PLAIN_BGR = (32, 30, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def deviation_color(d, max_dev):
    t = float(np.clip(d / max_dev, 0.0, 1.0)) * (len(_STOPS) - 1)
    k = min(int(t), len(_STOPS) - 2)
    c = _STOPS[k] + (_STOPS[k + 1] - _STOPS[k]) * (t - k)
    return tuple(int(v) for v in c)


def confident(conf, min_conf):
    """Joints confident enough to draw. conf 0 (frame with no player) never is, whatever min_conf."""
    conf = np.asarray(conf)
    return (conf >= min_conf) & (conf > 0)


def on_screen(px, shape):
    H, W = shape[:2]
    return (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)


def _point(px, conf, name, min_conf):
    """Pixel position + visibility of a joint; 'neck' is the shoulder midpoint."""
    if name == "neck":
        a, b = IDX["left_shoulder"], IDX["right_shoulder"]
        return (px[a] + px[b]) / 2, bool(confident(min(conf[a], conf[b]), min_conf))
    j = IDX[name]
    return px[j], bool(confident(conf[j], min_conf))


def _ip(p):
    return int(round(p[0])), int(round(p[1]))


def draw_ghost(img, px, conf, scale, min_conf, alpha=0.35):
    layer = img.copy()
    th = max(2, int(scale * 0.06))
    for a, b in BONES:
        pa, va = _point(px, conf, a, min_conf)
        pb, vb = _point(px, conf, b, min_conf)
        if va and vb:
            cv2.line(layer, _ip(pa), _ip(pb), GHOST_BGR, th, cv2.LINE_AA)
    for j in range(len(CORE)):
        if confident(conf[j], min_conf):
            cv2.circle(layer, _ip(px[j]), th, GHOST_BGR, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def draw_new(img, px, conf, dev, scale, min_conf, max_dev):
    th = max(2, int(scale * 0.05))
    r = max(3, int(scale * 0.055))
    dev_n = dict(zip(CORE, dev))
    dev_n["neck"] = (dev_n["left_shoulder"] + dev_n["right_shoulder"]) / 2
    for a, b in BONES:
        pa, va = _point(px, conf, a, min_conf)
        pb, vb = _point(px, conf, b, min_conf)
        if va and vb:
            col = deviation_color((dev_n[a] + dev_n[b]) / 2, max_dev)
            cv2.line(img, _ip(pa), _ip(pb), (20, 20, 20), th + 2, cv2.LINE_AA)
            cv2.line(img, _ip(pa), _ip(pb), col, th, cv2.LINE_AA)
    for j in range(len(CORE)):
        if confident(conf[j], min_conf):
            cv2.circle(img, _ip(px[j]), r + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, _ip(px[j]), r, deviation_color(dev[j], max_dev), -1, cv2.LINE_AA)


def _text(img, s, org, fs, color=(245, 245, 245), th=1, bg=None, pad=4):
    (w, h), base = cv2.getTextSize(s, FONT, fs, th)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad), bg, -1)
    cv2.putText(img, s, (x, y), FONT, fs, color, th, cv2.LINE_AA)
    return w, h + base


def draw_labels(img, px, dev, joints, center, fs, max_dev):
    """Label `joints` (indices, most deviant first) with leader lines pointing outward from center."""
    th = max(1, int(fs * 2))
    placed = []
    for rank, j in enumerate(joints, 1):
        p = px[j]
        v = p - center
        n = np.linalg.norm(v)
        v = v / n if n > 1e-6 else np.array([1.0, 0.0])
        off = 60 * fs / 0.6
        anchor = p + v * off
        s = f"{rank}. {CORE[j].replace('_', ' ')}  {dev[j]:.2f}"
        (w, h), base = cv2.getTextSize(s, FONT, fs, th)
        H, W = img.shape[:2]
        x = float(np.clip(anchor[0] if v[0] >= 0 else anchor[0] - w, 6, W - w - 6))
        y0 = float(np.clip(anchor[1], h + 6, H - base - 6))
        # Nearest vertical slot (alternating down/up) that stays on screen and clears earlier labels.
        step = h + base + 12
        for k in range(41):
            y = y0 + step * ((k + 1) // 2) * (1 if k % 2 else -1)
            box = (x - 6, y - h - 6, x + w + 6, y + base + 6)
            on_screen = box[1] >= 0 and box[3] <= H
            if on_screen and not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                                     for b in placed):
                break
        else:
            y = y0
        placed.append((x - 6, y - h - 6, x + w + 6, y + base + 6))
        col = deviation_color(dev[j], max_dev)
        end = (int(x if v[0] >= 0 else x + w), int(y - h / 2))
        cv2.line(img, _ip(p), end, col, max(1, th), cv2.LINE_AA)
        cv2.circle(img, _ip(p), int(10 * fs / 0.6), col, max(1, th), cv2.LINE_AA)
        _text(img, s, (int(x), int(y)), fs, (255, 255, 255), th, bg=(25, 25, 25))


def draw_hud(img, lines, max_dev, fs):
    th = max(1, int(fs * 2))
    y = int(30 * fs / 0.6)
    for s in lines:
        _, h = _text(img, s, (16, y), fs, th=th, bg=(25, 25, 25))
        y += h + 12
    # Color bar legend.
    H = img.shape[0]
    bw, bh = int(220 * fs / 0.6), int(14 * fs / 0.6)
    x0, y0 = 16, H - bh - int(36 * fs / 0.6)
    for i in range(bw):
        cv2.line(img, (x0 + i, y0), (x0 + i, y0 + bh), deviation_color(max_dev * i / (bw - 1), max_dev), 1)
    _text(img, "0", (x0, y0 + bh + int(22 * fs / 0.6)), fs * 0.8, th=th)
    lbl = f">= {max_dev:g} torso"
    (w, _), _ = cv2.getTextSize(lbl, FONT, fs * 0.8, th)
    _text(img, lbl, (x0 + bw - w, y0 + bh + int(22 * fs / 0.6)), fs * 0.8, th=th)
    _text(img, "distance from ghost", (x0, y0 - 8), fs * 0.8, th=th)
