"""Find shot reps in pose.json + ball.json and build reps.json (Workstream C).

Per rep:
  contact  the last frame the ball is at rest before its speed jumps, while a foot touches it: the
           ankle->toe segment within `foot_ball_max_gap` H of the ball's surface (or, when the toe
           isn't confidently detected, the ankle within `ankle_ball_max_gap`). When the foot hides
           the ball, the contact frame is back-projected from the ball's flight speed. When the
           detector loses the ball for good, the contact is the first frame a fast-moving foot
           touches it, provided the ball doesn't reappear at its resting spot.
  foot     the foot nearest the ball at contact that is also moving fastest.
  plant    the first frame of the support ankle's final stationary period before contact.
  end      the first frame after contact from which the kicking ankle stays still.
  start    contact minus `approach_s`, clamped to the player's visible segment and the previous rep.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RepsConfig

SCHEMA_VERSION = "1.0"
BODY_HEIGHT_FRAMES = 30   # CONTRACT.md: median over the first 30 frames with a player
_POINTS = ("nose", "left_ankle", "right_ankle", "left_foot_index", "right_foot_index")


def other_foot(foot: str) -> str:
    return "right" if foot == "left" else "left"


@dataclass
class Tracks:
    """Per-frame arrays; missing points are NaN."""
    fps: float
    has_player: np.ndarray        # (n,) bool
    points: dict[str, np.ndarray] # name -> (n, 2)
    ball: np.ndarray              # (n, 2)
    ball_r: np.ndarray            # (n,) radius px

    @property
    def n(self) -> int:
        return len(self.has_player)

    def ankle(self, foot: str) -> np.ndarray:
        return self.points[f"{foot}_ankle"]

    def toe(self, foot: str) -> np.ndarray:
        return self.points[f"{foot}_foot_index"]


@dataclass
class Contact:
    frame: int
    rest_pos: np.ndarray   # resting ball position, px
    rest_r: float          # resting ball radius, px
    ball_speed: float      # H/s just after contact; NaN when the ball was never seen leaving
    seen_leaving: bool = True


@dataclass
class FootCall:
    foot: str
    confidence: float      # 0.5 = coin flip, 1.0 = both cues agree decisively


def load_tracks(pose: dict, ball: dict, cfg: RepsConfig) -> Tracks:
    n = len(pose["frames"])
    if len(ball["frames"]) != n:
        raise ValueError(f"pose has {n} frames but ball has {len(ball['frames'])}")
    has_player = np.zeros(n, bool)
    points = {k: np.full((n, 2), np.nan) for k in _POINTS}
    for fr in pose["frames"]:
        kp = fr["keypoints"]
        if not kp:
            continue
        has_player[fr["f"]] = True
        for name in _POINTS:
            p = kp.get(name)
            min_conf = cfg.min_toe_conf if name.endswith("foot_index") else cfg.min_keypoint_conf
            if p is not None and p[2] >= min_conf:
                points[name][fr["f"]] = p[:2]
    b, r = np.full((n, 2), np.nan), np.full(n, np.nan)
    for fr in ball["frames"]:
        det = fr["ball"]
        if det is not None and det[3] >= cfg.min_ball_conf:
            b[fr["f"]], r[fr["f"]] = det[:2], det[2]
    return Tracks(float(pose["fps"]), has_player, points, b, r)


def body_height(tr: Tracks) -> float:
    """Nose to the lower ankle, median over the first 30 frames with a player (px)."""
    lower_ankle_y = np.fmax(tr.ankle("left")[:, 1], tr.ankle("right")[:, 1])
    h = lower_ankle_y - tr.points["nose"][:, 1]
    h = h[tr.has_player & np.isfinite(h)][:BODY_HEIGHT_FRAMES]
    if len(h) == 0:
        raise ValueError("no frame with a nose and an ankle; can't compute body height")
    return float(np.median(h))


def _dist(a, b) -> np.ndarray:
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def _segment_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance from point p to each segment a[i]->b[i]; NaN where either end is missing."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
    return _dist(a + t[:, None] * ab, p)


def foot_ball_dists(tr: Tracks, foot: str, lo: int, hi: int, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame in [lo, hi]: (ankle->toe segment, ankle) distance to the ball center, px.
    The segment distance is NaN where the toe isn't confidently detected."""
    ank, toe = tr.ankle(foot)[lo:hi + 1], tr.toe(foot)[lo:hi + 1]
    return _segment_dist(ank, toe, center), _dist(ank, center)


def touch_excess(tr: Tracks, foot: str, lo: int, hi: int, center: np.ndarray, radius: float,
                 H: float, cfg: RepsConfig) -> np.ndarray:
    """Per frame in [lo, hi]: how far (H) the foot is outside touching range of the ball's surface.
    <= 0 means touching; NaN where the ankle is missing."""
    seg, ank = foot_ball_dists(tr, foot, lo, hi, center)
    return np.where(np.isfinite(seg),
                    (seg - radius) / H - cfg.foot_ball_max_gap,
                    (ank - radius) / H - cfg.ankle_ball_max_gap)


def foot_touches_ball(tr: Tracks, lo: int, hi: int, center: np.ndarray, radius: float, H: float, cfg: RepsConfig) -> bool:
    return any(np.any(touch_excess(tr, ft, lo, hi, center, radius, H, cfg) <= 0) for ft in ("left", "right"))


def _resting(tr: Tracks, vis: np.ndarray, a: int, H: float, cfg: RepsConfig):
    """(position, radius) of the ball if it was at rest up to frame a, else None."""
    win = vis[(vis >= a - cfg.frames(cfg.rest_window_s, tr.fps)) & (vis <= a)]
    if len(win) < cfg.min_rest_samples:
        return None
    rest = np.median(tr.ball[win], axis=0)
    if _dist(tr.ball[win], rest).max() > cfg.rest_radius * H:
        return None
    return rest, float(np.median(tr.ball_r[win]))


def _vanish_contact(tr: Tracks, a: int, b: int | None, rest, rest_r, H: float, cfg: RepsConfig) -> Contact | None:
    """Ball last seen at rest in frame a, next seen at b (None = never). Was it kicked?"""
    fps = tr.fps
    if b is not None and b - a <= cfg.frames(cfg.vanish_confirm_s, fps) and _dist(tr.ball[b], rest) <= cfg.rest_radius * H:
        return None                         # a detector dropout: the ball is back where it was
    # The ball leaves at the first touch by a moving foot (later frames see the foot pass through
    # where it was). The planted support foot often touches the ball in side-on video, so a touch
    # only counts when that foot is moving fast.
    hi = min(tr.n - 1, a + cfg.frames(cfg.max_ball_gap_s, fps), b - 1 if b is not None else tr.n - 1)
    k = cfg.frames(cfg.foot_speed_window_s, fps)
    ex = {ft: touch_excess(tr, ft, a, hi, rest, rest_r, H, cfg) for ft in ("left", "right")}
    for frame in range(int(a), hi + 1):
        for ft in ("left", "right"):
            if ex[ft][frame - a] <= 0 and _speed(tr.ankle(ft), max(0, frame - k), frame) * fps / H >= cfg.vanish_min_foot_speed:
                return Contact(frame, rest, rest_r, np.nan, seen_leaving=False)
    return None                             # no moving foot reached the ball


def find_contacts(tr: Tracks, H: float, cfg: RepsConfig) -> list[Contact]:
    fps = tr.fps
    vis = np.flatnonzero(np.isfinite(tr.ball[:, 0]))
    max_gap = cfg.frames(cfg.max_ball_gap_s, fps) + 1   # +1: a gap of k hidden frames spans k+1
    touch_win = cfg.frames(cfg.touch_window_s, fps)
    min_gap = cfg.frames(cfg.min_rep_gap_s, fps)

    contacts, next_allowed = [], 0
    for i, a in enumerate(vis):            # a: last resting detection
        if a < next_allowed:
            continue
        resting = _resting(tr, vis, a, H, cfg)
        if resting is None:
            continue
        rest, rest_r = resting
        b = vis[i + 1] if i + 1 < len(vis) else None   # b: the next detection, if any
        if b is None or b - a > max_gap:
            contact = _vanish_contact(tr, a, b, rest, rest_r, H, cfg)
            if contact is not None:
                contacts.append(contact)
                next_allowed = contact.frame + min_gap
            continue
        d_b = float(_dist(tr.ball[b], rest))
        if d_b <= cfg.rest_radius * H:
            continue                        # still at rest

        # Flight speed from the next detection when there is one; it must keep moving away.
        c = vis[i + 2] if i + 2 < len(vis) and vis[i + 2] - b <= max_gap else None
        if c is not None:
            if _dist(tr.ball[c], rest) <= d_b:
                continue                    # a one-frame glitch, not a kick
            v = float(_dist(tr.ball[c], tr.ball[b])) / (c - b)
        else:
            v = d_b / (b - a)
        speed = v * fps / H
        if speed < cfg.jump_speed:
            continue

        # Back-project to when the ball left its resting spot (handles occlusion at contact).
        frame = int(np.clip(round(b - d_b / v), a, b - 1))

        lo, hi = max(0, frame - touch_win), min(tr.n - 1, frame + touch_win)
        if not foot_touches_ball(tr, lo, hi, rest, rest_r, H, cfg):
            continue                        # nobody's foot was at the ball

        contacts.append(Contact(frame, rest, rest_r, speed))
        next_allowed = frame + min_gap
    return contacts


def call_foot(tr: Tracks, contact: Contact, cfg: RepsConfig, strong_foot: str | None = None) -> FootCall:
    """Score left vs right on foot-to-ball distance and ankle speed; positive = left.

    Distances use the ankle->toe segment in frames where both toes are detected, else both ankles,
    so the two feet are always compared like for like.
    """
    c, fps = contact.frame, tr.fps
    w = cfg.frames(cfg.foot_window_s, fps)
    k = cfg.frames(cfg.foot_speed_window_s, fps)
    lo, hi = max(0, c - w), min(tr.n - 1, c + w)

    seg, ank = zip(*(foot_ball_dists(tr, ft, lo, hi, contact.rest_pos) for ft in ("left", "right")))
    both_toes = np.isfinite(seg[0]) & np.isfinite(seg[1])
    dist, speed = {}, {}
    for i, ft in enumerate(("left", "right")):
        d = np.where(both_toes, seg[i], ank[i])
        dist[ft] = float(np.nanmin(d)) if np.any(np.isfinite(d)) else np.nan
        speed[ft] = _speed(tr.ankle(ft), max(0, c - k), c)

    score, weight = 0.0, 0.0
    if np.isfinite(dist["left"]) and np.isfinite(dist["right"]) and dist["left"] + dist["right"] > 0:
        score += cfg.foot_dist_weight * (dist["right"] - dist["left"]) / (dist["right"] + dist["left"])
        weight += cfg.foot_dist_weight
    if np.isfinite(speed["left"]) and np.isfinite(speed["right"]) and speed["left"] + speed["right"] > 0:
        score += cfg.foot_speed_weight * (speed["left"] - speed["right"]) / (speed["left"] + speed["right"])
        weight += cfg.foot_speed_weight
    score = score / weight if weight else 0.0
    if strong_foot in ("left", "right"):
        score += cfg.strong_foot_prior * (1 if strong_foot == "left" else -1)
    score = float(np.clip(score, -1, 1))

    if score == 0:
        foot = strong_foot if strong_foot in ("left", "right") else "right"
    else:
        foot = "left" if score > 0 else "right"
    return FootCall(foot, 0.5 + 0.5 * abs(score))


def _speed(p: np.ndarray, f0: int, f1: int) -> float:
    """Average speed (px/frame) between the first and last valid points in [f0, f1]."""
    idx = np.flatnonzero(np.isfinite(p[f0:f1 + 1, 0])) + f0
    if len(idx) < 2:
        return np.nan
    return float(_dist(p[idx[-1]], p[idx[0]])) / (idx[-1] - idx[0])


def segment_bounds(has_player: np.ndarray, f: int, max_gap: int) -> tuple[int, int]:
    """First and last frame of the player-visible run around f, bridging short dropouts."""
    vis = np.flatnonzero(has_player)
    if len(vis) == 0:
        return f, f
    breaks = np.flatnonzero(np.diff(vis) > max_gap + 1)
    starts = np.concatenate([[vis[0]], vis[breaks + 1]])
    ends = np.concatenate([vis[breaks], [vis[-1]]])
    j = int(np.searchsorted(starts, f, side="right")) - 1
    j = max(j, 0)
    return int(starts[j]), int(ends[j])


def find_plant(tr: Tracks, support: str, contact: int, latest: int, earliest: int, H: float,
               cfg: RepsConfig) -> int | None:
    p = tr.ankle(support)
    # The support foot is planted through contact, so its position and jitter come from around it.
    win = p[max(0, contact - cfg.frames(cfg.plant_ref_window_s, tr.fps)):
            min(latest, contact + cfg.frames(cfg.plant_jitter_window_s, tr.fps)) + 1]
    win = win[np.isfinite(win[:, 0])]
    if len(win) == 0:
        return None
    ref = np.median(win, axis=0)
    radius = cfg.plant_still_radius * H
    if len(win) >= 3:
        sigma = np.median(_dist(win, ref)) / 1.1774   # median of a 2-D Gaussian's radius = 1.1774 sigma
        radius = min(max(radius, cfg.plant_jitter_mult * sigma), max(radius, cfg.plant_still_radius_max * H))
    lo = max(earliest, contact - cfg.frames(cfg.plant_max_lookback_s, tr.fps))
    plant, outliers = None, 0
    for t in range(contact, lo - 1, -1):
        if not np.isfinite(p[t, 0]):
            continue
        if _dist(p[t], ref) > radius:
            outliers += 1
            if outliers > cfg.still_max_outliers:
                break
        else:
            plant, outliers = t, 0
    return plant


def find_end(tr: Tracks, kick: str, contact: int, latest: int, H: float, cfg: RepsConfig) -> tuple[int, bool]:
    """Return (end frame, whether the ankle was actually seen stopping)."""
    p = tr.ankle(kick)
    hold = cfg.frames(cfg.end_hold_s, tr.fps)
    hi = min(latest, contact + cfg.frames(cfg.max_follow_through_s, tr.fps))
    for t in range(contact + 1, hi - hold + 1):
        if not np.isfinite(p[t, 0]):
            continue
        seg = p[t:t + hold + 1]
        seg = seg[np.isfinite(seg[:, 0])]
        if len(seg) < 2:
            continue
        if _dist(seg, np.median(seg, axis=0)).max() <= cfg.end_still_radius * H:
            return t, True
    return hi, False


def detect_reps(pose: dict, ball: dict, player: dict | None = None, cfg: RepsConfig | None = None) -> dict:
    cfg = cfg or RepsConfig()
    tr = load_tracks(pose, ball, cfg)
    H = body_height(tr)
    strong = (player or {}).get("strong_foot")
    pose_gap = cfg.frames(cfg.max_pose_gap_s, tr.fps)

    reps, prev_end = [], -1
    for contact in find_contacts(tr, H, cfg):
        c = contact.frame
        seg_start, seg_end = segment_bounds(tr.has_player, c, pose_gap)
        start = max(seg_start, c - cfg.frames(cfg.approach_s, tr.fps), prev_end + 1, 0)
        if start > c:
            continue                        # overlaps the previous rep's follow-through

        call = call_foot(tr, contact, cfg, strong)
        plant = find_plant(tr, other_foot(call.foot), c, seg_end, start, H, cfg)
        end, end_seen = find_end(tr, call.foot, c, seg_end, H, cfg)

        conf = call.confidence
        if plant is None:
            conf *= cfg.missing_phase_penalty
        if not end_seen:
            conf *= cfg.missing_phase_penalty
        if not contact.seen_leaving:
            conf *= cfg.vanish_penalty

        frames = {"start": start}
        if plant is not None:
            frames["plant"] = plant
        frames.update(contact=c, end=end)
        reps.append({
            "rep_id": f"r{len(reps) + 1:03d}",
            "move": cfg.move,
            "foot": call.foot,
            "frames": frames,
            "confidence": round(float(conf), 3),
        })
        prev_end = end

    return {
        "schema_version": SCHEMA_VERSION,
        "video": pose["video"],
        "player_id": (player or {}).get("player_id", cfg.default_player_id),
        "reps": reps,
    }
