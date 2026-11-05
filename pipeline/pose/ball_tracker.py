"""One ball track per video: constant-velocity Kalman filter + gating.

The filter only decides WHICH detection is the ball. Its output is always one of the real detections
passed in (never the prediction), or None. Contract: ball.json holds detections, not estimates.
"""
import math

import numpy as np

FEET = [f"{s}_{p}" for s in ("left", "right") for p in ("ankle", "heel", "foot_index")]

# State [x, y, vx, vy] in px and px/frame; one predict per video frame.
_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
_G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], float)  # acceleration -> state


class BallTracker:
    """Gate radius after n frames without a hit (n = 1 on consecutive hits), in px:

        gate = gate_radii * r + n * (speed_frac * |v| + kick_speed * r / fps)

    r is the ball's radius in px, so the gate scales with distance from the camera. The speed term
    covers velocity error, and the kick term covers a sudden launch or bounce: kick_speed is in ball radii per
    second (120 r/s is about 13 m/s). The track is dropped after `max_missing_s` seconds without a hit.
    With no track, the detection nearest the player's feet starts one (highest confidence if no player).
    """

    def __init__(self, fps, max_missing_s=0.5, gate_radii=3.0, speed_frac=0.25, kick_speed=120.0,
                 crop_diameters=4.0, min_crop=64, min_radius=3.0):
        self.fps = fps
        self.max_missing = max(1, round(max_missing_s * fps))
        self.gate_radii, self.speed_frac, self.kick_speed = gate_radii, speed_frac, kick_speed
        self.crop_diameters, self.min_crop, self.min_radius = crop_diameters, min_crop, min_radius
        self.x = None              # Kalman state, None when there is no track
        self.missed = 0
        self.radius = None
        self.last_gate = None      # (pred_x, pred_y, gate_px) for the debug video

    @property
    def tracking(self):
        return self.x is not None

    def gate(self):
        r = max(self.radius, self.min_radius)
        n = self.missed + 1
        speed = math.hypot(self.x[2], self.x[3])
        return self.gate_radii * r + n * (self.speed_frac * speed + self.kick_speed * r / self.fps)

    def crop_side(self):
        return max(self.crop_diameters * 2 * max(self.radius, self.min_radius), self.min_crop)

    def step(self, detections, feet=(), redetect=None):
        """detections: [(x, y, r, conf)] for this frame. feet: [(x, y)] of the tracked player.
        redetect(cx, cy, side) -> detections from a crop, used only when a track exists and missed.
        Returns one of the given detections, or None."""
        if self.x is None:
            self.last_gate = None
            pick = self._start(detections, feet)
            if pick is not None:
                self._init(pick)
            return pick

        self._predict()
        px, py = self.x[0], self.x[1]
        gate = self.gate()
        self.last_gate = (px, py, gate)
        pick = _nearest(detections, px, py, gate)
        if pick is None and redetect is not None:
            pick = _nearest(redetect(px, py, self.crop_side()), px, py, gate)

        if pick is None:
            self.missed += 1
            if self.missed > self.max_missing:
                self.x = None
        else:
            self._update(pick)
        return pick

    def _start(self, detections, feet):
        if not detections:
            return None
        if feet:
            return min(detections, key=lambda d: min(math.dist(d[:2], p) for p in feet))
        return max(detections, key=lambda d: d[3])

    def _init(self, det):
        x, y, r, _ = det
        self.radius = r
        self.x = np.array([x, y, 0.0, 0.0])
        rr = max(r, self.min_radius)
        self.P = np.diag([rr ** 2, rr ** 2, (4 * rr) ** 2, (4 * rr) ** 2])
        self.missed = 0

    def _predict(self):
        sigma_a = max(self.radius, self.min_radius)  # px/frame^2: about one radius per frame per frame
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + (sigma_a ** 2) * (_G @ _G.T)

    def _update(self, det):
        x, y, r, _ = det
        rr = max(r, self.min_radius)
        R = np.eye(2) * (0.3 * rr) ** 2
        innov = np.array([x, y]) - _H @ self.x
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(4) - K @ _H) @ self.P
        self.radius = 0.7 * self.radius + 0.3 * r
        self.missed = 0


def _nearest(detections, x, y, gate):
    best, best_d = None, gate
    for d in detections:
        dist = math.hypot(d[0] - x, d[1] - y)
        if dist <= best_d:
            best, best_d = d, dist
    return best


def feet_points(keypoints):
    """Foot landmarks of the tracked player as [(x, y)], or [] when there is no player."""
    return [keypoints[n][:2] for n in FEET if n in keypoints]
