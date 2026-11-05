"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

A low-pass filter whose cutoff rises with speed: heavy smoothing when a point is still,
little lag when it moves fast (a kicking foot). Causal, so it can run frame by frame.
"""
import math


def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters one scalar signal sampled at timestamps in seconds."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff and d_cutoff must be > 0, beta >= 0")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._t = self._x = None
        self._dx = 0.0

    def __call__(self, x, t):
        if self._t is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            raise ValueError(f"timestamps must increase (got {self._t} then {t})")
        dx = (x - self._x) / dt
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


class KeypointSmoother:
    """One Euro filter on x and y of every named keypoint. Confidence is passed through unchanged.

    Defaults are tuned for pixel coordinates: ~1.5 Hz cutoff when still, rising by 0.01 Hz per px/s
    (about 20 Hz for a foot moving 2000 px/s). Call reset() whenever the player is lost so a filter
    never bridges a gap.
    """

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self._params = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._filters = {}

    def reset(self):
        self._filters.clear()

    def __call__(self, keypoints, t):
        out = {}
        for name, (x, y, conf) in keypoints.items():
            fx, fy = self._filters.setdefault(
                name, (OneEuroFilter(**self._params), OneEuroFilter(**self._params)))
            out[name] = (fx(x, t), fy(y, t), conf)
        return out
