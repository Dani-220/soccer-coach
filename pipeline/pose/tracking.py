"""Pick one player and one ball per frame from the detector candidates. Never fabricates a detection."""
import math

from .keypoints import CORE_NAMES


def _bbox(kps):
    xs = [kps[n][0] for n in CORE_NAMES]
    ys = [kps[n][1] for n in CORE_NAMES]
    return min(xs), min(ys), max(xs), max(ys)


def _hip_mid(kps):
    (lx, ly, _), (rx, ry, _) = kps["left_hip"], kps["right_hip"]
    return (lx + rx) / 2, (ly + ry) / 2


class PlayerTracker:
    """Follows one player across frames.

    First sighting (or after the player has been lost for `reacquire_after` frames): take the
    largest person. Otherwise take the candidate whose hip midpoint is nearest the last one, but
    only if it moved less than `max_jump` body heights; a bigger jump means someone else, and the
    frame is reported as "no player" rather than switching people.
    """

    def __init__(self, max_jump=0.75, reacquire_after=15):
        self.max_jump, self.reacquire_after = max_jump, reacquire_after
        self._last = None          # (hip_mid, body_height) of the last accepted pose
        self._missed = 0

    def update(self, candidates):
        """Returns the tracked player's keypoints, or None. `reset_filter` is True when the
        returned pose does not continue the previous frame's track."""
        cands = [c for c in candidates if all(n in c for n in CORE_NAMES)]
        pick = None
        if self._last is not None and self._missed < self.reacquire_after:
            (hx, hy), body_h = self._last
            best = min(cands, key=lambda c: math.dist(_hip_mid(c), (hx, hy)), default=None)
            if best is not None and math.dist(_hip_mid(best), (hx, hy)) <= self.max_jump * body_h:
                pick = best
        elif cands:
            pick = max(cands, key=lambda c: _area(_bbox(c)))

        continues = pick is not None and self._last is not None and self._missed == 0
        if pick is None:
            self._missed += 1
        else:
            x0, y0, x1, y1 = _bbox(pick)
            self._last = (_hip_mid(pick), max(y1 - y0, 1.0))
            self._missed = 0
        return pick, not continues


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def pick_ball(candidates):
    """Highest-confidence ball, or None. No smoothing, no interpolation (contract: downstream's job)."""
    return max(candidates, key=lambda b: b[3], default=None)
