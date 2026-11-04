import random
import statistics

import pytest

from pipeline.pose.one_euro import KeypointSmoother, OneEuroFilter


def run(filt, xs, fps=120):
    return [filt(x, i / fps) for i, x in enumerate(xs)]


def test_first_sample_passes_through():
    assert OneEuroFilter()(123.4, 0.0) == 123.4


def test_constant_signal_unchanged():
    assert all(abs(v - 50.0) < 1e-9 for v in run(OneEuroFilter(1.5, 0.01), [50.0] * 100))


def test_reduces_jitter_on_still_point():
    rng = random.Random(0)
    raw = [500 + rng.gauss(0, 2.0) for _ in range(600)]
    out = run(OneEuroFilter(1.5, 0.01), raw)
    assert statistics.pstdev(out[100:]) < 0.4 * statistics.pstdev(raw[100:])


def test_beta_reduces_lag_on_fast_motion():
    ramp = [i * 20.0 for i in range(120)]  # 2400 px/s at 120 fps
    slow = run(OneEuroFilter(1.5, 0.0), ramp)
    fast = run(OneEuroFilter(1.5, 0.01), ramp)
    assert abs(fast[-1] - ramp[-1]) < 0.25 * abs(slow[-1] - ramp[-1])
    assert abs(fast[-1] - ramp[-1]) < 40  # under ~2 frames of lag


def test_reset_forgets_history():
    f = OneEuroFilter()
    run(f, [0.0] * 10)
    f.reset()
    assert f(999.0, 100.0) == 999.0


def test_timestamps_must_increase():
    f = OneEuroFilter()
    f(1.0, 0.5)
    with pytest.raises(ValueError):
        f(2.0, 0.5)


def test_keypoint_smoother_keeps_confidence_and_resets():
    s = KeypointSmoother()
    s({"nose": (10.0, 20.0, 0.9)}, 0.0)
    x, y, c = s({"nose": (30.0, 40.0, 0.4)}, 1 / 120)["nose"]
    assert c == 0.4 and 10 < x < 30 and 20 < y < 40
    s.reset()
    assert s({"nose": (30.0, 40.0, 0.4)}, 2 / 120) == {"nose": (30.0, 40.0, 0.4)}
