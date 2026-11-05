"""BallTracker with scripted detections: a kicked ball in flight plus a stationary ball far away."""
import pytest

from pipeline.pose import detectors
from pipeline.pose.ball_tracker import BallTracker, feet_points
from pipeline.pose.detectors import crop_window

FPS = 30
FEET = [(180.0, 610.0), (215.0, 612.0)]
KICK = 10
MISS = {20, 21, 22}                      # 3-frame miss mid-flight
DISTRACTOR = (900.0, 620.0, 4.0, 0.9)    # stationary, far away, and MORE confident than the real ball


def real_ball(f):
    """At rest near the feet until KICK, then a parabola: 14 px/frame right, 10 up, gravity 0.5 px/frame^2."""
    n = max(0, f - KICK)
    return (200.0 + 14 * n, 600.0 - 10 * n + 0.25 * n * n, 8.0, 0.5)


def script(n_frames=40):
    frames = []
    for f in range(n_frames):
        dets = [DISTRACTOR]
        if f not in MISS:
            dets.append(real_ball(f))
        frames.append(dets)
    return frames


def run(frames, feet=FEET, tracker=None, redetect=None):
    t = tracker or BallTracker(FPS)
    return t, [t.step(dets, feet, redetect) for dets in frames]


def test_starts_from_ball_nearest_feet_not_most_confident():
    _, out = run(script(1))
    assert out[0] == real_ball(0)


def test_without_player_starts_from_most_confident():
    _, out = run(script(1), feet=[])
    assert out[0] == DISTRACTOR


def test_stationary_distractor_never_selected_while_kicked_ball_tracked():
    t, out = run(script())
    assert DISTRACTOR not in out
    assert t.tracking
    for f, b in enumerate(out):
        assert b == (None if f in MISS else real_ball(f)), f


def test_three_frame_miss_mid_flight_keeps_track():
    frames = script(30)
    decoy = (190.0, 605.0, 8.0, 0.95)    # a ball at the feet: a restarted track would pick it
    frames[23] = frames[23] + [decoy]
    t, out = run(frames)
    assert out[20:23] == [None, None, None]
    assert out[23] == real_ball(23)
    assert t.missed == 0


def test_output_is_never_a_prediction():
    frames = script()
    crop_hits = {21: [(real_ball(21)[0] + 1.5, real_ball(21)[1] - 2.0, 7.5, 0.2)]}  # crop finds it on one miss
    calls = []

    def redetect(cx, cy, side):
        calls.append(len(calls))
        return crop_hits.get(f_now, [])

    t = BallTracker(FPS)
    for f_now, dets in enumerate(frames):
        b = t.step(dets, FEET, redetect)
        if b is not None:
            assert any(b is d for d in dets + crop_hits.get(f_now, [])), f"frame {f_now}: {b} is not a detection"
        if f_now in MISS:
            if f_now == 21:
                assert b is crop_hits[21][0]
            else:
                assert b is None
    assert calls  # the crop re-detection ran on misses


def test_redetect_only_on_miss_and_centred_on_prediction():
    seen = []

    def redetect(cx, cy, side):
        seen.append((f_now, cx, cy, side))
        return []

    t = BallTracker(FPS)
    for f_now, dets in enumerate(script()):
        t.step(dets, FEET, redetect)
    assert [s[0] for s in seen] == sorted(MISS)
    f, cx, cy, side = seen[0]
    tx, ty, r, _ = real_ball(f)
    assert abs(cx - tx) < 2 * r and abs(cy - ty) < 2 * r
    assert side == max(4 * 2 * r, 64)


def test_track_dropped_after_half_a_second():
    frames = script(KICK) + [[DISTRACTOR]] * 16 + [[(400.0, 300.0, 8.0, 0.4), DISTRACTOR]]
    t = BallTracker(FPS)
    for f, dets in enumerate(frames[:KICK + 15]):
        t.step(dets, FEET)
    assert t.tracking and t.missed == 15              # 0.5 s at 30 fps: still tracking
    assert t.step(frames[KICK + 15], FEET) is None
    assert not t.tracking                             # 16th miss drops it
    # A new track starts from the detection nearest the feet, not the far, confident one.
    assert t.step(frames[-1], FEET) == (400.0, 300.0, 8.0, 0.4)


def test_gate_grows_with_speed_and_misses():
    t = BallTracker(FPS)
    t.step([real_ball(0)], FEET)
    t._predict()
    g_rest = t.gate()
    t.x[2:] = [20.0, 0.0]
    g_fast = t.gate()
    t.missed = 3
    assert g_rest < g_fast < t.gate()


def test_feet_points():
    kps = {"left_ankle": (1, 2, 0.9), "right_foot_index": (3, 4, 0.2), "nose": (0, 0, 1)}
    assert sorted(feet_points(kps)) == [(1, 2), (3, 4)]
    assert feet_points({}) == []


@pytest.mark.parametrize("cx,cy,side,expected", [
    (300, 400, 64, (268, 368, 64)),      # inside
    (10, 10, 64, (0, 0, 64)),            # clamped at top-left
    (570, 1020, 64, (512, 960, 64)),     # clamped at bottom-right of 576x1024
    (300, 400, 2000, (0, 112, 576)),     # never larger than the frame
])
def test_crop_window(cx, cy, side, expected):
    assert crop_window(cx, cy, side, 576, 1024) == expected


def test_detect_crop_maps_back_to_full_frame():
    import numpy as np

    det = object.__new__(detectors.YoloBallDetector)
    det._imgsz = 1280
    seen = {}

    def fake_predict(img):
        seen["shape"] = img.shape
        return [(640.0, 320.0, 40.0, 0.5)]  # centre-x, quarter-height in the upscaled crop

    det._predict = fake_predict
    frame = np.zeros((1024, 576, 3), np.uint8)
    (x, y, r, c), = det.detect_crop(frame, 300, 400, 64)
    assert seen["shape"][:2] == (1280, 1280)
    assert (x, y, r, c) == pytest.approx((268 + 32, 368 + 16, 2.0, 0.5))
