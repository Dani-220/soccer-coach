"""python -m pytest contract/tools/test_body_height.py  (from the repo root)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from body_height import frame_height, rep_body_height, video_body_height


def kp(h, conf=0.9, nose_y=100.0):
    """Keypoints for a standing player whose nose-to-lower-ankle height is h px."""
    return {"nose": [0, nose_y, conf], "left_ankle": [0, nose_y + h, conf], "right_ankle": [0, nose_y + h - 5, conf]}


def frames_of(hs, conf=0.9):
    return [{"f": i, "keypoints": kp(h, conf) if h else {}} for i, h in enumerate(hs)]


def test_frame_height_skips_low_confidence_and_missing_player():
    assert frame_height(kp(200)) == 200
    assert frame_height(kp(200, conf=0.1)) is None
    assert frame_height({}) is None


def test_uses_only_the_last_0_3_s_before_plant():
    # Player runs away from the camera: 400 px tall at the start, 240 px at the plant (30 fps).
    hs = [400 - 5 * i for i in range(33)] + [240] * 10
    rep = {"start": 0, "plant": 40, "contact": 42, "end": 42}
    assert rep_body_height(frames_of(hs), rep, fps=30) == 240          # window = frames 31..40
    assert video_body_height(frames_of(hs)) > 280                       # a whole-video median is biased high


def test_falls_back_to_contact_when_there_is_no_plant():
    hs = [300] * 20 + [200] * 10
    assert rep_body_height(frames_of(hs), {"start": 0, "contact": 29, "end": 29}, fps=30) == 200


def test_widens_window_then_falls_back_to_video():
    hs = [210] * 10 + [None] * 10
    frames = frames_of(hs)
    # No usable frames in the 0.3 s window (frames 10..19 have no player) -> widen to [start, P].
    assert rep_body_height(frames, {"start": 0, "plant": 19, "contact": 19, "end": 19}, fps=30) == 210
    # Rep window entirely empty -> video-wide value.
    assert rep_body_height(frames, {"start": 12, "plant": 19, "contact": 19, "end": 19}, fps=30) == 210


def test_partial_body_frames_with_low_confidence_are_ignored():
    # Entering the frame: ankles cut off (low confidence) and a tiny apparent height.
    frames = [{"f": i, "keypoints": kp(90, conf=0.05)} for i in range(5)] + frames_of([220] * 10)[5:]
    frames += [{"f": i + 10, "keypoints": kp(220)} for i in range(5)]
    assert rep_body_height(frames, {"start": 0, "plant": 14, "contact": 15, "end": 15}, fps=30) == 220
