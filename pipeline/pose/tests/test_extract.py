"""Pipeline tests with scripted fake detectors; outputs are checked with contract/tools/validate.py."""
import json

import cv2
import pytest

from conftest import blank_frames, make_video, write_docs
from pipeline.pose import __main__ as cli
from pipeline.pose import detectors
from pipeline.pose.extract import extract
from pipeline.pose.keypoints import CORE_NAMES, LANDMARK_NAMES
from pipeline.pose.tracking import PlayerTracker

N = 30
NO_PLAYER = set(range(10, 15))
NO_BALL = {f for f in range(N) if f % 3 == 0} | {29}
TWO_PEOPLE = {20, 21}


def person(cx, top, h=100.0, conf=0.8):
    """A crude upright skeleton: every landmark spread over height h, hips at 55%."""
    kps = {}
    for i, name in enumerate(LANDMARK_NAMES):
        y = top + h * i / (len(LANDMARK_NAMES) - 1)
        if name.endswith("_hip"):
            y = top + 0.55 * h
        kps[name] = (cx + (-3 if name.startswith("left") else 3), y, conf)
    return kps


class FakePose:
    def __init__(self):
        self.f = -1
        self.timestamps = []

    def detect(self, frame, ms):
        self.f += 1
        self.timestamps.append(ms)
        if self.f in NO_PLAYER:
            return []
        me = person(100 + 2 * self.f, 60, conf=1.3 if self.f == 3 else 0.8)
        if self.f in TWO_PEOPLE:
            return [person(280, 10, h=200), me]  # a bigger stranger far away comes first
        return [me]

    def close(self):
        pass


class FakeBall:
    def __init__(self):
        self.f = -1

    def detect(self, frame):
        self.f += 1
        if self.f in NO_BALL:
            return []
        if self.f == 5:
            return [(300, 20, 4, 0.95), (155.0, 200.0, 5.0, 0.4)]  # far, confident decoy + the tracked ball
        return [(150.0 + self.f, 200.0, 5.0, 0.7)]

    def close(self):
        pass


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "drill.mp4", blank_frames(N), fps=30)


@pytest.fixture
def docs(video):
    return extract(video, FakePose(), FakeBall())


def test_outputs_pass_contract_validator(tmp_path, docs, validate):
    code, out = validate(write_docs(tmp_path / "out", *docs))
    assert code == 0, out
    assert "cross-file checks" in out


def test_header_and_frames(docs):
    pose, ball = docs
    for d in docs:
        assert d["video"] == "drill.mp4" and d["fps"] == 30 and d["schema_version"] == "1.0"
        assert (d["width"], d["height"]) == (320, 240)
        assert [fr["f"] for fr in d["frames"]] == list(range(N))
    assert pose["player_track_id"] == 1


def test_missing_player_is_empty_and_present_player_has_all_core_points(docs):
    for fr in docs[0]["frames"]:
        if fr["f"] in NO_PLAYER:
            assert fr["keypoints"] == {}
        else:
            assert set(CORE_NAMES) <= set(fr["keypoints"])
            assert all(0 <= v[2] <= 1 for v in fr["keypoints"].values())


def test_missing_ball_is_null_never_interpolated(docs):
    for fr in docs[1]["frames"]:
        assert (fr["ball"] is None) == (fr["f"] in NO_BALL)
    assert docs[1]["frames"][5]["ball"] == [155, 200, 5, 0.4]  # the tracked ball, not the most confident


def test_filter_resets_after_gap(docs):
    # First frame after the gap must be the raw detection, not pulled toward pre-gap positions.
    raw = person(100 + 2 * 15, 60)
    assert docs[0]["frames"][15]["keypoints"]["nose"][:2] == [round(raw["nose"][0], 1), round(raw["nose"][1], 1)]


def test_tracker_ignores_bigger_stranger(docs):
    for f in TWO_PEOPLE:
        assert abs(docs[0]["frames"][f]["keypoints"]["nose"][0] - (100 + 2 * f)) < 10


def test_confidence_clamped(docs):
    assert docs[0]["frames"][3]["keypoints"]["nose"][2] == 1.0


def test_mediapipe_timestamps_strictly_increase(video):
    pose = FakePose()
    extract(video, pose, FakeBall(), fps_override=2000)  # 0.5 ms per frame would repeat if rounded naively
    assert all(b > a for a, b in zip(pose.timestamps, pose.timestamps[1:]))


def test_visibility_logits_become_probabilities():
    assert detectors._prob(0.7) == 0.7
    assert detectors._prob(None) == 0.0
    assert 0.99 < detectors._prob(5.0) <= 1.0 and 0 <= detectors._prob(-5.0) < 0.01


def test_tracker_reacquires_after_long_loss():
    t = PlayerTracker(reacquire_after=3)
    me, other = person(100, 60), person(280, 10, h=200)
    assert t.update([me])[0] is me
    for _ in range(3):
        assert t.update([other])[0] is None  # too far to be the same player
    pick, new = t.update([other])
    assert pick is other and new


@pytest.mark.parametrize("out,expected", [
    ("out/pose.json", "out/ball.json"),
    ("out/clip_pose.json", "out/clip_ball.json"),
    ("out/clip.json", "out/clip_ball.json"),
])
def test_default_ball_path(out, expected):
    assert str(cli.default_ball_path(out)) == expected


def test_cli_end_to_end_with_debug(tmp_path, video, validate, monkeypatch):
    monkeypatch.setattr(detectors, "MediaPipePoseDetector", lambda *a, **k: FakePose())
    monkeypatch.setattr(detectors, "YoloBallDetector", lambda *a, **k: FakeBall())
    out = tmp_path / "res"
    cli.main(["--in", str(video), "--out", str(out / "pose.json"), "--debug"])
    code, msg = validate(out)
    assert code == 0, msg
    pose = json.loads((out / "pose.json").read_text())
    cap = cv2.VideoCapture(str(out / "drill_debug.mp4"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == len(pose["frames"]) == N


def test_unreadable_video_errors(tmp_path):
    bad = tmp_path / "nope.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises((IOError, ValueError)):
        extract(bad, FakePose(), FakeBall())
