"""Run from the repo root: python -m pytest pipeline/overlay/tests"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.overlay.__main__ import main
from pipeline.overlay.align import align, dtw, frame_cost
from pipeline.overlay.overlay import compare, contact_summary
from pipeline.overlay.skeleton import IDX, load_rep

EX = Path(__file__).resolve().parents[3] / "contract" / "examples"
POSE, REPS = EX / "pose.example.json", EX / "reps.example.json"


@pytest.fixture(scope="module")
def data():
    return json.loads(POSE.read_text()), json.loads(REPS.read_text())


def _mid(n, a, b):
    return (n[:, IDX[a]] + n[:, IDX[b]]) / 2


def test_normalized_hip_at_origin_and_unit_torso(data):
    r = load_rep(*data, "r001")
    assert np.allclose(_mid(r.norm, "left_hip", "right_hip"), 0, atol=1e-9)
    torso = np.linalg.norm(_mid(r.norm, "left_shoulder", "right_shoulder"), axis=1)
    assert np.median(torso) == pytest.approx(1.0)
    assert r.direction == 1


def test_direction_detected_and_mirror_normalizes_identically(data):
    pose, reps = data
    flipped = copy.deepcopy(pose)
    for fr in flipped["frames"]:
        for kp in fr["keypoints"].values():
            kp[0] = pose["width"] - kp[0]
    a, b = load_rep(pose, reps, "r001"), load_rep(flipped, reps, "r001")
    assert b.direction == -1
    assert np.allclose(a.norm, b.norm)


def test_frames_without_player_are_filled_with_zero_confidence(data):
    pose, reps = copy.deepcopy(data[0]), data[1]
    for fr in pose["frames"][50:55]:
        fr["keypoints"] = {}
    r = load_rep(pose, reps, "r001")
    assert not np.isnan(r.norm).any()
    assert (r.conf[50:55] == 0).all() and (r.conf[49] > 0).all()


def test_dtw_identity_is_diagonal():
    a = np.random.default_rng(0).normal(size=(30, 17, 2))
    c = np.ones((30, 17))
    assert dtw(frame_cost(a, c, a, c)) == [(i, i) for i in range(30)]


def test_contact_anchored_and_time_warp_recovered(data):
    new = load_rep(*data, "r001")
    # Old rep = new rep played at half speed before contact, normal speed after.
    pre = np.repeat(new.norm[:new.contact], 2, axis=0)
    old_norm = np.concatenate([pre, new.norm[new.contact:]])
    old_conf = np.concatenate([np.repeat(new.conf[:new.contact], 2, axis=0), new.conf[new.contact:]])
    old_contact = len(pre)
    m = align(new, old_norm, old_conf, old_contact)
    assert m[new.contact] == [old_contact]
    for i in range(new.contact):
        assert set(m[i]) <= {2 * i - 1, 2 * i, 2 * i + 1}
    for i in range(new.contact, len(new.frames)):
        assert m[i] == [old_contact + i - new.contact]


def test_rep_vs_itself_has_no_deviation(data):
    r = load_rep(*data, "r001")
    c = compare(r, r)
    assert not c.swapped
    assert np.allclose(c.dev, 0)


def test_bad_rep_deviates_most_at_kicking_foot(data):
    c = compare(load_rep(*data, "r002"), load_rep(*data, "r001"))
    assert c.swapped  # right-foot ghost relabeled to match the left-foot rep
    names = [n for n, _ in contact_summary(c)]
    assert len(names) == 3 and names[0].startswith("left_")


def _frames(p):
    cap = cv2.VideoCapture(str(p))
    n, w, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return n, int(w), int(h)


def test_cli_plain_background(tmp_path):
    out = tmp_path / "o" / "overlay.mp4"
    main(["--in", str(POSE), str(REPS), "--out", str(out), "--size", "640x480", "--hold-contact", "0"])
    assert _frames(out) == (144, 640, 480)          # r002 is frames 228..371
    assert cv2.imread(str(out.with_suffix(".png"))).shape == (480, 640, 3)


def test_cli_png_only(tmp_path):
    out = tmp_path / "contact.png"
    main(["--in", str(POSE), str(REPS), str(POSE), str(REPS), "--new", "r002", "--old", "r001", "--out", str(out)])
    assert out.exists() and not (tmp_path / "contact.mp4").exists()


def test_cli_video_background_scaled(tmp_path):
    # Source video at quarter resolution of pose.json's 1920x1080; pose coords must be rescaled.
    src = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 120, (480, 270))
    for _ in range(396):
        w.write(np.full((270, 480, 3), 90, np.uint8))
    w.release()
    out = tmp_path / "overlay.mp4"
    main(["--in", str(POSE), str(REPS), "--background", "video", "--video", str(src),
          "--out", str(out), "--hold-contact", "0"])
    assert _frames(out) == (144, 480, 270)


def test_cli_missing_video_errors(tmp_path):
    with pytest.raises(SystemExit):
        main(["--in", str(POSE), str(REPS), "--background", "video", "--out", str(tmp_path / "o.mp4")])
