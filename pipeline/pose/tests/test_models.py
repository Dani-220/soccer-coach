"""Runs the real MediaPipe + YOLO models. Downloads weights on first run into SOCCER_COACH_MODELS.

Set POSE_TEST_VIDEO=/path/to/clip.mp4 to also run on real footage.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import make_video
from pipeline.pose import __main__ as cli
from pipeline.pose.keypoints import CORE_NAMES

pytest.importorskip("mediapipe")
ultralytics = pytest.importorskip("ultralytics")
pytestmark = pytest.mark.models

FAST = ["--pose-model", "full", "--ball-model", "yolo11n.pt", "--ball-imgsz", "640"]


def test_real_models_on_person_and_empty_frames(tmp_path, validate):
    from ultralytics.utils import ASSETS

    person = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    person = person[: person.shape[0] // 2 * 2, : person.shape[1] // 2 * 2]
    empty = np.full_like(person, 40)
    frames = [person] * 8 + [empty] * 4 + [person] * 8
    video = make_video(tmp_path / "bus.mp4", frames, fps=30)

    out = tmp_path / "out"
    cli.main(["--in", str(video), "--out", str(out / "pose.json"), "--debug", *FAST])
    code, msg = validate(out)
    assert code == 0, msg

    pose = json.loads((out / "pose.json").read_text())
    ball = json.loads((out / "ball.json").read_text())
    assert len(pose["frames"]) == len(ball["frames"]) == 20
    for fr, br in zip(pose["frames"], ball["frames"]):
        if 8 <= fr["f"] < 12:
            assert fr["keypoints"] == {} and br["ball"] is None
    detected = [fr for fr in pose["frames"] if fr["keypoints"]]
    assert len(detected) >= 12
    assert all(set(CORE_NAMES) <= set(fr["keypoints"]) for fr in detected)
    assert (out / "bus_debug.mp4").stat().st_size > 0


@pytest.mark.skipif(not os.environ.get("POSE_TEST_VIDEO"), reason="set POSE_TEST_VIDEO to a real clip")
def test_real_clip(tmp_path, validate):
    out = tmp_path / "real"
    cli.main(["--in", os.environ["POSE_TEST_VIDEO"], "--out", str(out / "pose.json"), "--debug"])
    code, msg = validate(out)
    assert code == 0, msg
