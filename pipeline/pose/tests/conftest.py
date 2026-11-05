import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
VALIDATOR = REPO / "contract" / "tools" / "validate.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "models: runs the real MediaPipe and YOLO models (downloads weights)")


def make_video(path, frames, fps=30):
    """Writes a list of BGR frames to an mp4 and returns the path."""
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert out.isOpened()
    for fr in frames:
        out.write(fr)
    out.release()
    return path


def blank_frames(n, w=320, h=240):
    return [np.full((h, w, 3), 40, np.uint8) for _ in range(n)]


def contract_validate(*paths):
    """Runs the contract's own validator, unmodified. Returns (exit code, output)."""
    r = subprocess.run([sys.executable, str(VALIDATOR), *map(str, paths)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def write_docs(folder, pose_doc, ball_doc):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pose.json").write_text(json.dumps(pose_doc))
    (folder / "ball.json").write_text(json.dumps(ball_doc))
    return folder


@pytest.fixture
def validate():
    return contract_validate
