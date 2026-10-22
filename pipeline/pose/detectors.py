"""Model wrappers. Each returns raw candidates in pixels; choosing one is done in tracking.py.

A pose candidate is {name: (x_px, y_px, conf)} with all 33 MediaPipe names.
A ball candidate is (cx_px, cy_px, radius_px, conf).
Heavy imports (mediapipe, ultralytics) happen inside the classes so tests can run without them.
"""
import math
import os
import urllib.request
from pathlib import Path

from .keypoints import LANDMARK_NAMES

MODEL_DIR = Path(os.environ.get("SOCCER_COACH_MODELS", Path.home() / ".cache" / "soccer-coach" / "models"))

POSE_MODEL_URLS = {
    v: f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
    for v in ("lite", "full", "heavy")
}


def resolve_pose_model(spec):
    """'lite' | 'full' | 'heavy' (downloaded once into MODEL_DIR) or a path to a .task file."""
    if spec not in POSE_MODEL_URLS:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"pose model not found: {spec}")
        return path
    path = MODEL_DIR / f"pose_landmarker_{spec}.task"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {POSE_MODEL_URLS[spec]} -> {path}")
        urllib.request.urlretrieve(POSE_MODEL_URLS[spec], tmp)
        tmp.replace(path)
    return path


def resolve_ball_model(spec):
    """A path to a YOLO .pt file, or a stock name like 'yolo11m.pt' (Ultralytics downloads it into MODEL_DIR)."""
    path = Path(spec)
    if path.is_file() or path.parent != Path("."):
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / spec


def _prob(v):
    """MediaPipe visibility is a sigmoid for the stock models, but may be a logit for others."""
    if v is None:
        return 0.0
    if 0.0 <= v <= 1.0:
        return float(v)
    return 1.0 / (1.0 + math.exp(-v))


class MediaPipePoseDetector:
    def __init__(self, model="heavy", max_poses=2, min_detection=0.5, min_presence=0.5, min_tracking=0.5):
        import mediapipe as mp

        self._mp = mp
        vision = mp.tasks.vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(resolve_pose_model(model))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_poses,
            min_pose_detection_confidence=min_detection,
            min_pose_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, timestamp_ms):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        return [
            {name: (lm.x * w, lm.y * h, _prob(lm.visibility)) for name, lm in zip(LANDMARK_NAMES, pose)}
            for pose in result.pose_landmarks
        ]

    def close(self):
        self._landmarker.close()


class YoloBallDetector:
    def __init__(self, model="yolo11m.pt", conf=0.25, imgsz=1280, device=None):
        from ultralytics import YOLO

        self._model = YOLO(str(resolve_ball_model(model)))
        names = self._model.names
        self._cls = next((i for i, n in names.items() if n == "sports ball"), None)
        if self._cls is None:
            raise ValueError(f"model {model} has no 'sports ball' class (classes: {list(names.values())[:10]}...)")
        self._kw = dict(conf=conf, imgsz=imgsz, classes=[self._cls], verbose=False)
        if device:
            self._kw["device"] = device

    def detect(self, frame_bgr):
        boxes = self._model.predict(frame_bgr, **self._kw)[0].boxes
        out = []
        for (x1, y1, x2, y2), c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, ((x2 - x1) + (y2 - y1)) / 4, float(c)))
        return out

    def close(self):
        pass
