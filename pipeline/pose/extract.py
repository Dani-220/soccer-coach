"""Video -> pose.json + ball.json documents (contract v1.0). Detector-agnostic so tests can inject fakes."""
from pathlib import Path

import cv2

from .debug import DebugWriter
from .one_euro import KeypointSmoother
from .tracking import PlayerTracker, pick_ball

SCHEMA_VERSION = "1.0"
PLAYER_TRACK_ID = 1


def _clip01(v):
    return min(1.0, max(0.0, float(v)))


def _kp_out(keypoints):
    return {n: [round(x, 1), round(y, 1), round(_clip01(c), 3)] for n, (x, y, c) in keypoints.items()}


def _ball_out(ball):
    if ball is None:
        return None
    x, y, r, c = ball
    return [round(x, 1), round(y, 1), round(max(0.0, r), 1), round(_clip01(c), 3)]


def _fps_value(fps):
    fps = round(float(fps), 3)
    return int(fps) if fps.is_integer() else fps


def extract(video_path, pose_detector, ball_detector, *, smoother=None, tracker=None,
            debug_path=None, fps_override=None, progress=None):
    """Runs both detectors on every decoded frame and returns (pose_doc, ball_doc)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"video reports no fps ({fps}); pass --fps")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    smoother = smoother or KeypointSmoother()
    tracker = tracker or PlayerTracker()

    pose_frames, ball_frames = [], []
    width = height = None
    debug = None
    last_ms = -1
    f = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if width is None:
                height, width = frame.shape[:2]
                if debug_path:
                    debug = DebugWriter(debug_path, fps, width, height)
            # MediaPipe VIDEO mode needs strictly increasing integer milliseconds.
            ms = max(last_ms + 1, round(f * 1000 / fps))
            last_ms = ms

            player, new_track = tracker.update(pose_detector.detect(frame, ms))
            if player is None:
                smoother.reset()
                keypoints = {}
            else:
                if new_track:
                    smoother.reset()
                keypoints = smoother(player, f / fps)
            ball = pick_ball(ball_detector.detect(frame))

            pose_frames.append({"f": f, "keypoints": _kp_out(keypoints)})
            ball_frames.append({"f": f, "ball": _ball_out(ball)})
            if debug:
                debug.write(frame, f, keypoints, ball)
            f += 1
            if progress:
                progress(f, total)
    finally:
        cap.release()
        if debug:
            debug.close()

    if f == 0:
        raise ValueError(f"no frames could be decoded from {video_path}")
    header = {"video": Path(video_path).name, "fps": _fps_value(fps), "width": width, "height": height,
              "schema_version": SCHEMA_VERSION}
    pose_doc = {**header, "player_track_id": PLAYER_TRACK_ID, "frames": pose_frames}
    ball_doc = {**header, "frames": ball_frames}
    return pose_doc, ball_doc
