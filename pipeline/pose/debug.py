"""--debug video: skeleton (green = confident, red = low confidence), ball circle, ball gate, frame index."""
import cv2

from .keypoints import CORE_NAMES, SKELETON_EDGES


def _color(conf):
    c = max(0.0, min(1.0, conf))
    return (0, int(255 * c), int(255 * (1 - c)))  # BGR, red -> green


class DebugWriter:
    def __init__(self, path, fps, width, height):
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"could not open debug video for writing: {path}")
        self._scale = max(1, round(min(width, height) / 540))

    def write(self, frame, f, keypoints, ball, gate=None):
        img = frame.copy()
        s = self._scale
        if keypoints:
            pt = {n: (int(round(x)), int(round(y))) for n, (x, y, _) in keypoints.items()}
            for a, b in SKELETON_EDGES:
                cv2.line(img, pt[a], pt[b], _color(min(keypoints[a][2], keypoints[b][2])), 2 * s, cv2.LINE_AA)
            for n in CORE_NAMES:
                cv2.circle(img, pt[n], 3 * s, _color(keypoints[n][2]), -1, cv2.LINE_AA)
        if gate is not None:  # ball tracker's predicted position and search radius (not written to ball.json)
            gx, gy, gr = (int(round(v)) for v in gate)
            cv2.circle(img, (gx, gy), max(gr, 2), (200, 200, 200), s, cv2.LINE_AA)
            cv2.drawMarker(img, (gx, gy), (200, 200, 200), cv2.MARKER_CROSS, 8 * s, s)
        if ball is not None:
            x, y, r, c = ball
            cv2.circle(img, (int(round(x)), int(round(y))), max(int(round(r)), 2), (0, 255, 255), 2 * s, cv2.LINE_AA)
            cv2.putText(img, f"{c:.2f}", (int(x + r) + 4, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255), s)
        label = f"f={f}" + ("" if keypoints else "  no player") + ("" if ball is not None else "  no ball")
        cv2.putText(img, label, (10 * s, 25 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), 4 * s, cv2.LINE_AA)
        cv2.putText(img, label, (10 * s, 25 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), s, cv2.LINE_AA)
        self._w.write(img)

    def close(self):
        self._w.release()
