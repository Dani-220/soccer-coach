"""Reference implementation of body height (H), contract v1.1. Every stage imports this instead of
writing its own, so C, D and E always divide by the same number.

    from contract.tools.body_height import rep_body_height
    H = rep_body_height(pose["frames"], rep["frames"], pose["fps"])

Rule (CONTRACT.md, "Body height"):
  H_rep = median over the frames in [P - 0.3 s, P] of  max(left_ankle.y, right_ankle.y) - nose.y
  where P = the rep's plant frame (contact if there is no plant), counting only frames that have a
  player AND nose, left_ankle and right_ankle with confidence >= 0.3.
  Fewer than 3 such frames -> widen the window to [start, P]. Still fewer -> video-wide value
  (video_body_height). Nothing usable -> None.

Why the last 0.3 s and not the whole run-up: the player's size on screen changes as they run toward
or away from the camera, and while they enter the frame part of the body is cut off. Both bias a
whole-approach median (by ~20% on real behind-the-player footage). Just before the plant the player
is next to the ball, which is where the distances being measured are.
"""
import statistics

MIN_CONF = 0.3
WINDOW_S = 0.3
MIN_FRAMES = 3


def frame_height(keypoints, min_conf=MIN_CONF):
    """Nose-to-lower-ankle height in px for one frame's keypoints, or None if not usable."""
    try:
        nose, la, ra = keypoints["nose"], keypoints["left_ankle"], keypoints["right_ankle"]
    except (KeyError, TypeError):
        return None
    if min(nose[2], la[2], ra[2]) < min_conf:
        return None
    h = max(la[1], ra[1]) - nose[1]
    return h if h > 0 else None


def _median(frames, f0, f1, min_conf):
    hs = []
    for fr in frames[max(f0, 0): f1 + 1]:
        h = frame_height(fr.get("keypoints") if isinstance(fr, dict) and "keypoints" in fr else fr, min_conf)
        if h is not None:
            hs.append(h)
    return (statistics.median(hs), len(hs)) if hs else (None, 0)


def video_body_height(frames, min_conf=MIN_CONF):
    """Fallback: median over every usable frame in the video."""
    return _median(frames, 0, len(frames) - 1, min_conf)[0]


def rep_body_height(frames, rep_frames, fps, min_conf=MIN_CONF):
    """H for one rep, in px.

    frames:     pose.json "frames" (list of {"f", "keypoints"}) or a list of keypoint dicts
    rep_frames: reps.json rep["frames"] ({"start", "plant"?, "contact", "end"})
    fps:        pose.json "fps"
    """
    P = rep_frames.get("plant", rep_frames["contact"])
    w = max(1, round(WINDOW_S * fps))
    h, n = _median(frames, max(P - w, rep_frames["start"]), P, min_conf)
    if n >= MIN_FRAMES:
        return h
    h, n = _median(frames, rep_frames["start"], P, min_conf)
    if n >= MIN_FRAMES:
        return h
    return video_body_height(frames, min_conf)
