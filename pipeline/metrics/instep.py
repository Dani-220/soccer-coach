"""Instep-shot metrics. One function per metric; each returns (value, frame).

`value` is None when the inputs needed for it are missing: no player or no ball where it is needed,
or a needed keypoint below MIN_CONFIDENCE (context.py), which counts as missing.
Whether a metric is measurable from the camera view at all is decided by `views` in targets.yaml,
not here. Notation used in the docstrings:

    H      body height in px (nose to lower ankle, CONTRACT.md)
    dir    +1 if the player travels toward +x, -1 toward -x (from hip motion)
    K, S   kicking and support (plant) side, e.g. K_knee = right_knee for a right-foot shot
    P, C, E  plant, contact and end frames from reps.json
    p.x@F  x of point p at frame F;  ball@F = ball centre at frame F
"""
import math

import numpy as np

from .context import hip_mid, shoulder_mid, xy

# Anthropometric ratio (Drillis & Contini): foot length ~0.152 of stature, hip-to-shoulder ~0.288,
# so foot length ~0.53 x torso length. Used to undo foreshortening of the foot in front view.
FOOT_TO_TORSO = 0.53
BALL_SPEED_FIT_S = 0.10     # seconds of ball flight used to fit launch velocity
BALL_SPEED_FRAME_OFFSET = 3  # report ball_speed at C + 3, where the ball is clear of the foot
ANKLE_WINDOW_S = 0.02       # half-width of the averaging window for ankle_lock
MIN_SEGMENT_PX = 1.0        # points closer than this have no usable direction between them


def _angle_at(a, b, c):
    """Interior angle ABC in degrees, or None when A or C is within MIN_SEGMENT_PX of B.

    A zero-length side has no direction, so the angle is undefined (and dividing by its
    length would give NaN).
    """
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < MIN_SEGMENT_PX or n2 < MIN_SEGMENT_PX:
        return None
    cos = np.dot(v1, v2) / (n1 * n2)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def plant_foot_distance(ctx):
    """Distance between the plant foot and the ball when the plant foot lands, in body heights.

        plant_foot_distance = |S_ankle.x@P - ball.x@P| / H

    The horizontal image distance between the support ankle and the ball centre. Side-on, this is
    the fore-aft gap (how far behind the ball the foot is planted). From the front, it is the
    sideways gap. The ball position is the median of nearby detections (see ball_at_rest).
    Frame: P.
    """
    f = ctx.plant
    ankle, ball = ctx.point(f, f"{ctx.support}_ankle"), ctx.ball_at_rest(f)
    if ankle is None or ball is None:
        return None, f
    return abs(ankle[0] - ball[0]) / ctx.H, f


def plant_foot_angle(ctx):
    """Plant-foot direction vs. the target line, in degrees (front view only).

        L     = FOOT_TO_TORSO * |shoulder_mid@P - hip_mid@P|      (expected foot length, px)
        out   = sign(S_hip.x@P - K_hip.x@P)                       (image direction pointing away from the kicking leg)
        angle = asin(clip(out * (S_foot_index.x@P - S_heel.x@P) / L, -1, 1))

    With the camera on the target line (facing the player or behind them), a foot pointing at the
    target looks like a point: heel and toe share the same x. Turning the foot by `angle` moves the
    toe sideways by L * sin(angle). Positive = toes turned outward (away from the kicking leg),
    negative = turned in across the body. L comes from that frame's torso length, so it scales as
    the player comes closer to the camera. Frame: P.
    """
    f = ctx.plant
    kp = ctx.keypoints(f)
    s, k = ctx.support, ctx.foot
    sh = shoulder_mid(kp)
    s_hip, k_hip, heel, toe = (xy(kp, n) for n in (f"{s}_hip", f"{k}_hip", f"{s}_heel", f"{s}_foot_index"))
    if sh is None or s_hip is None or k_hip is None or heel is None or toe is None:
        return None, f
    L = FOOT_TO_TORSO * np.linalg.norm(sh - (s_hip + k_hip) / 2)
    if L < MIN_SEGMENT_PX:
        return None, f
    out = 1.0 if s_hip[0] >= k_hip[0] else -1.0
    dx = out * (toe[0] - heel[0])
    return math.degrees(math.asin(float(np.clip(dx / L, -1.0, 1.0)))), f


def knee_over_ball(ctx):
    """How far the kicking knee is ahead of the ball at contact, in body heights.

        knee_over_ball = dir * (K_knee.x@C - ball.x@C) / H

    Positive = knee in front of the ball centre (over the ball, which keeps the shot low),
    negative = knee behind the ball. If the ball is hidden by the foot at contact, its position
    comes from the detections just before contact (see ball_at_rest). Frame: C.
    """
    f = ctx.contact
    knee, ball = ctx.point(f, f"{ctx.foot}_knee"), ctx.ball_at_rest(f)
    if knee is None or ball is None:
        return None, f
    return ctx.direction * (knee[0] - ball[0]) / ctx.H, f


def trunk_lean(ctx):
    """Trunk angle from vertical at contact, in degrees, positive = leaning forward.

        trunk_lean = atan2(dir * (shoulder_mid.x@C - hip_mid.x@C), hip_mid.y@C - shoulder_mid.y@C)

    shoulder_mid and hip_mid are the midpoints of the left/right shoulders and hips. 0 = upright,
    positive = shoulders ahead of the hips in the direction of travel, negative = leaning back.
    Frame: C.
    """
    f = ctx.contact
    kp = ctx.keypoints(f)
    sh, hp = shoulder_mid(kp), hip_mid(kp)
    if sh is None or hp is None:
        return None, f
    return math.degrees(math.atan2(ctx.direction * (sh[0] - hp[0]), hp[1] - sh[1])), f


def backswing_frame(ctx):
    """Top of the backswing: the frame in [start, C] where the kicking ankle is highest relative to the hips.

        backswing = argmin over f in [start, C] of (K_ankle.y@f - hip_mid.y@f)

    This works side-on and from the front, because the heel lifts toward the hips in both.
    Frames where the kicking ankle or either hip is below MIN_CONFIDENCE are skipped. Falls back
    to P when no frame in the window qualifies.
    """
    best, best_f = None, ctx.plant
    for f in range(ctx.start, ctx.contact + 1):
        ankle, hp = ctx.point(f, f"{ctx.foot}_ankle"), hip_mid(ctx.keypoints(f))
        if ankle is not None and hp is not None:
            d = ankle[1] - hp[1]
            if best is None or d < best:
                best, best_f = d, f
    return best_f


def hip_shoulder_separation(ctx):
    """Hip-shoulder separation at the top of the backswing, in degrees (front view only).

        r_seg(f)   = |left_seg.x@f - right_seg.x@f| / |shoulder_mid@f - hip_mid@f|   (seg = hip, shoulder)
        theta_seg  = acos(clip(r_seg(B) / max over f in [start, C] of r_seg(f), 0, 1))
        separation = theta_hip - theta_shoulder,   B = backswing_frame

    Seen from the front, a hip or shoulder line turned by theta about the vertical axis looks
    cos(theta) as wide as when square to the camera. The widest the line looks during the rep is
    taken as square-on, and widths are divided by torso length so the player getting closer to
    the camera cancels out. Positive = hips turned further than shoulders.
    Limitation: 2D keypoints cannot show which way a line turned, so this assumes hips and
    shoulders turn the same way. If the shoulders counter-rotate, the true separation is
    larger than this value. Frame: B.
    """
    B = backswing_frame(ctx)

    def ratios(f):
        kp = ctx.keypoints(f)
        pts = [xy(kp, n) for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")]
        if any(p is None for p in pts):
            return None
        lh, rh, ls, rsh = pts
        torso = np.linalg.norm((ls + rsh) / 2 - (lh + rh) / 2)
        if torso < MIN_SEGMENT_PX:
            return None
        return abs(lh[0] - rh[0]) / torso, abs(ls[0] - rsh[0]) / torso

    rs = {f: r for f in range(ctx.start, ctx.contact + 1) if (r := ratios(f)) is not None}
    if B not in rs:
        return None, B
    max_hip = max(r[0] for r in rs.values())
    max_sh = max(r[1] for r in rs.values())
    if max_hip <= 0 or max_sh <= 0:
        return None, B
    th_hip = math.acos(float(np.clip(rs[B][0] / max_hip, 0.0, 1.0)))
    th_sh = math.acos(float(np.clip(rs[B][1] / max_sh, 0.0, 1.0)))
    return math.degrees(th_hip - th_sh), B


def ankle_lock(ctx):
    """Kicking-ankle angle at contact, in degrees (180 = foot pointed straight along the shin).

        ankle_lock = mean over f in [C - w, C + w] of angle(K_knee@f, K_ankle@f, K_foot_index@f)
        w = max(1, round(0.02 s * fps))   (2 frames at 120 fps)

    The interior angle at the ankle between the shin (ankle -> knee) and the foot (ankle -> toe).
    A locked, pointed instep is near 150-180; a loose or flexed ankle is smaller. Foot keypoints
    are the noisiest joints and sit only a few pixels apart, so one frame can be several degrees
    off. The ankle is held still through the strike, so averaging over a short window centred
    on contact cuts the noise without blurring the motion. Frames where any of the three points is
    below MIN_CONFIDENCE, or the angle is undefined, are left out of the mean. Frame: C.
    """
    C, k = ctx.contact, ctx.foot
    w = max(1, round(ANKLE_WINDOW_S * ctx.fps))
    angles = []
    for f in range(max(ctx.start, C - w), min(ctx.end, C + w) + 1):
        knee, ankle, toe = (ctx.point(f, f"{k}_{j}") for j in ("knee", "ankle", "foot_index"))
        if knee is not None and ankle is not None and toe is not None:
            a = _angle_at(knee, ankle, toe)
            if a is not None:
                angles.append(a)
    return (float(np.mean(angles)) if angles else None), C


def follow_through(ctx):
    """How far the kicking foot travels forward after contact, in body heights.

        follow_through = dir * (K_ankle.x@E - K_ankle.x@C) / H

    Horizontal travel of the kicking ankle from contact to the end of the rep, positive in the
    direction of travel. Frame: E.
    """
    c_ank, e_ank = ctx.point(ctx.contact, f"{ctx.foot}_ankle"), ctx.point(ctx.end, f"{ctx.foot}_ankle")
    if c_ank is None or e_ank is None:
        return None, ctx.end
    return ctx.direction * (e_ank[0] - c_ank[0]) / ctx.H, ctx.end


def ball_speed(ctx):
    """Ball launch speed in m/s.

        fit x(t), y(t) with t = f - C over the ball detections in (C, C + 0.10 s]:
            degree 2 if >= 4 detections (captures gravity), degree 1 if 2-3 detections
        v_px  = |(dx/dt, dy/dt) at t = 0|                     (px per frame, at the moment of contact)
        ball_speed = v_px * fps / px_per_m,   px_per_m = H / (height_cm / 100 * 0.93)

    The scale comes from the player's height (player.json), so this assumes the ball travels
    roughly in the plane the player is in. A ball going toward or away from the camera reads slow.
    Frame: C + 3, clipped to the video (the ball is clear of the foot there).
    """
    frame = min(ctx.contact + BALL_SPEED_FRAME_OFFSET, ctx.n_frames - 1)
    if ctx.px_per_m is None:
        return None, frame
    hi = min(ctx.contact + max(2, round(BALL_SPEED_FIT_S * ctx.fps)), ctx.n_frames - 1)
    pts = [(f - ctx.contact, *ctx.balls[f][:2]) for f in range(ctx.contact + 1, hi + 1) if ctx.balls[f] is not None]
    if len(pts) < 2:
        return None, frame
    t, x, y = np.array(pts).T
    deg = 2 if len(pts) >= 4 else 1
    vx = np.polyfit(t, x, deg)[-2]   # derivative at t = 0 is the linear coefficient
    vy = np.polyfit(t, y, deg)[-2]
    return float(math.hypot(vx, vy)) * ctx.fps / ctx.px_per_m, frame


# Output order matches metrics.example.json.
METRICS = {
    "plant_foot_distance": plant_foot_distance,
    "plant_foot_angle": plant_foot_angle,
    "knee_over_ball": knee_over_ball,
    "trunk_lean": trunk_lean,
    "hip_shoulder_separation": hip_shoulder_separation,
    "ankle_lock": ankle_lock,
    "follow_through": follow_through,
    "ball_speed": ball_speed,
}
