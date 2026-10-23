"""Each metric on a synthetic skeleton whose correct answer is known exactly."""
import math

import numpy as np
import pytest

from pipeline.metrics import instep
from pipeline.metrics.context import MIN_CONFIDENCE, RepContext, Video
from pipeline.metrics.tests.synth import ANKLE_Y, CONTACT, END, FPS, H, HIP_Y, PLANT, Clip, rep

BALL_X, BALL_Y = 1000.0, 880.0
HEIGHT_CM = 175


def ctx_for(clip, r=None, height_cm=HEIGHT_CM):
    return RepContext.build(Video.load(clip.pose(), clip.ball_doc()), r or rep(), height_cm=height_cm)


def resting_ball(clip, upto=CONTACT):
    for f in range(upto + 1):
        clip.ball(f, BALL_X, BALL_Y)


def both_directions(clip):
    """The clip as built (moving +x) and its mirror image (moving -x): metrics must agree."""
    return [clip, clip.mirrored()]


# --- plant_foot_distance ---------------------------------------------------------------------

def test_plant_foot_distance_is_gap_over_body_height():
    clip = Clip()
    resting_ball(clip)
    clip.set(PLANT, "left_ankle", x=BALL_X - 60)          # right-foot shot: left is the plant foot
    for c in both_directions(clip):
        value, frame = instep.plant_foot_distance(ctx_for(c))
        assert value == pytest.approx(60 / H)               # 0.1 body heights
        assert frame == PLANT


def test_plant_foot_distance_uses_nearby_ball_when_ball_missing_at_plant():
    clip = Clip()
    resting_ball(clip)
    clip.balls[PLANT] = None
    clip.set(PLANT, "left_ankle", x=BALL_X - 90)
    assert instep.plant_foot_distance(ctx_for(clip))[0] == pytest.approx(90 / H)


def test_plant_foot_distance_uses_support_foot_for_left_foot_shot():
    clip = Clip()
    resting_ball(clip)
    clip.set(PLANT, "right_ankle", x=BALL_X - 48)
    clip.set(PLANT, "left_ankle", x=BALL_X - 500)          # kicking foot, far back: must be ignored
    assert instep.plant_foot_distance(ctx_for(clip, rep(foot="left")))[0] == pytest.approx(48 / H)


def test_plant_foot_distance_none_without_ball():
    assert instep.plant_foot_distance(ctx_for(Clip()))[0] is None


# --- knee_over_ball --------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [30.0, -45.0])
def test_knee_over_ball_signed_by_direction_of_travel(offset):
    clip = Clip()
    resting_ball(clip)
    clip.set(CONTACT, "right_knee", x=BALL_X + offset)     # + = knee ahead of the ball
    for c in both_directions(clip):
        value, frame = instep.knee_over_ball(ctx_for(c))
        assert value == pytest.approx(offset / H)
        assert frame == CONTACT


def test_knee_over_ball_when_ball_hidden_at_contact():
    clip = Clip()
    resting_ball(clip)
    for f in (CONTACT - 1, CONTACT):                        # foot covers the ball, as in real footage
        clip.balls[f] = None
    clip.set(CONTACT, "right_knee", x=BALL_X + 24)
    assert instep.knee_over_ball(ctx_for(clip))[0] == pytest.approx(24 / H)


# --- trunk_lean ------------------------------------------------------------------------------

@pytest.mark.parametrize("lean", [0.0, 10.0, -16.0])
def test_trunk_lean_positive_forward(lean):
    clip = Clip()
    hx = (clip.get(CONTACT, "left_hip")[0] + clip.get(CONTACT, "right_hip")[0]) / 2
    sx = hx + 200 * math.sin(math.radians(lean))
    sy = HIP_Y - 200 * math.cos(math.radians(lean))
    clip.set(CONTACT, "left_shoulder", sx + 4, sy)
    clip.set(CONTACT, "right_shoulder", sx - 4, sy)
    for c in both_directions(clip):
        value, frame = instep.trunk_lean(ctx_for(c))
        assert value == pytest.approx(lean)
        assert frame == CONTACT


# --- ankle_lock ------------------------------------------------------------------------------

def set_ankle_angle(clip, f, deg, foot="right"):
    """Shin straight up from the ankle; toe at `deg` degrees from the shin."""
    ax, ay = clip.get(f, f"{foot}_ankle")
    clip.set(f, f"{foot}_knee", ax, ay - 170)
    r = math.radians(deg)
    clip.set(f, f"{foot}_foot_index", ax + 50 * math.sin(r), ay - 50 * math.cos(r))


@pytest.mark.parametrize("deg", [90.0, 130.0, 155.0, 175.0])
def test_ankle_lock_angle_at_ankle(deg):
    clip = Clip()
    for f in range(CONTACT - 3, CONTACT + 4):
        set_ankle_angle(clip, f, deg)
    value, frame = instep.ankle_lock(ctx_for(clip))
    assert value == pytest.approx(deg)
    assert frame == CONTACT


def test_ankle_lock_averages_only_the_window_around_contact():
    clip = Clip()
    w = round(instep.ANKLE_WINDOW_S * FPS)                  # 2 frames at 120 fps
    for f in range(len(clip.kps)):
        set_ankle_angle(clip, f, 90.0)                      # outside the window: must not count
    for i, f in enumerate(range(CONTACT - w, CONTACT + w + 1)):
        set_ankle_angle(clip, f, 146.0 + 2 * i)             # 146, 148, 150, 152, 154 -> mean 150
    assert instep.ankle_lock(ctx_for(clip))[0] == pytest.approx(150.0)


# --- follow_through --------------------------------------------------------------------------

def test_follow_through_is_forward_ankle_travel():
    clip = Clip()
    cx = clip.get(CONTACT, "right_ankle")[0]
    clip.set(END, "right_ankle", x=cx + 300)
    for c in both_directions(clip):
        value, frame = instep.follow_through(ctx_for(c))
        assert value == pytest.approx(300 / H)              # 0.5 body heights
        assert frame == END


def test_follow_through_negative_when_foot_ends_behind_contact():
    clip = Clip()
    cx = clip.get(CONTACT, "right_ankle")[0]
    clip.set(END, "right_ankle", x=cx - 60)
    assert instep.follow_through(ctx_for(clip))[0] == pytest.approx(-0.1)


# --- ball_speed ------------------------------------------------------------------------------

def px_per_m():
    return H / (HEIGHT_CM / 100 * 0.93)


def launch(clip, vx, vy, g=0.0, missing=()):
    resting_ball(clip)
    for f in range(CONTACT + 1, len(clip.balls)):
        dt = f - CONTACT
        if f not in missing:
            clip.ball(f, BALL_X + vx * dt, BALL_Y + vy * dt + g * dt * dt)


def test_ball_speed_with_gravity_and_occlusion():
    clip = Clip()
    launch(clip, vx=50, vy=-15, g=0.2, missing={CONTACT + 1, CONTACT + 2})
    value, frame = instep.ball_speed(ctx_for(clip))
    assert value == pytest.approx(math.hypot(50, 15) * FPS / px_per_m())
    assert frame == CONTACT + 3


def test_ball_speed_linear_fit_with_few_detections():
    clip = Clip()
    launch(clip, vx=-30, vy=0, missing=set(range(CONTACT + 4, len(clip.balls))))   # 3 detections
    assert instep.ball_speed(ctx_for(clip))[0] == pytest.approx(30 * FPS / px_per_m())


def test_ball_speed_none_without_player_height_or_ball():
    clip = Clip()
    launch(clip, vx=50, vy=0)
    assert instep.ball_speed(ctx_for(clip, height_cm=None))[0] is None
    assert instep.ball_speed(ctx_for(Clip()))[0] is None


# --- front-view metrics ----------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0.0, 20.0, -12.0])
def test_plant_foot_angle_from_foreshortened_foot(deg):
    clip = Clip(view="front", speed=0)
    L = instep.FOOT_TO_TORSO * 200                          # torso = hip_mid to shoulder_mid = 200 px
    heel_x = clip.get(PLANT, "left_heel")[0]
    # Left hip is on the image right of the right hip, so "outward" for the left foot is +x.
    clip.set(PLANT, "left_foot_index", x=heel_x + L * math.sin(math.radians(deg)))
    for c in both_directions(clip):
        value, frame = instep.plant_foot_angle(ctx_for(c))
        assert value == pytest.approx(deg)
        assert frame == PLANT


def test_backswing_frame_is_highest_kicking_ankle():
    clip = Clip()
    clip.set(15, "right_ankle", y=HIP_Y + 150)
    clip.set(17, "right_ankle", y=HIP_Y + 200)
    clip.set(15, "left_ankle", y=HIP_Y + 50)                # support foot: ignored
    assert instep.backswing_frame(ctx_for(clip)) == 15


def test_hip_shoulder_separation_from_line_widths():
    clip = Clip(view="front", speed=0)
    B = 15
    clip.set(B, "right_ankle", y=HIP_Y + 150)               # top of the backswing
    hx = clip.get(B, "left_hip")[0] - 50
    th_hip, th_sh = math.radians(40), math.radians(15)
    clip.set(B, "left_hip", x=hx + 50 * math.cos(th_hip))
    clip.set(B, "right_hip", x=hx - 50 * math.cos(th_hip))
    clip.set(B, "left_shoulder", x=hx + 80 * math.cos(th_sh))
    clip.set(B, "right_shoulder", x=hx - 80 * math.cos(th_sh))
    value, frame = instep.hip_shoulder_separation(ctx_for(clip))
    assert value == pytest.approx(25.0)
    assert frame == B


# --- missing data ----------------------------------------------------------------------------

def test_metrics_return_none_when_player_missing_at_key_frame():
    clip = Clip()
    resting_ball(clip)
    clip.hide_player(CONTACT)
    ctx = ctx_for(clip)
    for fn in (instep.knee_over_ball, instep.trunk_lean, instep.follow_through):
        assert fn(ctx)[0] is None
    clip.hide_player(PLANT)
    assert instep.plant_foot_distance(ctx_for(clip))[0] is None
    assert instep.plant_foot_angle(ctx_for(clip))[0] is None


# --- keypoint confidence ---------------------------------------------------------------------

LOW = MIN_CONFIDENCE - 0.05


@pytest.mark.parametrize("fn, view, frame, point", [
    (instep.plant_foot_distance, "side", PLANT, "left_ankle"),
    (instep.knee_over_ball, "side", CONTACT, "right_knee"),
    (instep.trunk_lean, "side", CONTACT, "left_shoulder"),
    (instep.trunk_lean, "side", CONTACT, "right_hip"),
    (instep.follow_through, "side", CONTACT, "right_ankle"),
    (instep.follow_through, "side", END, "right_ankle"),
    (instep.plant_foot_angle, "front", PLANT, "left_heel"),
    (instep.plant_foot_angle, "front", PLANT, "left_foot_index"),
    (instep.plant_foot_angle, "front", PLANT, "right_hip"),
    (instep.plant_foot_angle, "front", PLANT, "left_shoulder"),
])
def test_low_confidence_point_counts_as_missing(fn, view, frame, point):
    clip = Clip(view=view, speed=0 if view == "front" else 4.0)
    resting_ball(clip)
    assert fn(ctx_for(clip))[0] is not None                 # measurable when the point is confident
    clip.set_conf(frame, point, LOW)
    assert fn(ctx_for(clip))[0] is None


def test_confidence_at_threshold_is_kept():
    clip = Clip()
    resting_ball(clip)
    clip.set(PLANT, "left_ankle", x=BALL_X - 60)
    clip.set_conf(PLANT, "left_ankle", MIN_CONFIDENCE)
    assert instep.plant_foot_distance(ctx_for(clip))[0] == pytest.approx(60 / H)


def test_ankle_lock_skips_low_confidence_frames():
    clip = Clip()
    for f in range(CONTACT - 3, CONTACT + 4):
        set_ankle_angle(clip, f, 150.0)
    set_ankle_angle(clip, CONTACT, 60.0)                     # a bad detection...
    clip.set_conf(CONTACT, "right_foot_index", LOW)         # ...that the detector flagged
    assert instep.ankle_lock(ctx_for(clip))[0] == pytest.approx(150.0)
    for f in range(CONTACT - 3, CONTACT + 4):
        clip.set_conf(f, "right_foot_index", LOW)
    assert instep.ankle_lock(ctx_for(clip))[0] is None


def test_backswing_frame_skips_low_confidence_ankle():
    clip = Clip()
    clip.set(15, "right_ankle", y=HIP_Y + 100)
    clip.set_conf(15, "right_ankle", LOW)
    clip.set(17, "right_ankle", y=HIP_Y + 200)
    assert instep.backswing_frame(ctx_for(clip)) == 17


def test_hip_shoulder_separation_ignores_low_confidence_hips():
    clip = Clip(view="front", speed=0)
    clip.set(15, "right_ankle", y=HIP_Y + 150)
    clip.set(10, "left_hip", x=clip.get(10, "left_hip")[0] + 400)   # absurdly wide, but flagged
    clip.set_conf(10, "left_hip", LOW)
    # Without the confidence check, frame 10 would set the "square-on" hip width and give ~ +80 deg.
    assert instep.hip_shoulder_separation(ctx_for(clip)) == (pytest.approx(0.0), 15)
    for f in range(len(clip.kps)):
        clip.set_conf(f, "left_hip", LOW)
    assert instep.hip_shoulder_separation(ctx_for(clip))[0] is None


# --- degenerate geometry ---------------------------------------------------------------------

def test_angle_at_zero_length_side_is_none():
    a, b = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    assert instep._angle_at(a, a, b) is None                # A coincides with B
    assert instep._angle_at(b, a, a) is None                # C coincides with B
    assert instep._angle_at(a + 0.5, a, b) is None          # under MIN_SEGMENT_PX
    assert instep._angle_at(b, a, np.array([0.0, 10.0])) == pytest.approx(90.0)


def test_ankle_lock_coincident_toe_and_ankle_is_none_not_nan():
    clip = Clip()
    for f in range(CONTACT - 3, CONTACT + 4):
        clip.set(f, "right_foot_index", *clip.get(f, "right_ankle"))
    assert instep.ankle_lock(ctx_for(clip))[0] is None
    set_ankle_angle(clip, CONTACT, 140.0)                   # one usable frame is enough
    assert instep.ankle_lock(ctx_for(clip))[0] == pytest.approx(140.0)


def test_plant_foot_angle_none_when_torso_collapses():
    clip = Clip(view="front", speed=0)
    for side in ("left", "right"):
        clip.set(PLANT, f"{side}_shoulder", y=HIP_Y)        # shoulders on the hip line: no length scale
    assert instep.plant_foot_angle(ctx_for(clip))[0] is None


def test_every_metric_has_a_formula_docstring():
    for name, fn in instep.METRICS.items():
        assert fn.__doc__ and "=" in fn.__doc__, name
