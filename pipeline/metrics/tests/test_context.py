"""Per-rep body height wiring, travel direction and camera view.

The body-height rule itself is tested by the contract (contract/tools/test_body_height.py); these
tests check that metrics uses it per rep with the right inputs.
"""
import pytest

from pipeline.metrics import context
from pipeline.metrics.context import RepContext, Video, camera_view, travel_direction, _parse_frames
from pipeline.metrics.tests.synth import FPS, H, NOSE_Y, Clip, rep

WINDOW = round(0.3 * FPS)          # contract v1.1: H comes from [P - 0.3 s, P]


def frames_of(clip):
    return _parse_frames(clip.pose())


def rep_H(clip, r):
    return RepContext.build(Video.load(clip.pose(), clip.ball_doc()), r).H


def test_each_rep_gets_its_own_body_height():
    clip = Clip(n=100, speed=0)
    for f in range(50, 100):
        clip.set(f, "nose", y=NOSE_Y - 100)                 # player looks bigger in the second rep
    r1 = rep(start=0, plant=20, contact=25, end=40)
    r2 = rep(start=50, plant=70, contact=75, end=90, rep_id="r002")
    assert rep_H(clip, r1) == pytest.approx(H)
    assert rep_H(clip, r2) == pytest.approx(H + 100)


def test_body_height_from_the_window_before_plant_not_the_run_up():
    plant = 80
    clip = Clip(n=100, speed=0)
    for f in range(plant - WINDOW):
        clip.set(f, "nose", y=NOSE_Y - 200)                 # far from the camera early in the run-up
    assert rep_H(clip, rep(start=0, plant=plant, contact=85, end=95)) == pytest.approx(H)


def test_body_height_matches_contract_tool_on_fixture():
    from pipeline.metrics.tests.test_compute import load
    from contract.tools.body_height import rep_body_height
    pose, ball, reps = load("pose.example.json"), load("ball.example.json"), load("reps.example.json")
    video = Video.load(pose, ball)
    for r in reps["reps"]:
        expected = rep_body_height(pose["frames"], r["frames"], pose["fps"])
        assert RepContext.build(video, r).H == pytest.approx(expected)


def test_body_height_none_when_no_usable_frame():
    clip = Clip(n=30)
    for f in range(30):
        clip.set_conf(f, "nose", 0.1)
    ctx = RepContext.build(Video.load(clip.pose(), clip.ball_doc()), rep(start=0, plant=20, contact=25, end=29),
                           height_cm=175)
    assert ctx.H is None and ctx.px_per_m is None


def test_rep_body_height_called_once_per_rep(monkeypatch):
    calls = []
    real = context.rep_body_height
    monkeypatch.setattr(context, "rep_body_height", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    from pipeline.metrics.compute import compute_metrics
    from pipeline.metrics.tests.synth import reps_doc
    clip = Clip(n=100)
    r1, r2 = rep(start=0, plant=20, contact=25, end=40), rep(start=50, plant=70, contact=75, end=90, rep_id="r002")
    compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(r1, r2), {"height_cm": 175})
    assert calls == [r1["frames"], r2["frames"]]


def test_travel_direction_ignores_low_confidence_hips():
    clip = Clip(direction=1)
    for f in range(13, 26):
        for side in ("left", "right"):
            clip.set(f, f"{side}_hip", x=0.0)                # would drag the slope negative
            clip.set_conf(f, f"{side}_hip", 0.1)
    assert travel_direction(frames_of(clip), 0, 25) == 1


@pytest.mark.parametrize("direction", [1, -1])
def test_travel_direction_from_hip_motion(direction):
    clip = Clip(direction=direction)
    clip.hide_player(10)
    assert travel_direction(frames_of(clip), 0, 25) == direction


def test_camera_view():
    assert camera_view(frames_of(Clip(view="side"))) == "side"
    assert camera_view(frames_of(Clip(view="front"))) == "front"
