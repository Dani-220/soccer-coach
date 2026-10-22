"""Body height, travel direction and camera view."""
import pytest

from pipeline.metrics.context import Video, body_height_px, camera_view, travel_direction, _parse_frames
from pipeline.metrics.tests.synth import ANKLE_Y, H, NOSE_Y, Clip


def frames_of(clip):
    return _parse_frames(clip.pose())


def test_body_height_nose_to_lower_ankle():
    clip = Clip()
    clip.set(0, "left_ankle", y=ANKLE_Y - 40)              # raised foot: the lower ankle counts
    assert body_height_px(frames_of(clip)) == pytest.approx(H)


def test_body_height_median_of_first_30_player_frames():
    clip = Clip(n=60)
    for f in range(5):
        clip.hide_player(f)                                 # skipped: no player
    for f in (5, 6, 7):
        clip.set(f, "nose", y=NOSE_Y - 300)                 # outliers: median ignores them
    for f in range(35, 60):
        clip.set(f, "nose", y=NOSE_Y - 300)                 # beyond the first 30 player frames
    assert body_height_px(frames_of(clip)) == pytest.approx(H)


def test_body_height_none_without_player():
    clip = Clip(n=3)
    for f in range(3):
        clip.hide_player(f)
    assert body_height_px(frames_of(clip)) is None
    with pytest.raises(ValueError):
        Video.load(clip.pose(), clip.ball_doc())


@pytest.mark.parametrize("direction", [1, -1])
def test_travel_direction_from_hip_motion(direction):
    clip = Clip(direction=direction)
    clip.hide_player(10)
    assert travel_direction(frames_of(clip), 0, 25) == direction


def test_camera_view():
    assert camera_view(frames_of(Clip(view="side"))) == "side"
    assert camera_view(frames_of(Clip(view="front"))) == "front"
