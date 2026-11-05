"""Tests for run.py and the app, on the contract's synthetic fixture.  python -m pytest app/tests"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app import core
from app.history import History

REPO = core.REPO
EX = REPO / "contract" / "examples"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """One run.py --reuse-pose run on the fixture, shared by the tests below."""
    out = tmp_path_factory.mktemp("out") / "p1_fixture"
    out.mkdir()
    shutil.copyfile(EX / "pose.example.json", out / "pose.json")
    shutil.copyfile(EX / "ball.example.json", out / "ball.json")
    proc = subprocess.run([sys.executable, "run.py", "--reuse-pose", "--player", str(EX / "player.example.json"),
                           "--out", str(out)], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out


def test_run_writes_every_stage(run_dir):
    for name in ("player", "pose", "ball", "reps", "metrics"):
        assert (run_dir / f"{name}.json").exists()
    assert (run_dir / "overlays" / "r002_vs_r001.mp4").exists()
    assert (run_dir / "overlays" / "r002_vs_r001.png").exists()
    coach_built = (REPO / "pipeline" / "coach" / "__main__.py").exists()
    assert (run_dir / "cards.json").exists() == coach_built


def test_run_stops_at_first_failure(tmp_path):
    shutil.copyfile(EX / "pose.example.json", tmp_path / "pose.json")
    ball = json.loads((EX / "ball.example.json").read_text())
    ball["frames"] = ball["frames"][:-5]  # frame counts no longer match pose.json
    (tmp_path / "ball.json").write_text(json.dumps(ball))
    proc = subprocess.run([sys.executable, "run.py", "--reuse-pose", "--player", str(EX / "player.example.json"),
                           "--out", str(tmp_path)], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "FAILED: stage B (pose)" in proc.stderr
    assert not (tmp_path / "reps.json").exists()


def test_run_rejects_bad_player_without_overwriting(tmp_path):
    (tmp_path / "player.json").write_text((EX / "player.example.json").read_text())
    bad = tmp_path / "bad.json"
    bad.write_text('{"player_id": "x"}')
    proc = subprocess.run([sys.executable, "run.py", "--reuse-pose", "--player", str(bad), "--out", str(tmp_path)],
                          cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 1 and "not a valid player.json" in proc.stderr
    assert json.loads((tmp_path / "player.json").read_text())["player_id"] == "p1"


def test_history_and_dashboard_rows(run_dir, tmp_path):
    h = History(tmp_path / "h.sqlite")
    player = json.loads((EX / "player.example.json").read_text())
    h.save_player(player)
    reps, metrics = core.load_json(run_dir / "reps.json"), core.load_json(run_dir / "metrics.json")
    s1 = h.add_session("p1", "a", run_dir, None, "2026-10-01T10:00:00", reps, metrics)
    h.add_session("p1", "b", run_dir, None, "2026-10-08T10:00:00", reps, metrics)
    h.add_session("p1", "a", run_dir, None, "2026-10-01T10:00:00", reps, metrics)  # replaces, not duplicates
    assert [s["name"] for s in h.sessions("p1")] == ["a", "b"]
    assert s1 not in [s["session_id"] for s in h.sessions("p1")]

    trend = core.foot_trend(h.metric_rows("p1"), "right")
    plant = [t for t in trend if t["metric"] == "plant_foot_distance"]
    assert {t["foot_label"] for t in plant} == {"Left (weak)", "Right (strong)"}
    assert len(plant) == 4  # 2 sessions x 2 feet
    weak = next(t for t in plant if t["foot"] == "left")
    assert weak["reps"] == 1 and weak["in_range"] == 0
    assert not [t for t in trend if t["metric"] == "plant_foot_angle"]  # null values are left out


def test_previous_rep_crosses_sessions():
    sessions = [{"session_id": 1}, {"session_id": 2}]
    reps = {1: [{"rep_id": "r001"}, {"rep_id": "r002"}], 2: [{"rep_id": "r001"}, {"rep_id": "r002"}]}
    assert core.previous_rep(sessions, reps, 1, "r001") is None
    assert core.previous_rep(sessions, reps, 1, "r002") == (sessions[0], "r001")
    assert core.previous_rep(sessions, reps, 2, "r001") == (sessions[0], "r002")


def test_cross_session_overlay(run_dir, tmp_path):
    other = tmp_path / "p1_later"
    shutil.copytree(run_dir, other, ignore=shutil.ignore_patterns("overlays"))
    mp4, png, err = core.make_overlay(other, "r001", run_dir, "r002")
    assert err is None and mp4.exists() and png.exists()
    assert mp4.name == f"r001_vs_{run_dir.name}_r002.mp4"


def test_not_measured_grouping(run_dir):
    targets = core.load_targets()
    metrics = core.load_json(run_dir / "metrics.json")
    view = core.session_view(metrics, targets)
    assert view == "side"
    groups = core.not_measured(metrics["reps"][0]["metrics"], "shot_instep", targets, view)
    assert sorted(groups["front"]) == ["hip_shoulder_separation", "plant_foot_angle"]
    assert groups["side"] == []

    front_video = [{"name": "knee_over_ball", "status": "not_measured", "value": None},
                   {"name": "plant_foot_distance", "status": "not_measured", "value": None}]
    groups = core.not_measured(front_video, "shot_instep", targets, "front")
    assert groups["side"] == ["knee_over_ball"] and groups["other"] == ["plant_foot_distance"]


def test_cards_fall_back_to_example(run_dir, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(run_dir, run)  # a copy, so the shared run folder keeps F's cards.json
    (run / "cards.json").unlink(missing_ok=True)
    cards, source = core.load_cards(run)
    assert source == "example"
    assert [c["metric"] for c in cards["r002"]] == ["plant_foot_distance", "trunk_lean"]


def test_cards_from_pipeline_when_present(run_dir, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(run_dir, run)
    doc = core.load_json(EX / "cards.example.json")
    doc["reps"][1]["cards"][0]["headline"] = "Written by the pipeline"
    (run / "cards.json").write_text(json.dumps(doc))
    cards, source = core.load_cards(run)
    assert source == "pipeline"
    assert cards["r002"][0]["headline"] == "Written by the pipeline"


def test_tags_and_formatting():
    assert core.foot_tag("left", "right") == "weak" and core.foot_label("right", "right") == "Right (strong)"
    assert core.format_value(0.239, "body_heights", 175) == "0.239 body heights (≈ 39 cm)"
    assert core.format_value(-16.0, "degrees") == "-16°"
    assert core.format_target([0.04, 0.16], "body_heights") == "0.04 to 0.16 body heights"


def test_streamlit_app_end_to_end(tmp_path, monkeypatch):
    """Create a profile, run on the fixture, and render all three tabs without exceptions."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setenv("SOCCER_COACH_OUT", str(tmp_path))
    for mod in ("app", "app.core", "app.history"):
        sys.modules.pop(mod, None)  # so the app re-reads SOCCER_COACH_OUT
    at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    at.sidebar.text_input[0].input("t1")
    at.sidebar.button[0].click().run()
    assert not at.exception
    next(b for b in at.button if b.label == "Run on fixture").click().run()
    assert not at.exception, at.exception
    assert any("Done" in s.value for s in at.success)
    at.run()  # re-render with the new session in history
    text = " ".join(i.value for i in at.info)
    assert "Film front-on to measure: plant foot angle, hip shoulder separation" in text
    coach_built = (REPO / "pipeline" / "coach" / "__main__.py").exists()
    assert any("Example cards" in w.value for w in at.warning) != coach_built
    assert (tmp_path / "history.sqlite").exists()
    for mod in ("app", "app.core", "app.history"):
        sys.modules.pop(mod, None)
