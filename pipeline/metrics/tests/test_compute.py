"""metrics.json assembly, view gating, targets.yaml, the contract fixtures and the CLI."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from pipeline.metrics.__main__ import main
from pipeline.metrics.compute import DEFAULT_TARGETS, compute_metrics, load_targets, metric_entry
from pipeline.metrics.instep import METRICS
from pipeline.metrics.tests.synth import CONTACT, PLANT, Clip, rep, reps_doc

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "contract" / "examples"
PLAYER = {"player_id": "p1", "strong_foot": "right", "level": "casual", "height_cm": 175}


def load(name):
    return json.loads((EXAMPLES / name).read_text())


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("contract_validate", ROOT / "contract" / "tools" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_contract(validator, metrics, pose=None, reps=None):
    errs = validator.schema_errors("metrics", metrics)
    docs = {"metrics": metrics, **({"pose": pose} if pose else {}), **({"reps": reps} if reps else {})}
    errs += validator.cross_checks(docs)
    assert errs == []


def by_name(rep_out):
    return {m["name"]: m for m in rep_out["metrics"]}


def synthetic_rep(view):
    clip = Clip(view=view, speed=0 if view == "front" else 4.0)
    for f in range(len(clip.balls)):
        clip.ball(f, 1000 + max(0, f - CONTACT) * 40, 880)
    return clip


# --- assembly --------------------------------------------------------------------------------

def test_targets_yaml_matches_contract_example_and_covers_every_metric():
    ours, contract = load_targets(), yaml.safe_load((EXAMPLES / "targets.example.yaml").read_text())
    assert ours == contract
    assert set(ours["shot_instep"]) == set(METRICS)


def test_status_judged_on_rounded_value():
    cfg = {"range": [0.04, 0.16], "unit": "body_heights", "views": ["side"]}
    assert metric_entry("plant_foot_distance", 0.16049, 9, cfg, "side")["status"] == "in_range"
    assert metric_entry("plant_foot_distance", 0.1606, 9, cfg, "side")["status"] == "out_of_range"
    e = metric_entry("trunk_lean", 3.14159, 9, {"range": [-5, 15], "unit": "degrees", "views": ["side"]}, "side")
    assert (e["value"], e["status"], e["target"], e["frame"]) == (3.1, "in_range", [-5, 15], 9)


def test_view_not_listed_or_missing_value_is_not_measured():
    cfg = {"range": [-15, 15], "unit": "degrees", "views": ["front"]}
    assert metric_entry("plant_foot_angle", 3.0, 1, cfg, "side")["status"] == "not_measured"
    assert metric_entry("plant_foot_angle", 3.0, 1, cfg, "side")["value"] is None
    assert metric_entry("plant_foot_angle", None, 1, cfg, "front")["status"] == "not_measured"


@pytest.mark.parametrize("view, measured", [
    ("side", {"plant_foot_distance", "knee_over_ball", "trunk_lean", "ankle_lock", "follow_through", "ball_speed"}),
    ("front", {"plant_foot_distance", "plant_foot_angle", "hip_shoulder_separation"}),
])
def test_views_from_targets_decide_what_is_measured(validator, view, measured):
    clip = synthetic_rep(view)
    reps = reps_doc(rep())
    out = compute_metrics(clip.pose(), clip.ball_doc(), reps, PLAYER)
    check_contract(validator, out, clip.pose(), reps)
    got = {m["name"] for m in out["reps"][0]["metrics"] if m["status"] != "not_measured"}
    assert got == measured
    assert [m["name"] for m in out["reps"][0]["metrics"]] == list(METRICS)


def test_edited_targets_change_target_and_status():
    clip = synthetic_rep("side")
    clip.set(PLANT, "left_ankle", x=1000 - 60)                 # 0.1 body heights
    targets = copy.deepcopy(load_targets())
    out = by_name(compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(rep()), PLAYER, targets)["reps"][0])
    assert out["plant_foot_distance"]["status"] == "in_range"
    targets["shot_instep"]["plant_foot_distance"]["range"] = [0.0, 0.05]
    out = by_name(compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(rep()), PLAYER, targets)["reps"][0])
    assert out["plant_foot_distance"]["target"] == [0.0, 0.05]
    assert out["plant_foot_distance"]["status"] == "out_of_range"


def test_missing_player_height_and_unknown_moves(validator):
    clip = synthetic_rep("side")
    other = {**rep(rep_id="r002"), "move": "dribble"}
    out = compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(rep(), other), player=None)
    check_contract(validator, out)
    assert [r["rep_id"] for r in out["reps"]] == ["r001"]
    assert by_name(out["reps"][0])["ball_speed"]["status"] == "not_measured"


def test_player_missing_at_contact_still_valid(validator):
    clip = synthetic_rep("side")
    clip.hide_player(CONTACT)
    out = compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(rep()), PLAYER)
    check_contract(validator, out)
    assert by_name(out["reps"][0])["trunk_lean"]["status"] == "not_measured"

def test_no_confident_foot_points_gives_not_measured_plant_foot_angle(validator):
    """Regression: low-confidence foot points used to give plant_foot_angle = 0, 'in_range'."""
    clip = synthetic_rep("front")
    for f in range(len(clip.kps)):
        for side in ("left", "right"):
            for part in ("heel", "foot_index"):
                clip.set_conf(f, f"{side}_{part}", 0.05)
    out = compute_metrics(clip.pose(), clip.ball_doc(), reps_doc(rep()), PLAYER)
    check_contract(validator, out)
    m = by_name(out["reps"][0])["plant_foot_angle"]
    assert (m["value"], m["status"]) == (None, "not_measured")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_value_is_not_measured(bad):
    cfg = {"range": [140, 180], "unit": "degrees", "views": ["side"]}
    e = metric_entry("ankle_lock", bad, 5, cfg, "side")
    assert (e["value"], e["status"]) == (None, "not_measured")


# --- contract fixtures (side-on, noisy pose; truth measured on the noise-free skeleton) -------------

# Tolerances reflect the fixture's ~1.8 px keypoint noise (foot points are only ~48 px apart).
TOLERANCE = {"body_heights": 0.01, "degrees": 3.0, "m_per_s": 1.0}
TOLERANCE_BY_METRIC = {"ankle_lock": 6.0}


@pytest.fixture(scope="module")
def fixture_docs():
    return load("pose.example.json"), load("ball.example.json"), load("reps.example.json"), load("player.example.json")


def test_fixture_metrics_match_ground_truth(validator, fixture_docs):
    pose, ball, reps, player = fixture_docs
    out = compute_metrics(pose, ball, reps, player)
    check_contract(validator, out, pose, reps)
    truth = load("metrics.example.json")
    assert [r["rep_id"] for r in out["reps"]] == [r["rep_id"] for r in truth["reps"]]
    for got_rep, true_rep in zip(out["reps"], truth["reps"]):
        assert got_rep["foot"] == true_rep["foot"]
        got = by_name(got_rep)
        for t in true_rep["metrics"]:
            g = got[t["name"]]
            where = f"{true_rep['rep_id']}/{t['name']}"
            assert g["status"] == t["status"], where
            assert g["target"] == t["target"] and g["unit"] == t["unit"], where
            if t["value"] is None:
                assert g["value"] is None, where
                continue
            tol = TOLERANCE_BY_METRIC.get(t["name"], TOLERANCE[t["unit"]])
            assert g["value"] == pytest.approx(t["value"], abs=tol), where
            assert g["frame"] == t["frame"], where


def test_fixture_mirrored_left_right_gives_same_metrics(fixture_docs):
    """The contract says not to assume the player runs toward +x."""
    pose, ball, reps, player = fixture_docs
    W = pose["width"]
    mpose, mball = copy.deepcopy(pose), copy.deepcopy(ball)
    for fr in mpose["frames"]:
        for v in fr["keypoints"].values():
            v[0] = W - v[0]
    for fr in mball["frames"]:
        if fr["ball"]:
            fr["ball"][0] = W - fr["ball"][0]
    a, b = compute_metrics(pose, ball, reps, player), compute_metrics(mpose, mball, reps, player)
    for ra, rb in zip(a["reps"], b["reps"]):
        for ma, mb in zip(ra["metrics"], rb["metrics"]):
            assert ma["status"] == mb["status"]
            if ma["value"] is not None:
                assert mb["value"] == pytest.approx(ma["value"], abs=0.0011 if ma["unit"] == "body_heights" else 0.11)


# --- CLI -------------------------------------------------------------------------------------

def test_cli_any_input_order_and_custom_targets(tmp_path, fixture_docs):
    pose, ball, reps, player = fixture_docs
    paths = {}
    for name, doc in (("p.json", player), ("r.json", reps), ("b.json", ball), ("pose.json", pose)):
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(doc))
    targets = yaml.safe_load(DEFAULT_TARGETS.read_text())
    targets["shot_instep"]["ball_speed"]["range"] = [0, 100]
    (tmp_path / "t.yaml").write_text(yaml.safe_dump(targets))
    out = tmp_path / "metrics.json"
    main(["--in", *map(str, paths.values()), "--out", str(out), "--targets", str(tmp_path / "t.yaml")])
    result = json.loads(out.read_text())
    assert result == compute_metrics(pose, ball, reps, player, targets)
    assert all(by_name(r)["ball_speed"]["status"] == "in_range" for r in result["reps"])


def test_cli_rejects_missing_inputs(tmp_path, fixture_docs):
    (tmp_path / "pose.json").write_text(json.dumps(fixture_docs[0]))
    with pytest.raises(SystemExit):
        main(["--in", str(tmp_path / "pose.json"), "--out", str(tmp_path / "m.json")])
