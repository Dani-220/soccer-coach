"""Run from the repo root:  python -m pytest eval -v"""
import json
import os
from pathlib import Path

import pytest

from eval.__main__ import main
from eval.evaluate import Label, Run, evaluate, evaluate_video, find_runs, load_labels, match_reps, metric_score

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contract" / "examples"
LABELS = EXAMPLES / "labels.example.csv"
VIDEO = "synthetic_shots.mp4"
FPS = 120


def load(name):
    return json.loads((EXAMPLES / f"{name}.example.json").read_text())


def write_run(folder, **docs):
    """Writes pose/ball/reps/metrics (the examples unless overridden) into folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for kind in ("pose", "ball", "reps", "metrics"):
        doc = docs.get(kind, load(kind))
        if doc is not None:
            (folder / f"{kind}.json").write_text(json.dumps(doc))
    return folder


def rep(rep_id, contact, foot="right"):
    return {"rep_id": rep_id, "move": "shot_instep", "foot": foot,
            "frames": {"start": contact - 50, "plant": contact - 7, "contact": contact, "end": contact + 40},
            "confidence": 0.9}


def label(rep_id, contact, foot="right", coach=None, plant=None):
    return Label(VIDEO, rep_id, "shot_instep", foot, plant, contact, coach)


def test_examples_score_perfectly():
    report = evaluate(LABELS, EXAMPLES)
    (v,) = report["videos"]
    assert v["video"] == VIDEO and v["fps"] == FPS
    assert (v["labeled"], v["detected"], v["matched"]) == (2, 2, 2)
    assert v["precision"] == v["recall"] == v["foot_accuracy"] == 1.0
    assert v["contact_error_frames"]["max_abs"] == 0
    assert v["plant_error_frames"]["max_abs"] == 0
    assert v["missing_files"] == [] and v["warnings"] == []
    assert v["missed_labels"] == v["false_positives"] == []
    # r001 is all in range (coach 4); r002 has six problems (coach 2). Two pairs are too few for rho.
    assert [m["metric_score"] for m in v["matches"]] == [1.0, pytest.approx(0)]
    assert v["coach_correlation"] == {"spearman": None, "n": 2}


def test_ball_recall_counts_window_frames_around_each_contact():
    ball = load("ball")
    v = evaluate(LABELS, EXAMPLES, ball_window_s=0.1)["videos"][0]
    window = [f for c in (103, 331) for f in range(c - 12, c + 13)]
    expected = sum(ball["frames"][f]["ball"] is not None for f in window)
    assert v["ball"] == {"recall": round(expected / 50, 4), "frames": 50, "with_ball": expected}
    assert expected < 50   # the ball is hidden at r002's contact


def test_detector_change_shows_in_ball_recall(tmp_path):
    ball = load("ball")
    for fr in ball["frames"]:
        fr["ball"] = fr["ball"] or [0.0, 0.0, 5.0, 0.5]
    v = evaluate(LABELS, write_run(tmp_path, ball=ball))["videos"][0]
    assert v["ball"]["recall"] == 1.0


def test_shifted_contact_wrong_foot_and_extra_rep(tmp_path):
    reps = load("reps")
    reps["reps"][0]["frames"]["contact"] += 3
    reps["reps"][1]["foot"] = "right"
    reps["reps"].append(rep("r003", 390))
    v = evaluate(LABELS, write_run(tmp_path, reps=reps))["videos"][0]
    assert (v["detected"], v["matched"]) == (3, 2)
    assert v["precision"] == pytest.approx(2 / 3, abs=1e-4) and v["recall"] == 1.0
    assert v["contact_error_frames"] == {"n": 2, "mean_abs": 1.5, "median_abs": 1.5, "max_abs": 3, "mean_signed": 1.5}
    assert v["foot_accuracy"] == 0.5
    assert v["false_positives"] == ["r003"]


def test_contact_outside_tolerance_is_a_miss_and_a_false_positive(tmp_path):
    reps = load("reps")
    reps["reps"][1]["frames"]["contact"] = 331 + 31   # 0.26 s late at 120 fps
    reps["reps"][1]["frames"]["end"] = 380
    v = evaluate(LABELS, write_run(tmp_path, reps=reps), match_tol_s=0.25)["videos"][0]
    assert v["missed_labels"] == ["r002"] and v["false_positives"] == ["r002"]
    assert v["precision"] == v["recall"] == 0.5


def test_matching_is_one_to_one_and_picks_the_closest():
    labels = [label("r001", 100), label("r002", 140)]
    detected = [rep("a", 118), rep("b", 101), rep("c", 300)]
    pairs = match_reps(labels, detected, tol_frames=30)
    assert [(l.rep_id, d["rep_id"]) for l, d in pairs] == [("r001", "b"), ("r002", "a")]
    # A single detection between two labels counts once.
    assert len(match_reps(labels, [rep("x", 120)], tol_frames=30)) == 1


def test_coach_correlation_uses_metric_scores():
    def metrics_rep(rep_id, n_in, n_out):
        ms = [{"name": "trunk_lean", "status": "in_range"}] * n_in + [{"name": "trunk_lean", "status": "out_of_range"}] * n_out
        return {"rep_id": rep_id, "metrics": ms + [{"name": "plant_foot_angle", "status": "not_measured"}]}

    labels = [label(f"r{i}", 100 * i, coach=c) for i, c in enumerate([5, 4, 2, 1, 3], 1)]
    detected = [rep(f"d{i}", 100 * i) for i in range(1, 6)]
    metrics = [metrics_rep("d1", 6, 0), metrics_rep("d2", 5, 1), metrics_rep("d3", 2, 4), metrics_rep("d4", 0, 6), metrics_rep("d5", 3, 3)]
    run = Run(VIDEO, docs={"reps": {"reps": detected}, "metrics": {"reps": metrics}})
    v = evaluate_video(VIDEO, labels, run, match_tol_s=0.25, ball_window_s=0.25)
    assert v["coach_correlation"] == {"spearman": 1.0, "n": 5}
    assert metric_score(metrics[1]) == pytest.approx(5 / 6)
    assert metric_score({"metrics": [{"status": "not_measured"}]}) is None


def test_missing_run_counts_every_label_as_missed(tmp_path):
    v = evaluate(LABELS, tmp_path)["videos"][0]
    assert (v["detected"], v["matched"], v["recall"], v["precision"]) == (0, 0, 0.0, None)
    assert v["ball"]["recall"] is None
    assert any("no reps.json" in w for w in v["warnings"])


def test_runs_are_grouped_by_video_field_in_any_layout(tmp_path):
    write_run(tmp_path / "a" / "nested")
    other = {k: dict(load(k), video="clips/other.mp4") for k in ("pose", "ball", "reps")}
    for k, d in other.items():
        (tmp_path / f"other_{k}.json").write_text(json.dumps(d))
    (tmp_path / "notes.json").write_text("{}")
    runs = find_runs(tmp_path)
    assert set(runs) == {VIDEO, "other.mp4"}
    assert set(runs["other.mp4"].docs) == {"pose", "ball", "reps"}
    assert evaluate(LABELS, tmp_path)["unlabeled_runs"] == ["other.mp4"]


def test_duplicate_files_use_the_newest_and_warn(tmp_path):
    write_run(tmp_path / "old")
    reps = load("reps")
    reps["reps"] = reps["reps"][:1]
    write_run(tmp_path / "new", reps=reps, pose=None, ball=None, metrics=None)
    old, new = tmp_path / "old" / "reps.json", tmp_path / "new" / "reps.json"
    os.utime(old, (1, 1))
    run = find_runs(tmp_path)[VIDEO]
    assert run.paths["reps"] == new
    assert any("two reps files" in w for w in run.warnings)


def test_labels_with_blank_optional_fields(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video,rep_id,move,foot,plant_frame,contact_frame,coach_score_1to5,notes\n"
                 "dir/a.mp4,r001,shot_instep,Left,,50,,\n\n")
    (lab,) = load_labels(p)["a.mp4"]
    assert (lab.foot, lab.plant, lab.contact, lab.coach_score) == ("left", None, 50, None)
    p.write_text("video,rep_id,move,foot,plant_frame,contact_frame,coach_score_1to5,notes\na.mp4,r001,shot_instep,left,40,,3,\n")
    with pytest.raises(ValueError, match="contact_frame"):
        load_labels(p)


def test_cli_writes_report_and_compares_ball_recall_with_history(tmp_path, capsys):
    report, history = tmp_path / "report.json", tmp_path / "history.jsonl"
    args = ["--labels", str(LABELS), "--runs", str(EXAMPLES), "--report", str(report), "--history", str(history)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert VIDEO in out and "ALL" in out and "ball recall" in out

    data = json.loads(report.read_text())
    assert data["overall"]["recall"] == 1.0 and data["generated_at"] and "git_commit" in data

    ball = load("ball")
    for fr in ball["frames"]:
        fr["ball"] = fr["ball"] or [0.0, 0.0, 5.0, 0.5]
    runs = write_run(tmp_path / "run2", ball=ball)
    assert main(["--labels", str(LABELS), "--runs", str(runs), "--report", str(report), "--history", str(history)]) == 0
    assert "pts vs 75%" in capsys.readouterr().out
    entries = [json.loads(l) for l in history.read_text().splitlines()]
    assert len(entries) == 2 and entries[1]["ball_recall"] == 1.0
    assert entries[1]["videos"] == {VIDEO: 1.0}


def test_cli_rejects_missing_paths(tmp_path):
    with pytest.raises(SystemExit):
        main(["--labels", str(tmp_path / "nope.csv"), "--runs", str(EXAMPLES), "--no-history"])
