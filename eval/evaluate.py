"""Scores pipeline output against labels.csv, one video at a time.

A labeled rep and a detected rep match when their contact frames are within `match_tol_s` of each
other; the pairing is one-to-one and minimises the total contact error. Everything else (contact and
plant error, foot accuracy, coach-score correlation) is measured on the matched pairs only.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

# Use the contract's own file-name rule (pose.json, clip_pose.json, pose.example.json, ...).
_spec = importlib.util.spec_from_file_location("contract_validate", ROOT / "contract" / "tools" / "validate.py")
_validate = importlib.util.module_from_spec(_spec)
_dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
_spec.loader.exec_module(_validate)
sys.dont_write_bytecode = _dont_write
file_type = _validate.file_type

RUN_TYPES = ("pose", "ball", "reps", "metrics")
DEFAULT_FPS = 30.0
MIN_CORRELATION_N = 3


@dataclass
class Label:
    video: str
    rep_id: str
    move: str
    foot: str
    plant: int | None
    contact: int
    coach_score: float | None
    notes: str = ""


@dataclass
class Run:
    video: str
    docs: dict = field(default_factory=dict)      # type -> parsed JSON
    paths: dict = field(default_factory=dict)     # type -> Path
    warnings: list = field(default_factory=list)

    @property
    def fps(self):
        for k in ("pose", "ball"):
            if k in self.docs:
                return float(self.docs[k]["fps"])
        return None


def video_key(name: str) -> str:
    return Path(name).name


def _int_or_none(s):
    s = (s or "").strip()
    return int(float(s)) if s else None


def _float_or_none(s):
    s = (s or "").strip()
    return float(s) if s else None


def load_labels(path) -> dict[str, list[Label]]:
    by_video: dict[str, list[Label]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not (row.get("video") or "").strip():
                continue
            lab = Label(
                video=video_key(row["video"].strip()),
                rep_id=row["rep_id"].strip(),
                move=(row.get("move") or "").strip(),
                foot=(row.get("foot") or "").strip().lower(),
                plant=_int_or_none(row.get("plant_frame")),
                contact=_int_or_none(row["contact_frame"]),
                coach_score=_float_or_none(row.get("coach_score_1to5")),
                notes=(row.get("notes") or "").strip(),
            )
            if lab.contact is None:
                raise ValueError(f"{path}: {lab.video} {lab.rep_id} has no contact_frame")
            by_video.setdefault(lab.video, []).append(lab)
    for labs in by_video.values():
        labs.sort(key=lambda l: l.contact)
    return by_video


def find_runs(runs_dir) -> dict[str, Run]:
    """Every pose/ball/reps/metrics JSON under runs_dir, grouped by the `video` field inside it.

    If a video has two files of one type, the newest is used and the run gets a warning.
    """
    runs: dict[str, Run] = {}
    for p in sorted(Path(runs_dir).rglob("*.json")):
        kind = file_type(p)
        if kind not in RUN_TYPES:
            continue
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict) or "video" not in doc:
            continue
        run = runs.setdefault(video_key(doc["video"]), Run(video_key(doc["video"])))
        old = run.paths.get(kind)
        if old is not None:
            keep, drop = (p, old) if p.stat().st_mtime >= old.stat().st_mtime else (old, p)
            run.warnings.append(f"two {kind} files; using {keep}, ignoring {drop}")
            if keep == old:
                continue
        run.docs[kind], run.paths[kind] = doc, p
    return runs


def match_reps(labels: list[Label], detected: list[dict], tol_frames: float):
    """One-to-one pairs (label, detected rep) with |contact error| <= tol_frames, minimising total error."""
    if not labels or not detected:
        return []
    err = np.array([[abs(d["frames"]["contact"] - l.contact) for d in detected] for l in labels], float)
    cost = np.where(err <= tol_frames, err, 1e9)
    rows, cols = linear_sum_assignment(cost)
    return [(labels[i], detected[j]) for i, j in zip(rows, cols) if err[i, j] <= tol_frames]


def metric_score(metrics_rep: dict | None):
    """Share of measured metrics that are in range (0..1); None when nothing was measured."""
    if not metrics_rep:
        return None
    measured = [m for m in metrics_rep["metrics"] if m["status"] != "not_measured"]
    if not measured:
        return None
    return sum(m["status"] == "in_range" for m in measured) / len(measured)


def correlation(pairs):
    """Spearman rho over (coach score, metric score) pairs; None if too few or either side is constant."""
    pairs = [(c, m) for c, m in pairs if c is not None and m is not None]
    out = {"spearman": None, "n": len(pairs)}
    if len(pairs) >= MIN_CORRELATION_N:
        coach, score = zip(*pairs)
        if len(set(coach)) > 1 and len(set(score)) > 1:
            out["spearman"] = round(float(spearmanr(coach, score).statistic), 3)
    return out


def error_stats(errors):
    if not errors:
        return {"n": 0, "mean_abs": None, "median_abs": None, "max_abs": None, "mean_signed": None}
    e = np.asarray(errors, float)
    return {
        "n": len(errors),
        "mean_abs": round(float(np.mean(np.abs(e))), 2),
        "median_abs": round(float(np.median(np.abs(e))), 2),
        "max_abs": int(np.max(np.abs(e))),
        "mean_signed": round(float(np.mean(e)), 2),
    }


def contact_window_frames(labels: list[Label], window: int, n_frames: int) -> list[int]:
    frames = set()
    for l in labels:
        frames.update(range(max(0, l.contact - window), min(n_frames - 1, l.contact + window) + 1))
    return sorted(frames)


def ball_recall(ball_doc: dict | None, labels: list[Label], window: int):
    if ball_doc is None:
        return {"recall": None, "frames": 0, "with_ball": 0}
    frames = ball_doc["frames"]
    window_frames = contact_window_frames(labels, window, len(frames))
    hits = sum(frames[f]["ball"] is not None for f in window_frames)
    return {
        "recall": round(hits / len(window_frames), 4) if window_frames else None,
        "frames": len(window_frames),
        "with_ball": hits,
    }


def _ratio(num, den, digits=4):
    return round(num / den, digits) if den else None


def evaluate_video(video: str, labels: list[Label], run: Run | None, match_tol_s: float, ball_window_s: float) -> dict:
    run = run or Run(video)
    warnings = list(run.warnings)
    fps = run.fps
    if fps is None:
        fps = DEFAULT_FPS
        if run.docs:
            warnings.append(f"no pose.json or ball.json; assuming {DEFAULT_FPS:g} fps for tolerances")
    missing = [k for k in RUN_TYPES if k not in run.docs]
    if "reps" in missing:
        warnings.append("no reps.json; every labeled rep counts as missed")

    tol = match_tol_s * fps
    window = int(round(ball_window_s * fps))
    detected = sorted((run.docs.get("reps") or {"reps": []})["reps"], key=lambda r: r["frames"]["contact"])
    metrics_by_rep = {r["rep_id"]: r for r in (run.docs.get("metrics") or {"reps": []})["reps"]}

    pairs = match_reps(labels, detected, tol)
    matched_label_ids = {l.rep_id for l, _ in pairs}
    matched_run_ids = {d["rep_id"] for _, d in pairs}

    matches = []
    for l, d in sorted(pairs, key=lambda p: p[0].contact):
        plant = d["frames"].get("plant")
        matches.append({
            "label_rep": l.rep_id,
            "run_rep": d["rep_id"],
            "contact_label": l.contact,
            "contact_run": d["frames"]["contact"],
            "contact_error": d["frames"]["contact"] - l.contact,
            "plant_error": plant - l.plant if plant is not None and l.plant is not None else None,
            "foot_label": l.foot,
            "foot_run": d["foot"],
            "foot_correct": d["foot"] == l.foot,
            "coach_score": l.coach_score,
            "metric_score": metric_score(metrics_by_rep.get(d["rep_id"])),
        })

    n_lab, n_det, n_match = len(labels), len(detected), len(pairs)
    return {
        "video": video,
        "fps": fps,
        "files": {k: str(p) for k, p in run.paths.items()},
        "missing_files": missing,
        "warnings": warnings,
        "labeled": n_lab,
        "detected": n_det,
        "matched": n_match,
        "precision": _ratio(n_match, n_det),
        "recall": _ratio(n_match, n_lab),
        "contact_error_frames": error_stats([m["contact_error"] for m in matches]),
        "plant_error_frames": error_stats([m["plant_error"] for m in matches if m["plant_error"] is not None]),
        "foot_accuracy": _ratio(sum(m["foot_correct"] for m in matches), n_match),
        "coach_correlation": correlation([(m["coach_score"], m["metric_score"]) for m in matches]),
        "ball": ball_recall(run.docs.get("ball"), labels, window),
        "matches": matches,
        "missed_labels": [l.rep_id for l in labels if l.rep_id not in matched_label_ids],
        "false_positives": [d["rep_id"] for d in detected if d["rep_id"] not in matched_run_ids],
    }


def overall(videos: list[dict]) -> dict:
    """Pooled over every rep and every window frame, not an average of per-video numbers."""
    matches = [m for v in videos for m in v["matches"]]
    n_lab = sum(v["labeled"] for v in videos)
    n_det = sum(v["detected"] for v in videos)
    ball_frames = sum(v["ball"]["frames"] for v in videos)
    ball_hits = sum(v["ball"]["with_ball"] for v in videos)
    return {
        "videos": len(videos),
        "labeled": n_lab,
        "detected": n_det,
        "matched": len(matches),
        "precision": _ratio(len(matches), n_det),
        "recall": _ratio(len(matches), n_lab),
        "contact_error_frames": error_stats([m["contact_error"] for m in matches]),
        "plant_error_frames": error_stats([m["plant_error"] for m in matches if m["plant_error"] is not None]),
        "foot_accuracy": _ratio(sum(m["foot_correct"] for m in matches), len(matches)),
        "coach_correlation": correlation([(m["coach_score"], m["metric_score"]) for m in matches]),
        "ball": {
            "recall": _ratio(ball_hits, ball_frames),
            "frames": ball_frames,
            "with_ball": ball_hits,
        },
    }


def evaluate(labels_path, runs_dir, match_tol_s=0.25, ball_window_s=0.25) -> dict:
    labels = load_labels(labels_path)
    runs = find_runs(runs_dir)
    videos = [evaluate_video(v, labels[v], runs.get(v), match_tol_s, ball_window_s) for v in sorted(labels)]
    return {
        "settings": {"match_tol_s": match_tol_s, "ball_window_s": ball_window_s},
        "videos": videos,
        "overall": overall(videos),
        "unlabeled_runs": sorted(set(runs) - set(labels)),
    }
