"""App logic that doesn't need Streamlit: running the pipeline, reading a run's files, overlays, tagging."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SOCCER_COACH_OUT", REPO / "out"))  # tests point this at a temp folder
DB_PATH = OUT / "history.sqlite"
TARGETS = REPO / "pipeline" / "metrics" / "targets.yaml"
EXAMPLE_CARDS = REPO / "contract" / "examples" / "cards.example.json"
DRILL_FILES = [REPO / "pipeline" / "coach" / "drills.yaml", REPO / "contract" / "examples" / "drills.example.yaml"]

UNIT_LABELS = {"body_heights": "body heights", "degrees": "°", "m_per_s": "m/s"}


def load_json(path):
    return json.loads(Path(path).read_text())


def session_name(player_id, stamp, video_name):
    """Folder name under out/ for one run, e.g. p1_20261104-1530_shots."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(video_name).stem).strip("-") or "video"
    return f"{re.sub(r'[^A-Za-z0-9_-]+', '-', player_id)}_{stamp}_{stem}"


def run_pipeline(video, player_path, out_dir, on_line=None, reuse_pose=False):
    """Run run.py and return (exit code, full log). on_line(line) is called as output arrives."""
    cmd = [sys.executable, "-u", str(REPO / "run.py"), "--player", str(player_path), "--out", str(out_dir)]
    cmd += ["--reuse-pose"] if reuse_pose else [str(video)]
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if on_line:
            on_line(line)
    return proc.wait(), "".join(lines)


# --- tagging and labels ---

def foot_tag(foot, strong_foot):
    return "strong" if foot == strong_foot else "weak"


def foot_label(foot, strong_foot):
    return f"{foot.capitalize()} ({foot_tag(foot, strong_foot)})"


def metric_label(name):
    return name.replace("_", " ").capitalize()


def format_value(value, unit, height_cm=None):
    """'0.239 body heights (≈ 39 cm)', '-16.0°', '20.6 m/s'."""
    if value is None:
        return "not measured"
    text = f"{value:g}°" if unit == "degrees" else f"{value:g} {UNIT_LABELS.get(unit, unit)}"
    if unit == "body_heights" and height_cm:
        text += f" (≈ {value * height_cm * 0.93:.0f} cm)"
    return text


def format_target(target, unit):
    lo, hi = target
    return f"{lo:g} to {hi:g}{'°' if unit == 'degrees' else ' ' + UNIT_LABELS.get(unit, unit)}"


# --- not measured ---

def load_targets(path=TARGETS):
    return yaml.safe_load(Path(path).read_text())


def session_view(metrics_doc, targets):
    """'side' or 'front' if the measured metrics show which camera view the video was, else None."""
    seen = set()
    for r in metrics_doc["reps"]:
        cfg = targets.get(r.get("move", "shot_instep"), {})
        for m in r["metrics"]:
            views = cfg.get(m["name"], {}).get("views", [])
            if m["value"] is not None and len(views) == 1:
                seen.add(views[0])
    return seen.pop() if len(seen) == 1 else None


def not_measured(rep_metrics, move, targets, view):
    """Group a rep's not_measured metrics by what would fix them.

    Returns {"side": [...], "front": [...], "other": [...]}: metrics that need a side-on or front-on camera
    (their targets.yaml `views` exclude the view this video was filmed from), and metrics that could have been
    measured from this view but weren't (player not detected, ball hidden, no height for ball_speed, ...).
    """
    cfg = targets.get(move, {})
    out = {"side": [], "front": [], "other": []}
    for m in rep_metrics:
        if m["status"] != "not_measured":
            continue
        views = cfg.get(m["name"], {}).get("views", ["side", "front"])
        if len(views) == 1 and view != views[0]:
            out[views[0]].append(m["name"])
        else:
            out["other"].append(m["name"])
    return out


# --- cards and drills ---

def load_cards(out_dir):
    """({rep_id: [card, ...]}, source). source is 'pipeline', or 'example' until Workstream F exists."""
    path = Path(out_dir) / "cards.json"
    source = "pipeline" if path.exists() else "example"
    doc = load_json(path if path.exists() else EXAMPLE_CARDS)
    return {r["rep_id"]: r["cards"] for r in doc["reps"]}, source


def load_drills():
    for p in DRILL_FILES:
        if p.exists():
            return {d["id"]: d for d in yaml.safe_load(p.read_text()) or []}
    return {}


# --- overlays ---

def previous_rep(sessions, reps_by_session, session_id, rep_id):
    """(session, rep) to compare rep_id against: the rep before it in the same session, else the last rep
    of the player's previous session. None for the very first rep."""
    idx = next(i for i, s in enumerate(sessions) if s["session_id"] == session_id)
    ids = [r["rep_id"] for r in reps_by_session[session_id]]
    pos = ids.index(rep_id)
    if pos > 0:
        return sessions[idx], ids[pos - 1]
    for s in reversed(sessions[:idx]):
        if reps_by_session[s["session_id"]]:
            return s, reps_by_session[s["session_id"]][-1]["rep_id"]
    return None


def overlay_paths(new_dir, new_rep, old_dir, old_rep):
    """Where the overlay mp4/png for this pair lives. Same-session pairs use run.py's names."""
    name = f"{new_rep}_vs_{old_rep}" if Path(new_dir) == Path(old_dir) else f"{new_rep}_vs_{Path(old_dir).name}_{old_rep}"
    mp4 = Path(new_dir) / "overlays" / f"{name}.mp4"
    return mp4, mp4.with_suffix(".png")


def make_overlay(new_dir, new_rep, old_dir, old_rep):
    """Render the overlay with Workstream E's CLI if it isn't on disk yet. Returns (mp4, png, error)."""
    mp4, png = overlay_paths(new_dir, new_rep, old_dir, old_rep)
    if mp4.exists() and png.exists():
        return mp4, png, None
    mp4.parent.mkdir(parents=True, exist_ok=True)
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    inputs = [new_dir / "pose.json", new_dir / "reps.json"]
    if new_dir != old_dir:
        inputs += [old_dir / "pose.json", old_dir / "reps.json"]
    proc = subprocess.run([sys.executable, "-m", "pipeline.overlay", "--in", *map(str, inputs),
                           "--new", new_rep, "--old", old_rep, "--out", str(mp4)],
                          cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, None, (proc.stderr or proc.stdout).strip()
    return mp4, png, None


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def browser_video(mp4):
    """An H.264 copy of an overlay mp4 that browsers can play (E writes mp4v). None without ffmpeg."""
    mp4 = Path(mp4)
    web = mp4.with_name(mp4.stem + ".web.mp4")
    if web.exists() and web.stat().st_mtime >= mp4.stat().st_mtime:
        return web
    exe = _ffmpeg()
    if not exe:
        return None
    proc = subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(web)], capture_output=True)
    return web if proc.returncode == 0 else None


# --- weak-foot dashboard ---

def foot_trend(metric_rows, strong_foot):
    """Per session, metric and foot: mean value, rep count and in-range count, for the dashboard."""
    groups = {}
    for m in metric_rows:
        if m["value"] is None:
            continue
        key = (m["session_id"], m["name"], m["foot"])
        g = groups.setdefault(key, {"session": m["session"], "recorded_at": m["recorded_at"], "metric": m["name"],
                                    "foot": m["foot"], "foot_label": foot_label(m["foot"], strong_foot),
                                    "unit": m["unit"], "target_lo": m["target_lo"], "target_hi": m["target_hi"],
                                    "values": [], "in_range": 0})
        g["values"].append(m["value"])
        g["in_range"] += m["status"] == "in_range"
    out = []
    for g in groups.values():
        vals = g.pop("values")
        out.append({**g, "value": round(sum(vals) / len(vals), 3), "reps": len(vals)})
    return out
