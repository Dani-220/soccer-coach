"""CLI:  python -m eval --labels fixtures/labels.csv --runs out/ [--report eval/report.json]

Finds every pose/ball/reps/metrics JSON under --runs, groups them by the `video` field inside, and
scores each labeled video: rep precision and recall, contact-frame error, kicking-foot accuracy,
coach-score correlation and ball recall around the labeled contacts. Prints a table, writes
report.json and appends one line per run to --history so detector changes can be compared over time.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import ROOT, evaluate

DEFAULT_REPORT = ROOT / "eval" / "report.json"
DEFAULT_HISTORY = ROOT / "eval" / "history.jsonl"


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def num(x, fmt="{:.1f}"):
    return "—" if x is None else fmt.format(x)


def table(report) -> str:
    head = ["video", "lab", "det", "hit", "prec", "recall", "contact MAE", "bias", "foot", "coach ρ (n)", "ball recall"]
    rows = []
    for v in report["videos"] + [dict(report["overall"], video="ALL")]:
        ce, cc, b = v["contact_error_frames"], v["coach_correlation"], v["ball"]
        rows.append([
            v["video"], str(v["labeled"]), str(v["detected"]), str(v["matched"]),
            pct(v["precision"]), pct(v["recall"]),
            num(ce["mean_abs"]) + " f" if ce["mean_abs"] is not None else "—",
            num(ce["mean_signed"], "{:+.1f}"),
            pct(v["foot_accuracy"]),
            f"{num(cc['spearman'], '{:+.2f}')} ({cc['n']})",
            f"{pct(b['recall'])} ({b['with_ball']}/{b['frames']})" if b["frames"] else "—",
        ])
    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    rule = "-" * len(fmt(head))
    return "\n".join([fmt(head), rule, *map(fmt, rows[:-1]), rule, fmt(rows[-1])])


def history_entry(report) -> dict:
    return {
        "generated_at": report["generated_at"],
        "git_commit": report["git_commit"],
        "runs": report["runs"],
        "ball_recall": report["overall"]["ball"]["recall"],
        "rep_precision": report["overall"]["precision"],
        "rep_recall": report["overall"]["recall"],
        "contact_mae": report["overall"]["contact_error_frames"]["mean_abs"],
        "videos": {v["video"]: v["ball"]["recall"] for v in report["videos"]},
    }


def last_history(path: Path, settings: dict):
    """The newest earlier entry made with the same settings (different windows aren't comparable)."""
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("settings") == settings:
            return e
    return None


def ball_change(prev, entry) -> str:
    if prev is None or prev.get("ball_recall") is None or entry["ball_recall"] is None:
        return ""
    d = 100 * (entry["ball_recall"] - prev["ball_recall"])
    return f" ({d:+.1f} pts vs {pct(prev['ball_recall'])} at {prev.get('git_commit') or '?'}, {prev['generated_at'][:10]})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="labels.csv (video, rep_id, move, foot, plant_frame, contact_frame, coach_score_1to5, notes)")
    ap.add_argument("--runs", required=True, help="folder searched recursively for pipeline JSON")
    ap.add_argument("--report", default=str(DEFAULT_REPORT), help="report JSON to write (default: eval/report.json)")
    ap.add_argument("--history", default=str(DEFAULT_HISTORY), help="JSONL file to append a summary to (default: eval/history.jsonl)")
    ap.add_argument("--no-history", action="store_true", help="don't read or append --history")
    ap.add_argument("--match-tol", type=float, default=0.25, help="max contact-frame error, in seconds, for a detected rep to count as a labeled one (default: %(default)s)")
    ap.add_argument("--ball-window", type=float, default=0.25, help="ball recall counts frames within this many seconds of each labeled contact (default: %(default)s)")
    a = ap.parse_args(argv)

    if not Path(a.labels).is_file():
        ap.error(f"labels file not found: {a.labels}")
    if not Path(a.runs).is_dir():
        ap.error(f"runs folder not found: {a.runs}")

    report = evaluate(a.labels, a.runs, a.match_tol, a.ball_window)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "labels": str(a.labels),
        "runs": str(a.runs),
        **report,
    }

    print(table(report))
    for v in report["videos"]:
        for w in v["warnings"]:
            print(f"warning: {v['video']}: {w}")
        if v["missed_labels"] or v["false_positives"]:
            print(f"{v['video']}: missed {v['missed_labels'] or '-'}, false positives {v['false_positives'] or '-'}")
    if report["unlabeled_runs"]:
        print(f"no labels for: {', '.join(report['unlabeled_runs'])}")

    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"report written to {out}")

    if not a.no_history:
        hist = Path(a.history)
        entry = dict(history_entry(report), settings=report["settings"])
        prev = last_history(hist, report["settings"])
        hist.parent.mkdir(parents=True, exist_ok=True)
        with hist.open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"ball recall {pct(entry['ball_recall'])}{ball_change(prev, entry)}; appended to {hist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
