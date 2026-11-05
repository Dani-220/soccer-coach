"""plan.json: a one-week drill plan built from metrics.json, player.json and the drill library.

The weak foot gets more of the week's shooting reps when its reps score worse than the strong foot's.
"""
import datetime as dt

from .cards import OUTCOME_METRICS, SCHEMA_VERSION, drills_for, severity, weak_foot

DAY_PATTERNS = {
    1: ["wed"],
    2: ["tue", "fri"],
    3: ["mon", "wed", "fri"],
    4: ["mon", "tue", "thu", "sat"],
    5: ["mon", "tue", "wed", "fri", "sat"],
    6: ["mon", "tue", "wed", "thu", "fri", "sat"],
    7: ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
}
SESSION_MINUTES = {"casual": 30, "pro": 45}
BASE_WEAK_SHARE, MAX_WEAK_SHARE = 0.5, 0.8
RETEST_DRILL = "d_shot_speed_radar"
DEFAULT_FOCUS = "ball_speed"
DRILLS_PER_FOCUS = 4


def foot_score(metrics, foot):
    """Mean share of measured metrics that are out_of_range, over this foot's reps (None: no measured reps)."""
    shares = []
    for rep in metrics["reps"]:
        measured = [m for m in rep["metrics"] if m["status"] != "not_measured"]
        if rep["foot"] == foot and measured:
            shares.append(sum(m["status"] == "out_of_range" for m in measured) / len(measured))
    return sum(shares) / len(shares) if shares else None


def weak_foot_share(metrics, player):
    """0.5 by default; up to 0.8 as the weak foot's score falls behind the strong foot's."""
    weak = foot_score(metrics, weak_foot(player))
    strong = foot_score(metrics, player["strong_foot"])
    if weak is None or strong is None or weak <= strong:
        return BASE_WEAK_SHARE
    gap = min(1.0, (weak - strong) / 0.5)
    return round(BASE_WEAK_SHARE + (MAX_WEAK_SHARE - BASE_WEAK_SHARE) * gap, 2)


def choose_focus(metrics, n=3):
    """Up to n metrics with the most total severity, technique before outcome. With nothing out of range,
    the measured metric closest to the edge of its target; with nothing measured, ball_speed."""
    total, margin = {}, {}
    for rep in metrics["reps"]:
        for m in rep["metrics"]:
            s = severity(m)
            if s > 0:
                total[m["name"]] = total.get(m["name"], 0.0) + s
            elif m["value"] is not None:
                lo, hi = m["target"]
                edge = min(m["value"] - lo, hi - m["value"]) / ((hi - lo) or 1.0)
                margin[m["name"]] = min(edge, margin.get(m["name"], edge))
    ranked = sorted(total, key=lambda k: -total[k])
    focus = [k for k in ranked if k not in OUTCOME_METRICS][:n] or ranked[:n]
    if not focus and margin:
        technique = [k for k in margin if k not in OUTCOME_METRICS] or list(margin)
        focus = [min(technique, key=lambda k: margin[k])]
    return focus or [DEFAULT_FOCUS]


def next_monday(today=None):
    today = today or dt.date.today()
    return today + dt.timedelta(days=(7 - today.weekday()) % 7)


def uses_ball(drill):
    return any(e.strip().lower() in ("ball", "balls") for e in drill.get("equipment", []))


def drill_queue(focus, drills, level):
    """Round-robin over the focus metrics' drills, so every focus gets work each session."""
    per = [drills_for(f, drills, level, n=DRILLS_PER_FOCUS) for f in focus]
    queue = []
    for i in range(max(map(len, per), default=0)):
        for ids in per:
            if i < len(ids) and ids[i] not in queue:
                queue.append(ids[i])
    return queue


def build_plan(metrics, player, drills, week_start=None):
    level = player["level"]
    by_id = {d["id"]: d for d in drills}
    focus = choose_focus(metrics)
    share = weak_foot_share(metrics, player)
    weak, strong = weak_foot(player), player["strong_foot"]
    week_start = week_start or next_monday()
    days = DAY_PATTERNS[max(1, min(7, player.get("sessions_per_week") or 3))]
    queue = drill_queue(focus, drills, level)
    cap = SESSION_MINUTES[level]
    extra_set = 1 if level == "pro" else 0

    weak_reps = total_reps = 0
    pos = 0
    out_days = []
    for di, day in enumerate(days):
        last = di == len(days) - 1
        budget = cap - (by_id[RETEST_DRILL]["minutes"] if last and RETEST_DRILL in by_id else 0)
        picked, minutes = [], 0
        for _ in range(len(queue)):
            d = by_id[queue[pos % len(queue)]]
            if len(picked) >= 2 and minutes + d["minutes"] > budget:
                break
            picked.append(d)
            minutes += d["minutes"]
            pos += 1

        entries = []
        for d in picked:
            sets, reps = d["default_sets"] + extra_set, d["default_reps"]
            entry = {"drill_id": d["id"], "sets": sets, "reps": reps}
            if not uses_ball(d):
                entry["foot"] = "both"
            else:
                # Greedy split so the week's ball reps land close to the weak-foot share.
                go_weak = total_reps == 0 or weak_reps / total_reps < share
                entry["foot"] = weak if go_weak else strong
                weak_reps += sets * reps if go_weak else 0
                total_reps += sets * reps
            entries.append(entry)
        if entries:
            hit = [f for f in focus if f in by_id[entries[0]["drill_id"]]["fixes"]]
            if hit:
                entries[0]["note"] = "Focus: " + ", ".join(h.replace("_", " ") for h in hit)
        if last and RETEST_DRILL in by_id:
            d = by_id[RETEST_DRILL]
            entries.append({"drill_id": d["id"], "sets": d["default_sets"], "reps": d["default_reps"],
                            "foot": "both", "note": "Film this session for the re-test"})
        out_days.append({"day": day, "drills": entries})

    return {
        "schema_version": SCHEMA_VERSION,
        "player_id": metrics.get("player_id") or player["player_id"],
        "week_start": week_start.isoformat(),
        "focus": focus,
        "weak_foot_share": share,
        "days": out_days,
        "retest_on": (week_start + dt.timedelta(days=7)).isoformat(),
    }
