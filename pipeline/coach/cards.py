"""cards.json from metrics.json + player.json: the 1-2 worst out_of_range metrics per rep."""
from pathlib import Path

import yaml

from .wording import Writer, headline

SCHEMA_VERSION = "1.0"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DRILLS = HERE / "drills.yaml"
DEFAULT_TARGETS = ROOT / "pipeline" / "metrics" / "targets.yaml"
FALLBACK_TARGETS = ROOT / "contract" / "examples" / "targets.example.yaml"

MAX_CARDS_PER_REP = 2
DRILLS_PER_CARD = 2
# ball_speed is the outcome the technique metrics explain, so it only gets a card when nothing else does.
OUTCOME_METRICS = {"ball_speed"}


def load_drills(path=DEFAULT_DRILLS):
    return yaml.safe_load(Path(path).read_text())


def load_targets(path=None):
    if path is None:
        path = DEFAULT_TARGETS if DEFAULT_TARGETS.is_file() else FALLBACK_TARGETS
    return yaml.safe_load(Path(path).read_text())


def weak_foot(player):
    return "left" if player["strong_foot"] == "right" else "right"


def severity(m):
    """How far outside its target a metric is, in target widths (0 when in range or not measured)."""
    if m["status"] != "out_of_range" or m["value"] is None:
        return 0.0
    lo, hi = m["target"]
    width = (hi - lo) or 1.0
    return max(lo - m["value"], m["value"] - hi, 0.0) / width


def worst_metrics(rep, n=MAX_CARDS_PER_REP):
    """The rep's n worst out_of_range metrics, technique before outcome, then by severity."""
    bad = [m for m in rep["metrics"] if m["status"] == "out_of_range" and m["value"] is not None]
    bad.sort(key=lambda m: (m["name"] in OUTCOME_METRICS, -severity(m)))
    return bad[:n]


def drills_for(metric, drills, level, n=DRILLS_PER_CARD):
    """Drills that fix `metric` at this level, most targeted first (metric listed earliest, fewest fixes)."""
    matches = [d for d in drills if metric in d["fixes"]]
    at_level = [d for d in matches if level in d["levels"]] or matches
    at_level.sort(key=lambda d: (d["fixes"].index(metric), len(d["fixes"])))
    return [d["id"] for d in at_level[:n]]


def round_like(value, unit):
    """Same precision metrics.json uses: 3 decimals for body heights, 1 otherwise."""
    return round(value, 3 if unit == "body_heights" else 1)


def strong_foot_averages(metrics, player):
    """{metric: mean measured value over the strong-foot reps}, for compare_to_strong_foot."""
    sums = {}
    for rep in metrics["reps"]:
        if rep["foot"] != player["strong_foot"]:
            continue
        for m in rep["metrics"]:
            if m["value"] is not None:
                sums.setdefault(m["name"], []).append(m["value"])
    return {k: sum(v) / len(v) for k, v in sums.items()}


def why_text(targets, move, metric):
    return ((targets or {}).get(move, {}).get(metric) or {}).get("why", "")


def build_cards(metrics, player, drills, targets, writer=None):
    writer = writer or Writer()
    strong_avg = strong_foot_averages(metrics, player)
    out = []
    for rep in metrics["reps"]:
        cards = []
        for m in worst_metrics(rep):
            words = writer.words(m["name"], m["value"], m["unit"], m["target"],
                                 why_text(targets, rep["move"], m["name"]))
            card = {
                "metric": m["name"],
                "headline": headline(m["name"], m["value"], m["target"]),
                "measured_value": m["value"],
                "unit": m["unit"],
                "target": list(m["target"]),
                "why": words["why"],
                "fix": words["fix"],
                "drill_ids": drills_for(m["name"], drills, player["level"]),
                "frame": m["frame"],
            }
            if rep["foot"] != player["strong_foot"] and m["name"] in strong_avg:
                card["compare_to_strong_foot"] = round_like(strong_avg[m["name"]], m["unit"])
            cards.append(card)
        out.append({"rep_id": rep["rep_id"], "cards": cards})
    return {"schema_version": SCHEMA_VERSION, "player_id": metrics.get("player_id") or player["player_id"],
            "reps": out}
