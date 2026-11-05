"""Workstream F CLI: python -m pipeline.coach --in metrics.json player.json --out cards.json [--plan plan.json]

Input files can be given in either order; each one's type is recognised from its contents.
plan.json goes next to cards.json unless --plan says otherwise. Card wording uses the Claude API when
ANTHROPIC_API_KEY is set (environment or .env at the repo root), else a fixed template.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .cards import DEFAULT_DRILLS, build_cards, load_drills, load_targets
from .plan import build_plan
from .wording import Writer, load_api_key


def kind_of(doc):
    if "strong_foot" in doc:
        return "player"
    if "reps" in doc and "video" in doc:
        return "metrics"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.coach", description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inputs", nargs=2, required=True, help="metrics.json player.json")
    ap.add_argument("--out", required=True, help="cards.json to write")
    ap.add_argument("--plan", help="plan.json to write (default: plan.json next to --out)")
    ap.add_argument("--drills", default=str(DEFAULT_DRILLS), help="drill library YAML (default: %(default)s)")
    ap.add_argument("--targets", help="targets YAML for the 'why' text (default: pipeline/metrics/targets.yaml)")
    ap.add_argument("--week-start", type=dt.date.fromisoformat, help="plan start date (default: next Monday)")
    ap.add_argument("--no-llm", action="store_true", help="always use template wording")
    a = ap.parse_args(argv)

    docs = {}
    for p in a.inputs:
        doc = json.loads(Path(p).read_text())
        kind = kind_of(doc)
        if kind is None or kind in docs:
            ap.error(f"can't use {p}: expected one metrics.json and one player.json")
        docs[kind] = doc
    metrics, player = docs["metrics"], docs["player"]

    drills = load_drills(a.drills)
    writer = Writer(api_key=None if a.no_llm else load_api_key())
    cards = build_cards(metrics, player, drills, load_targets(a.targets), writer=writer)
    plan = build_plan(metrics, player, drills, week_start=a.week_start)

    out = Path(a.out)
    plan_path = Path(a.plan) if a.plan else out.with_name("plan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=2))
    plan_path.write_text(json.dumps(plan, indent=2))
    n = sum(len(r["cards"]) for r in cards["reps"])
    source = f"Claude for {writer.llm_cards}" if writer.llm_cards else "template"
    print(f"coach: wrote {n} cards for {len(cards['reps'])} reps to {out} (wording: {source}); "
          f"plan with {len(plan['days'])} days to {plan_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
