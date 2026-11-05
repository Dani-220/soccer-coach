"""Cards, plan, drill library, wording and CLI. Runs without an API key: the LLM path uses a fake client."""
import copy
import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.coach.__main__ import main
from pipeline.coach.cards import build_cards, load_drills, load_targets, worst_metrics
from pipeline.coach.plan import build_plan, uses_ball, weak_foot_share
from pipeline.coach.wording import (FIX_MAX, HEADLINE_MAX, TEMPLATES, WHY_MAX, Writer, headline, load_api_key,
                                    template_words)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "contract" / "examples"
METRIC_NAMES = json.loads((ROOT / "contract" / "schemas" / "cards.schema.json").read_text())[
    "properties"]["reps"]["items"]["properties"]["cards"]["items"]["properties"]["metric"]["enum"]
WEEK = dt.date(2026, 11, 9)


def load(name):
    return json.loads((EXAMPLES / name).read_text())


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(scope="module")
def validator():
    spec = importlib.util.spec_from_file_location("contract_validate", ROOT / "contract" / "tools" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drills():
    return load_drills()


@pytest.fixture
def metrics():
    return load("metrics.example.json")


@pytest.fixture
def player():
    return load("player.example.json")


def check_contract(validator, metrics, cards=None, plan=None):
    errs = []
    if cards is not None:
        errs += validator.schema_errors("cards", cards)
    if plan is not None:
        errs += validator.schema_errors("plan", plan)
    errs += validator.cross_checks({"metrics": metrics, **({"cards": cards} if cards else {})})
    assert errs == []


def behind_camera(metrics):
    """Every metric not_measured, as on footage shot from behind the player."""
    m = copy.deepcopy(metrics)
    for rep in m["reps"]:
        for x in rep["metrics"]:
            x["value"], x["status"] = None, "not_measured"
    return m


# --- cards -----------------------------------------------------------------------------------------

def test_fixture_cards_match_contract(validator, metrics, player, drills):
    cards = build_cards(metrics, player, drills, load_targets())
    check_contract(validator, metrics, cards=cards)
    by_rep = {r["rep_id"]: r["cards"] for r in cards["reps"]}
    assert by_rep["r001"] == []  # the good reference rep
    assert [c["metric"] for c in by_rep["r002"]] == ["plant_foot_distance", "trunk_lean"]
    plant = by_rep["r002"][0]
    assert plant["measured_value"] == 0.239 and plant["target"] == [0.04, 0.16] and plant["frame"] == 324
    assert plant["compare_to_strong_foot"] == 0.132
    assert plant["headline"] == "Plant foot too far from the ball"


def test_values_are_copied_exactly(validator, metrics, player, drills):
    for x in metrics["reps"][1]["metrics"]:
        if x["value"] is not None:
            x["value"] = round(x["value"] * 1.0371, 3)
            lo, hi = x["target"]
            x["status"] = "in_range" if lo <= x["value"] <= hi else "out_of_range"
    cards = build_cards(metrics, player, drills, load_targets())
    check_contract(validator, metrics, cards=cards)


def test_behind_camera_gives_zero_cards(validator, metrics, player, drills):
    m = behind_camera(metrics)
    cards = build_cards(m, player, drills, load_targets())
    assert all(r["cards"] == [] for r in cards["reps"])
    assert [r["rep_id"] for r in cards["reps"]] == ["r001", "r002"]
    check_contract(validator, m, cards=cards)


def test_at_most_two_cards_and_outcome_last(metrics):
    rep = metrics["reps"][1]
    assert len(worst_metrics(rep)) == 2
    only_speed = {"metrics": [x for x in rep["metrics"] if x["name"] in ("ball_speed", "plant_foot_angle")]}
    assert [x["name"] for x in worst_metrics(only_speed)] == ["ball_speed"]
    one_technique = {"metrics": [x for x in rep["metrics"] if x["name"] in ("ball_speed", "follow_through")]}
    assert [x["name"] for x in worst_metrics(one_technique)] == ["follow_through", "ball_speed"]


def test_no_compare_without_strong_foot_reps(metrics, player, drills):
    metrics["reps"] = [metrics["reps"][1]]
    cards = build_cards(metrics, player, drills, load_targets())
    assert all("compare_to_strong_foot" not in c for c in cards["reps"][0]["cards"])


# --- drill library -----------------------------------------------------------------------------------

def test_drill_library(drills):
    assert len(drills) >= 30
    assert len({d["id"] for d in drills}) == len(drills)
    for d in drills:
        assert {"id", "name", "fixes", "levels", "equipment", "setup", "coaching_points",
                "default_sets", "default_reps", "minutes"} <= d.keys(), d["id"]
        assert d["fixes"] and set(d["fixes"]) <= set(METRIC_NAMES), d["id"]
        assert d["levels"] and set(d["levels"]) <= {"casual", "pro"}, d["id"]
        assert d["default_sets"] >= 1 and d["default_reps"] >= 1 and d["minutes"] > 0, d["id"]
    for metric in METRIC_NAMES:
        for level in ("casual", "pro"):
            assert any(metric in d["fixes"] and level in d["levels"] for d in drills), (metric, level)


def test_card_drills_exist_and_fit_level(metrics, player, drills):
    by_id = {d["id"]: d for d in drills}
    for level in ("casual", "pro"):
        cards = build_cards(metrics, {**player, "level": level}, drills, load_targets())
        for c in cards["reps"][1]["cards"]:
            for i in c["drill_ids"]:
                assert c["metric"] in by_id[i]["fixes"] and level in by_id[i]["levels"]


# --- plan --------------------------------------------------------------------------------------------

def test_fixture_plan(validator, metrics, player, drills):
    plan = build_plan(metrics, player, drills, week_start=WEEK)
    check_contract(validator, metrics, plan=plan)
    assert plan["focus"][:2] == ["plant_foot_distance", "trunk_lean"]
    assert plan["week_start"] == "2026-11-09" and plan["retest_on"] == "2026-11-16"
    assert [d["day"] for d in plan["days"]] == ["mon", "wed", "fri"]  # sessions_per_week: 3
    ids = {d["id"] for d in drills}
    assert all(e["drill_id"] in ids for day in plan["days"] for e in day["drills"])


def ball_reps_share(plan, drills, foot):
    by_id = {d["id"]: d for d in drills}
    reps = [(e["foot"], e["sets"] * e["reps"]) for day in plan["days"] for e in day["drills"]
            if e["foot"] != "both" and uses_ball(by_id[e["drill_id"]])]
    return sum(n for f, n in reps if f == foot) / sum(n for _, n in reps)


def test_weak_foot_weighted_when_worse(metrics, player, drills):
    # Fixture: strong (right) rep is clean, weak (left) rep is all out of range.
    assert weak_foot_share(metrics, player) == 0.8
    plan = build_plan(metrics, player, drills, week_start=WEEK)
    assert ball_reps_share(plan, drills, "left") > 0.6


def test_even_split_when_weak_foot_is_not_worse(metrics, player, drills):
    lefty = {**player, "strong_foot": "left"}  # now the clean right-foot rep is the weak foot
    assert weak_foot_share(metrics, lefty) == 0.5
    assert weak_foot_share(behind_camera(metrics), player) == 0.5


def test_plan_behind_camera(validator, metrics, player, drills):
    m = behind_camera(metrics)
    plan = build_plan(m, player, drills, week_start=WEEK)
    check_contract(validator, m, plan=plan)
    assert plan["focus"] == ["ball_speed"] and plan["weak_foot_share"] == 0.5


def test_plan_in_range_focus_is_closest_to_edge(metrics, player, drills):
    metrics["reps"] = [metrics["reps"][0]]  # only the clean rep
    plan = build_plan(metrics, player, drills, week_start=WEEK)
    # 0.364 in [0.30, 1.20] is 7% of the range from its edge; knee_over_ball 0.086 in [-0.04, 0.10] is 10%.
    assert plan["focus"] == ["follow_through"]


@pytest.mark.parametrize("sessions", [1, 2, 5, 7])
def test_sessions_per_week(validator, metrics, player, drills, sessions):
    plan = build_plan(metrics, {**player, "sessions_per_week": sessions, "level": "pro"}, drills, week_start=WEEK)
    check_contract(validator, metrics, plan=plan)
    assert len(plan["days"]) == sessions
    assert all(day["drills"] for day in plan["days"])


# --- wording -----------------------------------------------------------------------------------------

def test_templates_fit_schema_limits():
    long_why = "x" * 150
    for (metric, direction) in TEMPLATES:
        value, target = (-1.0, [0.0, 1.0]) if direction == "low" else (2.0, [0.0, 1.0])
        assert len(headline(metric, value, target)) <= HEADLINE_MAX
        w = template_words(metric, value, "body_heights", target, long_why)
        assert len(w["why"]) <= WHY_MAX and len(w["fix"]) <= FIX_MAX


def test_no_key_uses_template(tmp_path):
    assert load_api_key(tmp_path / "missing.env") is None
    assert Writer(api_key=load_api_key(tmp_path / "missing.env")).client is None


def test_api_key_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOTHER=1\nexport ANTHROPIC_API_KEY='sk-test'\n")
    assert load_api_key(env) == "sk-test"


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.calls = []
        self.reply, self.error = reply, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        block = SimpleNamespace(type="text", text=json.dumps(self.reply))
        return SimpleNamespace(stop_reason="end_turn", content=[block])


GOOD = {"why": "You reached for the ball, so your body tipped back and the shot rose.",
        "fix": "Plant beside the ball, not behind it."}


def test_llm_words_used_and_only_facts_sent(validator, metrics, player, drills):
    fake = FakeClient(reply=GOOD)
    writer = Writer(client=fake)
    cards = build_cards(metrics, player, drills, load_targets(), writer=writer)
    check_contract(validator, metrics, cards=cards)
    c = cards["reps"][1]["cards"][0]
    assert (c["why"], c["fix"]) == (GOOD["why"], GOOD["fix"])
    assert c["measured_value"] == 0.239 and writer.llm_cards == 2
    sent = json.loads(fake.calls[0]["messages"][0]["content"])
    assert set(sent) == {"metric", "measured_value", "unit", "target", "why_it_matters"}
    assert sent["measured_value"] == 0.239 and sent["why_it_matters"].startswith("Planting too far")


@pytest.mark.parametrize("fake", [
    FakeClient(reply={"why": "You planted 39 cm from the ball.", "fix": "Plant closer."}),  # numbers
    FakeClient(reply={"why": "y" * (WHY_MAX + 1), "fix": "Plant closer."}),  # too long
    FakeClient(reply={"why": "", "fix": "Plant closer."}),  # empty
    FakeClient(error=RuntimeError("network down")),
])
def test_bad_llm_reply_falls_back_to_template(metrics, player, drills, fake):
    writer = Writer(client=fake)
    cards = build_cards(metrics, player, drills, load_targets(), writer=writer)
    c = cards["reps"][1]["cards"][0]
    assert c["fix"] == TEMPLATES[("plant_foot_distance", "high")][1]
    assert writer.llm_cards == 0


# --- CLI ---------------------------------------------------------------------------------------------

def test_cli_end_to_end(validator, tmp_path):
    out = tmp_path / "out" / "cards.json"
    main(["--in", str(EXAMPLES / "player.example.json"), str(EXAMPLES / "metrics.example.json"),
          "--out", str(out), "--week-start", "2026-11-09", "--no-llm"])
    cards = json.loads(out.read_text())
    plan = json.loads((out.parent / "plan.json").read_text())
    check_contract(validator, load("metrics.example.json"), cards=cards, plan=plan)
    assert plan["week_start"] == "2026-11-09"
