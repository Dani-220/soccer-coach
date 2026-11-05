"""Words for a card: headline, why and fix.

The headline always comes from a fixed table. `why` and `fix` come from the Claude API when an
ANTHROPIC_API_KEY is set (environment or .env), else from a fixed template. The API only ever sees the
metric name, measured value, unit, target and the metric's "why" text from targets.yaml, and its reply
is rejected (falling back to the template) if it contains any digit: numbers on a card are copied from
metrics.json by code, never written by the LLM.
"""
import json
import os
import re
import sys
from pathlib import Path

MODEL = os.environ.get("COACH_MODEL", "claude-opus-5-5")
WHY_MAX, FIX_MAX, HEADLINE_MAX = 220, 160, 60

UNIT_LABELS = {"body_heights": "body heights", "degrees": "degrees", "m_per_s": "m/s", "ratio": "", "count": ""}

# (metric, direction) -> (headline, fix). direction is "low" (below target) or "high" (above target).
TEMPLATES = {
    ("plant_foot_distance", "high"): ("Plant foot too far from the ball",
                                      "Land your plant foot level with the ball and a hand's width to its side."),
    ("plant_foot_distance", "low"): ("Plant foot too close to the ball",
                                     "Give the kicking leg room: plant a hand's width beside the ball, not against it."),
    ("plant_foot_angle", "high"): ("Plant foot not pointing at the target",
                                   "Point your plant toe straight at where you want the ball to go."),
    ("plant_foot_angle", "low"): ("Plant foot not pointing at the target",
                                  "Point your plant toe straight at where you want the ball to go."),
    ("knee_over_ball", "low"): ("Knee behind the ball at contact",
                                "Get your kicking knee over the ball as you strike it."),
    ("knee_over_ball", "high"): ("Knee too far past the ball",
                                 "Strike a touch earlier so the knee is over the ball, not well past it."),
    ("trunk_lean", "low"): ("Leaning back at contact",
                            "Keep your chest over the ball and land on your kicking foot after the strike."),
    ("trunk_lean", "high"): ("Leaning too far forward at contact",
                             "Stay a little taller through the strike; chest over the ball, not past it."),
    ("hip_shoulder_separation", "low"): ("Not enough hip-shoulder coil",
                                         "Swing your opposite arm back and let your hips turn before your shoulders."),
    ("hip_shoulder_separation", "high"): ("Over-rotating before the strike",
                                          "Turn your hips a little less early and keep the shoulders closer to square."),
    ("ankle_lock", "low"): ("Ankle not locked at contact",
                            "Point your toe down and hold the ankle firm through the strike."),
    ("ankle_lock", "high"): ("Ankle over-extended at contact",
                             "Keep the toe pointed but relaxed enough to strike with the laces, not the toe."),
    ("follow_through", "low"): ("Follow-through too short",
                                "Keep swinging after contact and land a step past the ball."),
    ("follow_through", "high"): ("Follow-through too long",
                                 "Finish toward the target and land balanced instead of swinging on."),
    ("ball_speed", "low"): ("Shot speed below target",
                            "Fix the technique cards first, then add a faster last two steps."),
    ("ball_speed", "high"): ("Shot speed above target",
                             "Great power; keep the same technique and aim for accuracy."),
}
GENERIC = ("{label} out of range", "Work on this with the drills below, slowly first, then at full speed.")

SYSTEM_PROMPT = (
    "You are a football (soccer) shooting coach giving feedback on one measured part of a player's shot. "
    "Write two plain-English pieces in the second person:\n"
    f"- why: one or two sentences, at most {WHY_MAX - 20} characters, saying what the measurement means "
    "for the shot and why it matters.\n"
    f"- fix: one concrete cue the player can apply on the next shot, at most {FIX_MAX - 20} characters.\n"
    "Never write digits or numbers: the app shows the measured value and target beside your text. "
    "No greetings, no hedging, no markdown."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"why": {"type": "string"}, "fix": {"type": "string"}},
    "required": ["why", "fix"],
    "additionalProperties": False,
}


def direction(value, target):
    return "low" if value < target[0] else "high"


def fmt(x):
    return f"{x:g}"


def headline(metric, value, target):
    h = TEMPLATES.get((metric, direction(value, target)), GENERIC)[0]
    return h.format(label=metric.replace("_", " ").capitalize())[:HEADLINE_MAX]


def template_words(metric, value, unit, target, why_text):
    """Deterministic why/fix. Numbers here are formatted from the copied metrics.json values."""
    unit_label = UNIT_LABELS.get(unit, unit)
    measured = f"Measured {fmt(value)} {unit_label} vs target {fmt(target[0])} to {fmt(target[1])}.".replace("  ", " ")
    why = f"{why_text} {measured}".strip() if why_text else measured
    if len(why) > WHY_MAX:
        why = why_text[:WHY_MAX]
    fix = TEMPLATES.get((metric, direction(value, target)), GENERIC)[1]
    return {"why": why, "fix": fix[:FIX_MAX]}


def load_api_key(env_file=None):
    """ANTHROPIC_API_KEY from the environment, else from a .env file (default: the repo root's)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    path = Path(env_file) if env_file else Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def valid_llm_words(words):
    """True when the LLM reply is usable as-is: both fields present, short enough, and number-free."""
    if not isinstance(words, dict):
        return False
    why, fix = words.get("why"), words.get("fix")
    if not (isinstance(why, str) and isinstance(fix, str) and why.strip() and fix.strip()):
        return False
    if len(why) > WHY_MAX or len(fix) > FIX_MAX:
        return False
    return not re.search(r"\d", why + fix)


class Writer:
    """Produces why/fix for each card, via Claude when a key is available, else the template."""

    def __init__(self, api_key=None, client=None, model=MODEL):
        self.model = model
        self.client = client
        if self.client is None and api_key:
            try:
                import anthropic
            except ImportError:
                print("coach: anthropic is not installed; using template wording", file=sys.stderr)
            else:
                self.client = anthropic.Anthropic(api_key=api_key, timeout=60.0, max_retries=2)
        self.llm_cards = 0

    def words(self, metric, value, unit, target, why_text):
        if self.client is not None:
            got = self._ask_llm(metric, value, unit, target, why_text)
            if got is not None:
                self.llm_cards += 1
                return got
        return template_words(metric, value, unit, target, why_text)

    def _ask_llm(self, metric, value, unit, target, why_text):
        facts = {"metric": metric, "measured_value": value, "unit": unit, "target": target,
                 "why_it_matters": why_text}
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(facts)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if resp.stop_reason != "end_turn":
                raise ValueError(f"stop_reason {resp.stop_reason}")
            text = next(b.text for b in resp.content if b.type == "text")
            words = json.loads(text)
        except Exception as e:  # any API, network or parse failure falls back to the template
            print(f"coach: Claude wording failed for {metric} ({type(e).__name__}: {e}); using template",
                  file=sys.stderr)
            return None
        if not valid_llm_words(words):
            print(f"coach: Claude wording for {metric} rejected (too long or contains numbers); using template",
                  file=sys.stderr)
            return None
        return {"why": words["why"].strip(), "fix": words["fix"].strip()}
