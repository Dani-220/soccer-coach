"""Soccer Coach app (Workstream G).  streamlit run app/streamlit_app.py

Upload a video and set the player profile; the app runs run.py, then shows the reps, the ghost overlay
of each rep vs. the previous one, the coaching cards and a weak-foot dashboard. History is in out/history.sqlite.
"""
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import altair as alt
import jsonschema
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402
from app.history import History  # noqa: E402

PLAYER_SCHEMA = json.loads((core.REPO / "contract" / "schemas" / "player.schema.json").read_text())
EXAMPLES = core.REPO / "contract" / "examples"
# Categorical slots 1 and 2 of the reference palette. Color follows the foot, never strong/weak,
# so a player's charts don't repaint if their strong foot is changed.
FOOT_COLORS = {"left": "#2a78d6", "right": "#eb6834"}
STATUS = {"in_range": "✅ in range", "out_of_range": "⚠️ out of range", "not_measured": "— not measured"}

st.set_page_config(page_title="Soccer Coach", page_icon="⚽", layout="wide")


@st.cache_resource
def history():
    return History(core.DB_PATH)


@st.cache_data
def targets():
    return core.load_targets()


@st.cache_data
def drills():
    return core.load_drills()


h = history()


# --- sidebar: player profile ---

def player_form():
    st.sidebar.header("Player")
    ids = h.player_ids()
    NEW = "+ New player"
    current = st.session_state.get("player_id")
    choice = st.sidebar.selectbox("Profile", [NEW] + ids, index=(ids.index(current) + 1) if current in ids else 0)
    p = h.player(choice) if choice != NEW else {}
    with st.sidebar.form("player"):
        pid = st.text_input("Player ID", p.get("player_id", ""), disabled=bool(p),
                            help="Short ID, e.g. your initials. Can't be changed later.")
        name = st.text_input("Display name", p.get("display_name", ""))
        strong = st.radio("Strong foot", ["right", "left"], horizontal=True,
                          index=["right", "left"].index(p.get("strong_foot", "right")))
        level = st.radio("Level", ["casual", "pro"], horizontal=True,
                         index=["casual", "pro"].index(p.get("level", "casual")))
        height = st.number_input("Height (cm)", 100, 230, int(p.get("height_cm", 175)))
        goals = st.text_area("Goals (one per line)", "\n".join(p.get("goals", [])))
        spw = st.number_input("Sessions per week", 1, 14, int(p.get("sessions_per_week", 3)))
        if st.form_submit_button("Save profile", type="primary"):
            profile = {"schema_version": "1.0", "player_id": (p.get("player_id") or pid).strip(),
                       "strong_foot": strong, "level": level, "height_cm": height,
                       "goals": [g.strip() for g in goals.splitlines() if g.strip()], "sessions_per_week": spw}
            if name.strip():
                profile["display_name"] = name.strip()
            if not profile["player_id"]:
                st.error("Player ID is required.")
            else:
                try:
                    jsonschema.validate(profile, PLAYER_SCHEMA)
                except jsonschema.ValidationError as e:
                    st.error(f"Invalid profile: {e.message}")
                else:
                    h.save_player(profile)
                    st.session_state.player_id = profile["player_id"]
                    st.rerun()
    return h.player(choice) if choice != NEW else None


# --- new session ---

def run_session(player, out_dir, video_name, recorded_at, reuse_pose=False):
    """Run run.py into out_dir, then store the session. Returns True on success."""
    (out_dir / "player.json").write_text(json.dumps(player, indent=2))
    log_box = st.empty()
    lines = []

    def show(line):
        lines.append(line)
        log_box.code("".join(lines[-25:]), language=None)

    with st.status("Running the pipeline…", expanded=True) as status:
        rc, log = core.run_pipeline(out_dir / video_name if video_name else None, out_dir / "player.json",
                                    out_dir, on_line=show, reuse_pose=reuse_pose)
        if rc != 0:
            status.update(label="Pipeline failed", state="error")
            failed = next((l for l in reversed(log.splitlines()) if l.startswith("run.py: FAILED")), None)
            st.error(failed or f"run.py exited with code {rc}")
            with st.expander("Full log"):
                st.code(log, language=None)
            return False
        status.update(label="Pipeline finished", state="complete", expanded=False)
    sid = h.add_session(player["player_id"], out_dir.name, out_dir, video_name, recorded_at,
                        core.load_json(out_dir / "reps.json"), core.load_json(out_dir / "metrics.json"))
    st.session_state.session_id = sid
    return True


def new_session_tab(player):
    st.subheader("Analyse a new video")
    st.caption("Film side-on, the whole body and the ball in frame. One clip can hold several shots.")
    upload = st.file_uploader("Training video", type=["mp4", "mov", "avi", "m4v"])
    filmed = st.date_input("Filmed on", date.today())
    recorded_at = datetime.combine(filmed, datetime.now().time()).isoformat(timespec="seconds")
    if st.button("Analyse", type="primary", disabled=upload is None):
        name = core.session_name(player["player_id"], datetime.now().strftime("%Y%m%d-%H%M%S"), upload.name)
        out_dir = core.OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / upload.name).write_bytes(upload.getbuffer())
        if run_session(player, out_dir, upload.name, recorded_at):
            st.success(f"Done: {name}. Open the Reps tab to see the results.")

    with st.expander("No video handy? Run on the synthetic fixture"):
        st.caption("Uses contract/examples pose and ball data (two shots, right then left) and skips pose extraction.")
        if st.button("Run on fixture"):
            name = core.session_name(player["player_id"], datetime.now().strftime("%Y%m%d-%H%M%S"), "fixture")
            out_dir = core.OUT / name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXAMPLES / "pose.example.json", out_dir / "pose.json")
            shutil.copyfile(EXAMPLES / "ball.example.json", out_dir / "ball.json")
            if run_session(player, out_dir, None, recorded_at, reuse_pose=True):
                st.success(f"Done: {name}. Open the Reps tab to see the results.")


# --- reps ---

def metric_table(rep_metrics, height_cm):
    return pd.DataFrame([{"Metric": core.metric_label(m["name"]),
                          "Measured": core.format_value(m["value"], m["unit"], height_cm),
                          "Target": core.format_target(m["target"], m["unit"]),
                          "Status": STATUS[m["status"]]} for m in rep_metrics])


def not_measured_notes(rep_metrics, move, view):
    groups = core.not_measured(rep_metrics, move, targets(), view)
    labels = lambda names: ", ".join(core.metric_label(n).lower() for n in names)  # noqa: E731
    if groups["side"]:
        st.info(f"Film side-on to measure: {labels(groups['side'])}")
    if groups["front"]:
        st.info(f"Film front-on to measure: {labels(groups['front'])}")
    if groups["other"]:
        hint = " (set your height in the profile)" if "ball_speed" in groups["other"] else ""
        st.caption(f"Couldn't measure on this clip: {labels(groups['other'])}{hint}")


def overlay_panel(sessions, reps_by_session, session, rep_id, strong_foot):
    prev = core.previous_rep(sessions, reps_by_session, session["session_id"], rep_id)
    if prev is None:
        st.caption("This is the player's first rep, so there's nothing to compare it with yet.")
        return
    old_session, old_rep = prev
    old_foot = next(r["foot"] for r in reps_by_session[old_session["session_id"]] if r["rep_id"] == old_rep)
    where = "" if old_session["session_id"] == session["session_id"] else f" from {old_session['name']}"
    st.markdown(f"**Ghost:** {old_rep} · {core.foot_label(old_foot, strong_foot)}{where}")
    with st.spinner("Rendering overlay…"):
        mp4, png, err = core.make_overlay(session["out_dir"], rep_id, old_session["out_dir"], old_rep)
    if err:
        st.error(f"Overlay failed: {err.splitlines()[-1]}")
        return
    web = core.browser_video(mp4)
    if web:
        st.video(str(web), loop=True)
    st.image(str(png), caption="Contact frame. Joints go green to red with distance from the ghost.")
    st.download_button("Download overlay video", mp4.read_bytes(), file_name=mp4.name, mime="video/mp4",
                       key=f"dl_{session['session_id']}_{rep_id}")


def card_view(card, height_cm, drill_lib):
    with st.container(border=True):
        st.markdown(f"#### {card['headline']}")
        st.markdown(f"**Measured** {core.format_value(card['measured_value'], card['unit'], height_cm)} · "
                    f"**Target** {core.format_target(card['target'], card['unit'])}")
        if card.get("compare_to_strong_foot") is not None:
            st.caption(f"Strong foot: {core.format_value(card['compare_to_strong_foot'], card['unit'], height_cm)}")
        st.markdown(f"**Why it matters.** {card['why']}")
        st.markdown(f"**Fix.** {card['fix']}")
        if card.get("drill_ids"):
            st.markdown("**Drills:** " + ", ".join(drill_lib.get(d, {}).get("name", d) for d in card["drill_ids"]))


def reps_tab(player):
    sessions = h.sessions(player["player_id"])
    if not sessions:
        st.info("No sessions yet. Analyse a video in the New session tab.")
        return
    strong, height = player["strong_foot"], player["height_cm"]
    by_id = {s["session_id"]: s for s in sessions}
    default = st.session_state.get("session_id", sessions[-1]["session_id"])
    ids = [s["session_id"] for s in sessions][::-1]
    sid = st.selectbox("Session", ids, index=ids.index(default) if default in ids else 0,
                       format_func=lambda i: f"{by_id[i]['recorded_at'][:10]} · {by_id[i]['name']}")
    session = by_id[sid]
    out_dir = Path(session["out_dir"])
    if not (out_dir / "metrics.json").exists():
        st.error(f"The files for this session are gone from {out_dir}.")
        return
    reps_by_session = {s["session_id"]: h.reps(s["session_id"]) for s in sessions}
    metrics_doc = core.load_json(out_dir / "metrics.json")
    metrics_by_rep = {r["rep_id"]: r for r in metrics_doc["reps"]}
    view = core.session_view(metrics_doc, targets())
    cards, card_source = core.load_cards(out_dir)
    reps = reps_by_session[sid]
    if not reps:
        st.warning("No reps were found in this video.")
        return

    st.dataframe(pd.DataFrame([{
        "Rep": r["rep_id"], "Foot": r["foot"].capitalize(), "Strong / weak": core.foot_tag(r["foot"], strong),
        "Contact frame": r["f_contact"], "Confidence": r["confidence"],
        "Out of range": sum(m["status"] == "out_of_range" for m in metrics_by_rep.get(r["rep_id"], {}).get("metrics", [])),
    } for r in reps]), hide_index=True, width="stretch")

    rep_id = st.radio("Rep", [r["rep_id"] for r in reps], horizontal=True,
                      format_func=lambda i: f"{i} · {core.foot_label(next(r['foot'] for r in reps if r['rep_id'] == i), strong)}")
    rep = next(r for r in reps if r["rep_id"] == rep_id)
    rep_metrics = metrics_by_rep.get(rep_id, {"metrics": []})["metrics"]
    tag = core.foot_tag(rep["foot"], strong)
    st.markdown(f"### {rep_id} · {rep['foot'].capitalize()} foot "
                f"{':green-badge[strong]' if tag == 'strong' else ':orange-badge[weak]'}")

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(metric_table(rep_metrics, height), hide_index=True, width="stretch")
        not_measured_notes(rep_metrics, rep["move"], view)
    with right:
        overlay_panel(sessions, reps_by_session, session, rep_id, strong)

    st.markdown("### Why it matters")
    if card_source == "example":
        st.warning("Example cards: the coaching stage (Workstream F) isn't built yet, so these come from "
                   "contract/examples/cards.example.json and describe the fixture, not this video.")
    rep_cards = cards.get(rep_id)
    if rep_cards is None:
        st.caption("No cards for this rep.")
    elif not rep_cards:
        st.success("Every measured metric is in range. Nothing to fix on this rep.")
    for c in rep_cards or []:
        card_view(c, height, drills())


# --- weak-foot dashboard ---

def trend_chart(df, metric, strong_foot):
    d = df[df["metric"] == metric].copy()
    unit = d["unit"].iloc[0]
    lo, hi = d["target_lo"].iloc[0], d["target_hi"].iloc[0]
    d["date"] = pd.to_datetime(d["recorded_at"])
    labels = [core.foot_label(f, strong_foot) for f in ("left", "right")]
    color = alt.Color("foot_label:N", title="Foot",
                      scale=alt.Scale(domain=labels, range=[FOOT_COLORS["left"], FOOT_COLORS["right"]]))
    band = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(color="#8a8984", opacity=0.15).encode(
        y="lo:Q", y2="hi:Q")
    base = alt.Chart(d).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=core.UNIT_LABELS.get(unit, unit), scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip("session:N"), alt.Tooltip("foot_label:N", title="Foot"),
                 alt.Tooltip("value:Q", title="Mean", format=".3g"), alt.Tooltip("reps:Q"),
                 alt.Tooltip("in_range:Q", title="Reps in range")])
    lines = base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=70, strokeWidth=2, stroke="white")
    return (band + lines).properties(title=f"{core.metric_label(metric)} (target {core.format_target([lo, hi], unit)})",
                                     height=220)


def dashboard_tab(player):
    strong = player["strong_foot"]
    weak = "left" if strong == "right" else "right"
    rows = h.metric_rows(player["player_id"])
    trend = core.foot_trend(rows, strong)
    if not trend:
        st.info("No measured metrics yet. Analyse a video first.")
        return
    df = pd.DataFrame(trend)
    st.caption(f"Each point is one session's mean over its reps with that foot. Grey band = target range. "
               f"Strong foot: {strong}.")

    summary = []
    for metric in dict.fromkeys(df["metric"]):
        d = df[df["metric"] == metric]
        row = {"Metric": core.metric_label(metric),
               "Target": core.format_target([d["target_lo"].iloc[0], d["target_hi"].iloc[0]], d["unit"].iloc[0])}
        for foot, label in ((strong, "Strong"), (weak, "Weak")):
            f = d[d["foot"] == foot]
            row[f"{label} latest"] = (core.format_value(f["value"].iloc[-1], f["unit"].iloc[-1], None)
                                      if len(f) else "–")
            row[f"{label} in range"] = f"{f['in_range'].sum()}/{f['reps'].sum()}" if len(f) else "–"
        summary.append(row)
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")

    metrics = list(dict.fromkeys(df["metric"]))
    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        cols[i % 2].altair_chart(trend_chart(df, metric, strong), width="stretch")

    never = sorted({m["name"] for m in rows} - set(metrics))
    if never:
        st.caption("Never measured so far: " + ", ".join(core.metric_label(n).lower() for n in never))
    with st.expander("Table view"):
        st.dataframe(df[["recorded_at", "session", "metric", "foot_label", "value", "unit", "reps", "in_range"]],
                     hide_index=True, width="stretch")


# --- page ---

st.title("⚽ Soccer Coach")
player = player_form()
if player is None:
    st.info("Create a player profile in the sidebar to start.")
    st.stop()
st.session_state.player_id = player["player_id"]
st.caption(f"{player.get('display_name', player['player_id'])} · strong foot {player['strong_foot']} · "
           f"{player['level']} · {player['height_cm']} cm")

tab_new, tab_reps, tab_dash = st.tabs(["New session", "Reps", "Weak-foot dashboard"])
with tab_new:
    new_session_tab(player)
with tab_reps:
    reps_tab(player)
with tab_dash:
    dashboard_tab(player)
