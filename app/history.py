"""Per-player history in SQLite (out/history.sqlite): profiles, sessions, reps and metrics."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_id  TEXT PRIMARY KEY,
    profile    TEXT NOT NULL,          -- the player.json document
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  INTEGER PRIMARY KEY,
    player_id   TEXT NOT NULL REFERENCES players(player_id),
    name        TEXT NOT NULL UNIQUE,  -- folder name under out/
    out_dir     TEXT NOT NULL,
    video       TEXT,
    recorded_at TEXT NOT NULL,         -- when the video was filmed; the dashboard's time axis
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reps (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    rep_id     TEXT NOT NULL,
    move       TEXT NOT NULL,
    foot       TEXT NOT NULL,
    confidence REAL,
    f_start INTEGER, f_plant INTEGER, f_contact INTEGER, f_end INTEGER,
    PRIMARY KEY (session_id, rep_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    rep_id     TEXT NOT NULL,
    name       TEXT NOT NULL,
    foot       TEXT NOT NULL,
    value      REAL,
    unit       TEXT,
    target_lo  REAL,
    target_hi  REAL,
    status     TEXT NOT NULL,
    frame      INTEGER,
    PRIMARY KEY (session_id, rep_id, name)
);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


class History:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # players
    def save_player(self, profile):
        with self.db:
            self.db.execute("INSERT INTO players VALUES (?, ?, ?) ON CONFLICT(player_id) DO UPDATE "
                            "SET profile = excluded.profile, updated_at = excluded.updated_at",
                            (profile["player_id"], json.dumps(profile), _now()))

    def player(self, player_id):
        row = self.db.execute("SELECT profile FROM players WHERE player_id = ?", (player_id,)).fetchone()
        return json.loads(row["profile"]) if row else None

    def player_ids(self):
        return [r[0] for r in self.db.execute("SELECT player_id FROM players ORDER BY player_id")]

    # sessions
    def add_session(self, player_id, name, out_dir, video, recorded_at, reps_doc, metrics_doc):
        """Store one run's reps and metrics. Re-adding a session name replaces it."""
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE name = ?", (name,))
            cur = self.db.execute(
                "INSERT INTO sessions (player_id, name, out_dir, video, recorded_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (player_id, name, str(out_dir), video, recorded_at, _now()))
            sid = cur.lastrowid
            for r in reps_doc["reps"]:
                fr = r["frames"]
                self.db.execute("INSERT INTO reps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (sid, r["rep_id"], r["move"], r["foot"], r.get("confidence"),
                                 fr["start"], fr.get("plant"), fr["contact"], fr["end"]))
            for r in metrics_doc["reps"]:
                for m in r["metrics"]:
                    lo, hi = m["target"]
                    self.db.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (sid, r["rep_id"], m["name"], r["foot"], m["value"], m["unit"],
                                     lo, hi, m["status"], m["frame"]))
        return sid

    def sessions(self, player_id):
        """The player's sessions, oldest first."""
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM sessions WHERE player_id = ? ORDER BY recorded_at, session_id", (player_id,))]

    def delete_session(self, session_id):
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def reps(self, session_id):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM reps WHERE session_id = ? ORDER BY f_contact", (session_id,))]

    def metric_rows(self, player_id):
        """Every metric of every rep for the player, joined with its session's time."""
        return [dict(r) for r in self.db.execute(
            "SELECT s.session_id, s.name AS session, s.recorded_at, m.* FROM metrics m "
            "JOIN sessions s USING (session_id) WHERE s.player_id = ? "
            "ORDER BY s.recorded_at, s.session_id, m.rep_id", (player_id,))]
