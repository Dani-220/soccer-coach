"""Tunable thresholds for rep detection.

Every threshold lives here. Distances are in body heights (H, see CONTRACT.md), speeds in
body heights per second and durations in seconds, so the same values work at any resolution
and frame rate. Override any subset with a YAML file:  python -m pipeline.reps --config my.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RepsConfig:
    # --- input filtering ---
    min_keypoint_conf: float = 0.2      # ankle/nose points below this are treated as missing
    min_toe_conf: float = 0.5           # foot_index below this: measure foot-ball gap from the ankle alone
    min_ball_conf: float = 0.0          # ball detections below this are treated as missing

    # --- contact: the ball is at rest, then its speed jumps ---
    rest_window_s: float = 0.15         # how far back to look when checking the ball is at rest
    min_rest_samples: int = 3           # visible ball detections needed inside that window
    rest_radius: float = 0.03           # H; all resting detections lie within this of their median
    jump_speed: float = 3.0             # H/s; ball speed after the jump must reach this
    max_ball_gap_s: float = 0.17        # longest ball occlusion bridged around contact (foot hides it);
                                        #    0.17 s bridges 5 hidden frames at 30 fps
    # A foot must touch the resting ball: gaps are measured to the ball's surface (ball.json radius).
    foot_ball_max_gap: float = 0.08     # H; ankle->toe segment to ball surface, when the toe is confident
    ankle_ball_max_gap: float = 0.15    # H; ankle to ball surface, when it isn't (the ball is struck
                                        #    a foot-length ahead of the ankle, so this must be looser)
    touch_window_s: float = 0.02        # either gap is checked within +/- this of the contact frame
    # The detector often loses a struck ball. A ball that vanishes from rest still counts as a contact if:
    vanish_confirm_s: float = 0.3       # ...it doesn't reappear at its resting spot within this,
    vanish_min_foot_speed: float = 3.0  # ...and the touching foot's ankle moves at least this fast (H/s)
    min_rep_gap_s: float = 0.5          # minimum time between two contacts

    # --- rep boundaries ---
    approach_s: float = 1.0             # start = contact minus this (clamped to the player's segment)
    max_pose_gap_s: float = 0.1         # pose dropouts shorter than this don't split a segment

    # --- plant: support ankle stops moving before contact ---
    plant_ref_window_s: float = 0.03    # support ankle is planted from contact - this...
    plant_jitter_window_s: float = 0.1  # ...to contact + this; its reference position and jitter come from there
    plant_still_radius: float = 0.015   # H; ankle counts as planted while within this of the reference,
    plant_jitter_mult: float = 4.0      # ...or within this many jitter sigmas, whichever is larger
    plant_still_radius_max: float = 0.06  # H; cap on the jitter-based radius
    still_max_outliers: int = 1         # consecutive noisy samples tolerated before the support ankle "moved"
    plant_max_lookback_s: float = 0.5   # never search for plant further back than this

    # --- end of follow-through: kicking ankle stops moving after contact ---
    end_still_radius: float = 0.015     # H; ankle counts as stopped while within this...
    end_hold_s: float = 0.05            # ...for at least this long
    max_follow_through_s: float = 0.6   # end is capped at contact plus this

    # --- kicking foot ---
    foot_window_s: float = 0.02         # ankle-ball distance = minimum within +/- this of contact
    foot_speed_window_s: float = 0.05   # ankle speed = displacement over this window ending at contact
    foot_dist_weight: float = 0.5       # weight of "nearest the ball" in the foot decision
    foot_speed_weight: float = 0.5      # weight of "moving fastest" in the foot decision
    strong_foot_prior: float = 0.05     # bias toward player.json's strong foot (breaks near-ties only)

    # --- confidence ---
    missing_phase_penalty: float = 0.7  # confidence multiplier when plant or end could not be found
    vanish_penalty: float = 0.8         # confidence multiplier when the ball was never seen leaving

    # --- output ---
    move: str = "shot_instep"
    default_player_id: str = "unknown"  # used when no player.json is given

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RepsConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls().override(**data)

    def override(self, **values) -> "RepsConfig":
        known = {f.name for f in fields(self)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown reps config keys: {', '.join(unknown)}")
        return replace(self, **values)

    def frames(self, seconds: float, fps: float) -> int:
        """Convert a duration to a whole number of frames (at least 1)."""
        return max(1, int(round(seconds * fps)))
