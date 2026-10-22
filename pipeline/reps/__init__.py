"""Workstream C: pose.json + ball.json (+ player.json) -> reps.json."""
from .config import RepsConfig
from .detect import detect_reps

__all__ = ["RepsConfig", "detect_reps"]
