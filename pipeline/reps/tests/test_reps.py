"""Run from the repo root:  python -m unittest pipeline.reps.tests.test_reps -v"""
import importlib.util
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline.reps import RepsConfig, detect_reps
from pipeline.reps.__main__ import main as cli_main
from pipeline.reps.detect import body_height, load_tracks

from .synthetic import BODY_H, RepSpec, Scenario, generate

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "contract" / "examples"

# Load the contract's validator without writing a __pycache__ into contract/.
_spec = importlib.util.spec_from_file_location("contract_validate", ROOT / "contract" / "tools" / "validate.py")
validate = importlib.util.module_from_spec(_spec)
_dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
_spec.loader.exec_module(validate)
sys.dont_write_bytecode = _dont_write

PHASE_TOL = 3   # frames; plant/end are judged on noisy ankles (3 frames = 25 ms at 120 fps)
# The synthetic kicking ankle decelerates smoothly to zero at `end`, so its last ~60 ms move less
# than `end_still_radius` and the detector reports the end 30-80 ms early. Allowed in seconds.
SYNTH_END_TOL_S = 0.09


def load(name):
    return json.loads((EXAMPLES / name).read_text())


class ContractMixin:
    def assertValid(self, reps, pose):
        errs = validate.schema_errors("reps", reps) + validate.cross_checks({"reps": reps, "pose": pose})
        self.assertEqual(errs, [])

    def assertMatches(self, rep, truth, contact_tol=0, phase_tol=PHASE_TOL, end_tol=PHASE_TOL):
        fr = rep["frames"]
        self.assertEqual(rep["foot"], truth["foot"])
        self.assertLessEqual(abs(fr["contact"] - truth["contact"]), contact_tol, fr)
        self.assertIn("plant", fr)
        self.assertLessEqual(abs(fr["plant"] - truth["plant"]), phase_tol, fr)
        self.assertLessEqual(abs(fr["end"] - truth["end"]), end_tol, fr)
        self.assertEqual(fr["start"], truth["start"])


class TestExampleFixtures(ContractMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pose, cls.ball, cls.player = load("pose.example.json"), load("ball.example.json"), load("player.example.json")
        cls.truth = load("reps.example.json")["reps"]
        cls.out = detect_reps(cls.pose, cls.ball, cls.player)

    def test_valid_against_contract(self):
        self.assertValid(self.out, self.pose)
        self.assertEqual(self.out["video"], self.pose["video"])
        self.assertEqual(self.out["player_id"], "p1")

    def test_finds_both_reps_in_order(self):
        self.assertEqual([r["rep_id"] for r in self.out["reps"]], ["r001", "r002"])
        self.assertTrue(all(r["move"] == "shot_instep" for r in self.out["reps"]))

    def test_matches_ground_truth(self):
        for rep, truth in zip(self.out["reps"], self.truth):
            with self.subTest(rep=truth["rep_id"]):
                self.assertMatches(rep, {"foot": truth["foot"], **truth["frames"]})

    def test_occluded_contact_r002(self):
        # ball is null at frames 330-332; contact must still be 331, back-projected from flight
        self.assertIsNone(self.ball["frames"][331]["ball"])
        self.assertEqual(self.out["reps"][1]["frames"]["contact"], 331)

    def test_ball_lost_in_flight(self):
        # Regression: the detector loses the struck ball; it's never seen leaving its resting spot.
        ball = copy.deepcopy(self.ball)
        for fr in ball["frames"]:
            if 103 < fr["f"] < 228 or fr["f"] > 329:
                fr["ball"] = None
        out = detect_reps(self.pose, ball, self.player)
        json.dumps(out)   # frame indices must be plain ints
        self.assertValid(out, self.pose)
        self.assertEqual(len(out["reps"]), 2)
        for rep, normal, truth in zip(out["reps"], self.out["reps"], self.truth):
            self.assertMatches(rep, {"foot": truth["foot"], **truth["frames"]}, contact_tol=1)
            self.assertLess(rep["confidence"], normal["confidence"])

    def test_confident(self):
        for rep in self.out["reps"]:
            self.assertGreater(rep["confidence"], 0.75)

    def test_without_player_json(self):
        out = detect_reps(self.pose, self.ball)
        self.assertEqual(out["player_id"], "unknown")
        self.assertEqual([r["foot"] for r in out["reps"]], ["right", "left"])

    def test_body_height(self):
        H = body_height(load_tracks(self.pose, self.ball, RepsConfig()))
        self.assertAlmostEqual(H, 577, delta=5)


class TestSynthetic(ContractMixin, unittest.TestCase):
    def assertMatches(self, rep, truth, fps=120, **kw):
        kw.setdefault("end_tol", round(SYNTH_END_TOL_S * fps))
        super().assertMatches(rep, truth, **kw)

    def run_scenario(self, sc, cfg=None):
        pose, ball, truths = generate(sc)
        out = detect_reps(pose, ball, None, cfg)
        self.assertValid(out, pose)
        return out["reps"], truths

    def test_single_rep_each_foot(self):
        for foot in ("left", "right"):
            with self.subTest(foot=foot):
                reps, truths = self.run_scenario(Scenario(reps=[RepSpec(foot=foot)]))
                self.assertEqual(len(reps), 1)
                self.assertMatches(reps[0], truths[0])

    def test_direction_independent(self):
        reps, truths = self.run_scenario(Scenario(reps=[RepSpec(foot="left"), RepSpec(foot="right")], direction=-1))
        self.assertEqual(len(reps), 2)
        for rep, truth in zip(reps, truths):
            self.assertMatches(rep, truth)

    def test_noisy_multi_rep(self):
        sc = Scenario(reps=[RepSpec("right"), RepSpec("left"), RepSpec("left")], noise_px=1.8, seed=3)
        reps, truths = self.run_scenario(sc)
        self.assertEqual([r["rep_id"] for r in reps], ["r001", "r002", "r003"])
        for rep, truth in zip(reps, truths):
            self.assertMatches(rep, truth)

    def test_low_frame_rate(self):
        reps, truths = self.run_scenario(Scenario(reps=[RepSpec("left")], fps=30, noise_px=1.0, seed=1))
        self.assertEqual(len(reps), 1)
        self.assertMatches(reps[0], truths[0], fps=30, phase_tol=1)

    def test_noise_robustness_sweep(self):
        for seed in range(20):
            for fps in (30, 60, 120):
                with self.subTest(seed=seed, fps=fps):
                    sc = Scenario(reps=[RepSpec("right"), RepSpec("left", hide_ball_at_contact=1)],
                                  fps=fps, noise_px=1.8, seed=seed, direction=1 if seed % 2 else -1)
                    reps, truths = self.run_scenario(sc)
                    self.assertEqual(len(reps), 2)
                    for rep, truth in zip(reps, truths):
                        self.assertMatches(rep, truth, fps=fps, contact_tol=1, phase_tol=max(1, round(0.025 * fps)))

    def test_ball_hidden_by_foot(self):
        reps, truths = self.run_scenario(Scenario(reps=[RepSpec("right", hide_ball_at_contact=2)], noise_px=1.0))
        self.assertEqual(len(reps), 1)
        self.assertMatches(reps[0], truths[0], contact_tol=1)

    def test_ball_struck_a_foot_length_ahead_of_the_ankle(self):
        # Regression: on real footage the ankle is 0.16-0.18 H from the ball's center at contact,
        # and an ankle-to-center rule of 0.08 H found no shots at all.
        for toe_conf, why in ((0.9, "toe segment"), (0.3, "ankle-only fallback")):
            with self.subTest(why):
                sc = Scenario(reps=[RepSpec("right"), RepSpec("left")], noise_px=1.8, toe_conf=toe_conf)
                pose, ball, truths = generate(sc)
                for t in truths:
                    ank = pose["frames"][t["contact"]]["keypoints"][f"{t['foot']}_ankle"]
                    ctr = ball["frames"][t["contact"]]["ball"]
                    self.assertGreater(math.dist(ank[:2], ctr[:2]) / BODY_H, 0.15)
                reps, truths = self.run_scenario(sc)
                self.assertEqual(len(reps), 2)
                for rep, truth in zip(reps, truths):
                    self.assertMatches(rep, truth)

    def test_ball_lost_in_flight_synthetic(self):
        for fps in (30, 120):
            for toe_conf in (0.9, 0.3):
                with self.subTest(fps=fps, toe_conf=toe_conf):
                    sc = Scenario(reps=[RepSpec("right", lose_ball_in_flight=True),
                                        RepSpec("left", lose_ball_in_flight=True, hide_ball_at_contact=1)],
                                  fps=fps, noise_px=1.8, toe_conf=toe_conf, seed=fps)
                    reps, truths = self.run_scenario(sc)
                    self.assertEqual(len(reps), 2)
                    for rep, truth in zip(reps, truths):
                        self.assertMatches(rep, truth, fps=fps, contact_tol=1, phase_tol=max(1, round(0.025 * fps)))

    def test_ball_briefly_lost_at_rest_is_not_a_shot(self):
        # the foot swings through a ball that never moves; the detector drops it for 17 frames around then
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(ball_speed_hs=0, hide_ball_at_contact=8)]))
        self.assertEqual(reps, [])

    def test_ball_lost_while_player_stands_still_beside_it(self):
        pose, ball, truths = generate(Scenario())
        still = pose["frames"][truths[0]["end"] + 5]["keypoints"]   # support toe touching the ball
        for fr in pose["frames"]:
            fr["keypoints"] = copy.deepcopy(still)
        for fr in ball["frames"]:
            fr["ball"] = ball["frames"][0]["ball"] if fr["f"] < 60 else None
        self.assertEqual(detect_reps(pose, ball)["reps"], [])

    def test_plant_with_real_pose_jitter(self):
        # Regression: on a player ~200 px tall, 0.015 H is ~3 px, below pose jitter, and plant came
        # out equal to contact. The radius now widens to the jitter measured on the planted foot.
        for fps in (30, 60):
            for seed in range(10):
                with self.subTest(fps=fps, seed=seed):
                    sc = Scenario(reps=[RepSpec("right")], fps=fps, noise_px=BODY_H / 200 * 2.5, seed=seed)
                    reps, truths = self.run_scenario(sc)
                    fr = reps[0]["frames"]
                    self.assertIn("plant", fr)
                    self.assertLess(fr["plant"], fr["contact"])
                    self.assertLessEqual(abs(fr["plant"] - truths[0]["plant"]), round(0.05 * fps))

    def test_ball_moves_without_a_foot_near_it(self):
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(player_offset=-1.0)]))
        self.assertEqual(reps, [])

    def test_slow_roll_is_not_a_shot(self):
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(ball_speed_hs=1.5)]))
        self.assertEqual(reps, [])

    def test_thresholds_are_configurable(self):
        sc = Scenario(reps=[RepSpec(ball_speed_hs=1.5)])
        reps, truths = self.run_scenario(sc, RepsConfig(jump_speed=1.0))
        self.assertEqual(len(reps), 1)
        # the player stands ~1 H from the ball: rejected by default, accepted with loose touch thresholds
        loose = RepsConfig(foot_ball_max_gap=2.0, ankle_ball_max_gap=2.0)
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(player_offset=-1.0)]), loose)
        self.assertEqual(len(reps), 1)
        # and a touch threshold tighter than the real strike gap rejects a genuine shot
        reps, _ = self.run_scenario(Scenario(toe_conf=0.3), RepsConfig(ankle_ball_max_gap=0.02))
        self.assertEqual(reps, [])

    def test_strong_foot_only_breaks_ties(self):
        pose, ball, truths = generate(Scenario(reps=[RepSpec("left")]))
        out = detect_reps(pose, ball, {"player_id": "p9", "strong_foot": "right"})
        self.assertEqual(out["reps"][0]["foot"], "left")
        self.assertEqual(out["player_id"], "p9")

    def test_missing_plant_lowers_confidence(self):
        pose, ball, truths = generate(Scenario(reps=[RepSpec("right")]))
        base = detect_reps(pose, ball)["reps"][0]["confidence"]
        for fr in pose["frames"]:                     # support (left) ankle undetectable
            if fr["keypoints"]:
                fr["keypoints"]["left_ankle"][2] = 0.05
        rep = detect_reps(pose, ball)["reps"][0]
        self.assertNotIn("plant", rep["frames"])
        self.assertLess(rep["confidence"], base)
        self.assertValid({"schema_version": "1.0", "video": "v", "player_id": "p", "reps": [rep]}, pose)


class TestConfigAndCli(unittest.TestCase):
    def test_yaml_override_and_unknown_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.yaml"
            p.write_text("jump_speed: 5.5\nfoot_ball_max_gap: 0.1\n")
            cfg = RepsConfig.from_yaml(p)
            self.assertEqual((cfg.jump_speed, cfg.foot_ball_max_gap), (5.5, 0.1))
            p.write_text("jump_sped: 5.5\n")
            with self.assertRaises(ValueError):
                RepsConfig.from_yaml(p)

    def test_cli_writes_valid_reps(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "reps.json"
            # order of --in files doesn't matter
            rc = cli_main(["--in", str(EXAMPLES / "player.example.json"), str(EXAMPLES / "ball.example.json"),
                           str(EXAMPLES / "pose.example.json"), "--out", str(out)])
            self.assertEqual(rc, 0)
            reps = json.loads(out.read_text())
            self.assertEqual(validate.schema_errors("reps", reps), [])
            self.assertEqual(len(reps["reps"]), 2)


if __name__ == "__main__":
    unittest.main()
