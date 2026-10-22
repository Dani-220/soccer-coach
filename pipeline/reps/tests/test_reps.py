"""Run from the repo root:  python -m unittest pipeline.reps.tests.test_reps -v"""
import importlib.util
import json
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

    def test_ball_moves_without_a_foot_near_it(self):
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(foot_reaches_ball=False)]))
        self.assertEqual(reps, [])

    def test_slow_roll_is_not_a_shot(self):
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(ball_speed_hs=1.5)]))
        self.assertEqual(reps, [])

    def test_thresholds_are_configurable(self):
        sc = Scenario(reps=[RepSpec(ball_speed_hs=1.5)])
        reps, truths = self.run_scenario(sc, RepsConfig(jump_speed=1.0))
        self.assertEqual(len(reps), 1)
        # the foot stops ~0.33 H short of the ball: rejected by default, accepted with a loose threshold
        reps, _ = self.run_scenario(Scenario(reps=[RepSpec(foot_reaches_ball=False)]), RepsConfig(ankle_ball_max_dist=0.5))
        self.assertEqual(len(reps), 1)

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
            p.write_text("jump_speed: 5.5\nankle_ball_max_dist: 0.1\n")
            cfg = RepsConfig.from_yaml(p)
            self.assertEqual((cfg.jump_speed, cfg.ankle_ball_max_dist), (5.5, 0.1))
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
