"""Frozen model assets and regression history metrics for the new policy."""
import importlib
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark import ROOT, _crop_statistics, episode_metrics, load_baseline, make_environment, run_match, save_replay


class BenchmarkRegressionTests(unittest.TestCase):
    def test_repeated_seeds_never_overwrite_archived_episode(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            target = Path(directory) / "adaptive_starter_11_0.json"
            first = save_replay(target, {"money": 1})
            original = first.read_bytes()
            second = save_replay(target, {"money": 2})
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            self.assertEqual(save_replay(target, {"money": 2}), second)

    def test_frozen_012_includes_original_calibration_and_plays(self):
        factory, config = load_baseline("0.1.2")
        module = importlib.import_module("_baseline_012.quantile")
        self.assertTrue(Path(module.__file__).with_name("quantile_coefficients.json").is_file())
        self.assertIsNotNone(module.load_approved_model())
        result = run_match(config(), seed=7, opponent="baseline_012", steps=4)
        self.assertTrue(result["valid"])
        self.assertEqual(result["policy_source"]["version"], "0.1.2")
        self.assertEqual(result["opponent_source"]["version"], "0.1.2")

    def test_territory_use_records_actual_owned_production(self):
        env = make_environment(7, 8, {"weedSpawnChance": 0})
        idle = {"farmer": ["PASS"], "hands": [], "market": []}
        env.step([{"market": [["BUY_LAND"], ["BUY_SEED", "TOMATO", 1]]}, idle])
        env.step([{"farmer": ["EAST"]}, idle])
        env.step([{"farmer": ["PLANT", "TOMATO"]}, idle])
        metrics = episode_metrics(env, 0, audit_unit_actions=False)
        self.assertEqual(metrics["land"]["purchased_quadrants"], 1)
        self.assertEqual(len(metrics["land"]["purchase_steps"]), 1)
        self.assertIn("NE", metrics["land"]["productive_quadrants"])
        crops = _crop_statistics(env, 0)
        self.assertEqual(crops["first_planted_day"], {"TOMATO": 0})
        self.assertEqual(crops["recurrent_occupied_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
