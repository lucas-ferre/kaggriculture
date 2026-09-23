"""Regression fitting, episode isolation and calibrated spot probabilities."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kaggriculture_agent.market import MarketModel, PRODUCTS
from kaggriculture_agent.mrlm import (CLASS_NAMES, MODEL_KIND, load_approved_model,
                                     predict_probabilities, predict_ratio, residual_probabilities)
from kaggriculture_agent.quantile import FEATURE_NAMES, configuration_signature
from scripts.train_mrlm import (calibrate_residuals, collect_rows, fit_linear_regression,
                                probability_metrics, train_and_validate)
from tests.test_agent import observation


def approved_model():
    width = len(FEATURE_NAMES)
    return {"schema_version": 1, "model_kind": MODEL_KIND, "approved": True,
            "feature_names": list(FEATURE_NAMES), "class_names": list(CLASS_NAMES),
            "training_configuration": configuration_signature(), "horizon_turns": 24,
            "models": {"MILK": {"approved": True, "coefficients": [1.] + [0.] * width,
                                "feature_means": [0.] * width, "feature_scales": [1.] * width,
                                "feature_min": [-100.] * width, "feature_max": [100.] * width,
                                "calibration_residuals": [-.1] * 15 + [0.] * 15 + [.1] * 15}}}


class RegressionTests(unittest.TestCase):
    def test_multivariate_fit_recovers_signal_and_uses_training_statistics(self):
        rows = [{"features": [float(x), float(x * x)], "target_ratio": 3. + .5 * x + .02 * x * x}
                for x in range(-10, 11)]
        fit = fit_linear_regression(rows, l2=1e-6)
        self.assertEqual(fit["feature_means"][0], 0.)
        self.assertLess(fit["training_rmse_ratio"], .00001)
        self.assertAlmostEqual(predict_ratio(fit, [3., 9.]), 4.68, places=4)
        before = copy.deepcopy(fit)
        calibrated = calibrate_residuals(fit, [
            {"features": [0., 0.], "target_ratio": 4.},
            {"features": [1000., 1000000.], "target_ratio": 99999.}])
        self.assertEqual(fit, before)
        self.assertEqual(calibrated["coefficients"], fit["coefficients"])
        self.assertEqual(calibrated["feature_means"], fit["feature_means"])
        self.assertEqual(calibrated["calibration_rows"], 1)
        self.assertAlmostEqual(calibrated["calibration_residuals"][0], 1., places=4)

    def test_invalid_regression_settings_are_rejected(self):
        row = {"features": [1.], "target_ratio": 2.}
        for rows, penalty in (([], .01), ([row], 0), ([row], math.nan),
                              ([{"features": [math.inf], "target_ratio": 1.}], .01),
                              ([row, {"features": [1., 2.], "target_ratio": 2.}], .01)):
            with self.subTest(rows=rows, penalty=penalty), self.assertRaises(ValueError):
                fit_linear_regression(rows, penalty)

    def test_residual_events_match_integer_rounding_and_impossible_floor_direction(self):
        values = [-10., 1., 1.5, 2., 2.5, 3., 3.5, 9.]
        fit = {"coefficients": [0.], "feature_means": [], "feature_scales": [],
               "calibration_residuals": values}
        for current in (1, 2, 3, 4):
            prices = [max(1, round(value)) for value in values]
            counts = [sum(value > current for value in prices),
                      sum(value == current for value in prices),
                      sum(value < current for value in prices)]
            prior = [1, 1, int(current > 1)]
            expected = tuple((count + pseudo) / (len(values) + sum(prior))
                             for count, pseudo in zip(counts, prior))
            with self.subTest(current=current):
                actual = residual_probabilities(fit, [], current, 1.)
                self.assertEqual(actual, expected)
                self.assertAlmostEqual(sum(actual), 1.)
                self.assertTrue(all(0 <= value <= 1 for value in actual))
        self.assertEqual(residual_probabilities(fit, [], 1, 1.)[2], 0.)

    def test_probability_metrics_distinguish_calibration_from_wrong_certainty(self):
        rows = [{"outcome": 0}, {"outcome": 1}, {"outcome": 2}]
        perfect = probability_metrics(rows, [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)])
        wrong = probability_metrics(rows, [(0., 1., 0.), (0., 0., 1.), (1., 0., 0.)])
        self.assertEqual(perfect["brier"], 0.)
        self.assertEqual(perfect["ece"], 0.)
        self.assertEqual(wrong["brier"], 2.)
        self.assertGreater(wrong["ece"], 0.)


class RuntimeTests(unittest.TestCase):
    def test_runtime_gates_horizon_product_rules_and_feature_domain(self):
        model, obs = approved_model(), observation(100)
        probabilities = predict_probabilities(model, obs, "MILK", 24)
        self.assertEqual(probabilities, (1 / 3, 1 / 3, 1 / 3))
        self.assertIsNone(predict_probabilities(model, obs, "MILK", 72))
        self.assertIsNone(predict_probabilities(model, obs, "WOOL", 24))
        self.assertIsNone(predict_probabilities(model, obs, "MILK", 24, {"turnsPerDay": 12}))
        changed = copy.deepcopy(obs)
        changed["market"]["params"] = {"MILK": {"base": 999}}
        self.assertIsNone(predict_probabilities(model, changed, "MILK", 24))
        changed = copy.deepcopy(obs)
        changed["market"]["inventory"]["MILK"] = 10_000_000
        self.assertIsNone(predict_probabilities(model, changed, "MILK", 24))
        changed = copy.deepcopy(obs)
        changed["market"]["prices"]["MILK"] = 1.5
        self.assertIsNone(predict_probabilities(model, changed, "MILK", 24))

    def test_malformed_or_unapproved_fits_fall_back(self):
        mutations = [lambda p: p.update(approved=False),
                     lambda p: p.update(class_names=["down", "flat", "up"]),
                     lambda p: p["models"].update(MILK=None),
                     lambda p: p["models"]["MILK"].update(approved=False),
                     lambda p: p["models"]["MILK"].update(coefficients=[math.nan] * 10),
                     lambda p: p["models"]["MILK"].update(feature_scales=[0.] * 9),
                     lambda p: p["models"]["MILK"].update(feature_min=None),
                     lambda p: p["models"]["MILK"].update(calibration_residuals=[0.] * 29),
                     lambda p: p["models"]["MILK"].update(calibration_residuals=[1., -1.] * 20)]
        for index, mutate in enumerate(mutations):
            model = approved_model()
            mutate(model)
            with self.subTest(mutation=index):
                self.assertIsNone(predict_probabilities(model, observation(100), "MILK", 24))

    def test_integration_changes_spot_probabilities_only(self):
        learned = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None,
                               "mrlmModel": approved_model()})
        empirical = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None, "mrlmModel": None})
        for market in (learned, empirical):
            market.update(observation(100))
        forecast, baseline = learned.forecast("MILK", 24, 10), empirical.forecast("MILK", 24, 10)
        self.assertIn("validated_mrlm", forecast.confidence)
        self.assertEqual((forecast.prob_up, forecast.prob_flat, forecast.prob_down), (1 / 3,) * 3)
        for field in ("expected_price", "expected_revenue", "p10", "p50", "p90", "samples"):
            self.assertEqual(getattr(forecast, field), getattr(baseline, field))
        self.assertEqual(learned.forecast_batches("MILK", [(24, 5), (48, 10)]),
                         empirical.forecast_batches("MILK", [(24, 5), (48, 10)]))
        self.assertEqual(learned.forecast("WOOL", 24), empirical.forecast("WOOL", 24))
        self.assertEqual(learned.forecast("MILK", 23), empirical.forecast("MILK", 23))
        for market in (learned, empirical):
            market.update(observation(718))
        self.assertEqual(learned.forecast("MILK", 24), empirical.forecast("MILK", 24))

    def test_loader_requires_approved_compatible_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.assertIsNone(load_approved_model(path))
            path.write_text("broken json", encoding="utf-8")
            self.assertIsNone(load_approved_model(path))
            model = approved_model()
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertEqual(load_approved_model(path), model)
            model["model_kind"] = "different_regression"
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertIsNone(load_approved_model(path))


class EpisodeIsolationTests(unittest.TestCase):
    def _manifests(self):
        return [[{"sha256": str(seed), "seed": seed, "configuration": configuration_signature()}
                 for seed in seeds] for seeds in ((101, 103), (151, 157), (211, 223))]

    def test_all_three_split_pairs_reject_shared_seed_or_episode_before_extraction(self):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            for key in ("seed", "sha256"):
                manifests = self._manifests()
                manifests[second][0][key] = manifests[first][0][key]
                with self.subTest(pair=(first, second), key=key), \
                     patch("scripts.train_mrlm.replay_manifest", side_effect=manifests), \
                     patch("scripts.train_mrlm.collect_rows") as collect, self.assertRaises(ValueError):
                    train_and_validate(["train"], ["calibration"], ["validation"])
                collect.assert_not_called()

    def test_duplicate_missing_seed_and_changed_rules_are_rejected(self):
        for defect in ("duplicate", "missing_seed", "configuration"):
            manifests = self._manifests()
            if defect == "duplicate":
                manifests[0].append(dict(manifests[0][0]))
            elif defect == "missing_seed":
                manifests[1][0]["seed"] = None
            else:
                manifests[2][0]["configuration"] = configuration_signature({"turnsPerDay": 12})
            with self.subTest(defect=defect), \
                 patch("scripts.train_mrlm.replay_manifest", side_effect=manifests), \
                 self.assertRaises(ValueError):
                train_and_validate(["train"], ["calibration"], ["validation"])

    def test_gate_approves_only_when_calibrated_probabilities_improve(self):
        rows = [{"features": [0.] * len(FEATURE_NAMES), "target_ratio": 1.1,
                 "base": 100., "current_price": 100, "outcome": 0,
                 "baseline_probabilities": [.01, .98, .01], "episode": str(i % 2)}
                for i in range(40)]
        datasets = [{product: copy.deepcopy(rows) for product in PRODUCTS} for _ in range(3)]
        # A perfect baseline must keep its product out of the approved model.
        for row in datasets[2]["WOOL"]:
            row["baseline_probabilities"] = [1., 0., 0.]
        with patch("scripts.train_mrlm.replay_manifest", side_effect=self._manifests()), \
             patch("scripts.train_mrlm.collect_rows", side_effect=datasets):
            report, model = train_and_validate(["train"], ["calibration"], ["validation"])
        self.assertTrue(report["approved"])
        self.assertIn("MILK", model["models"])
        self.assertNotIn("WOOL", model["models"])
        self.assertFalse(report["products"]["WOOL"]["gate"]["brier_improved_2pct"])
        self.assertEqual(model["models"]["MILK"]["calibration_rows"], 40)
        self.assertEqual(model["provenance"]["training_replay_sha256"], ["101", "103"])

    def test_future_replay_changes_do_not_enter_earlier_features(self):
        frames = []
        for step in range(6):
            obs = observation(step)
            frames.append([{"observation": obs, "action": {}},
                           {"observation": {"private": {}}, "action": {}}])
        payload = {"steps": frames, "configuration": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            original = collect_rows([path], horizon=2, stride=100)["MILK"]
            frames[2][0]["observation"]["market"]["prices"]["MILK"] = 999
            frames[0][0]["observation"]["private"] = {"shed": {"MILK": 12345}}
            frames[2][0]["observation"]["farms"][1]["tiles"][0][0] = {"animal": "COW", "yield_units": 99}
            path.write_text(json.dumps(payload), encoding="utf-8")
            changed = collect_rows([path], horizon=2, stride=100)["MILK"]
        self.assertEqual(original[0]["features"], changed[0]["features"])
        self.assertEqual(original[0]["baseline_probabilities"], changed[0]["baseline_probabilities"])
        self.assertNotEqual(original[0]["target_ratio"], changed[0]["target_ratio"])


if __name__ == "__main__":
    unittest.main()
