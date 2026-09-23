"""Offline quantile fitting, split isolation and honest runtime fallback."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from kaggriculture_agent.market import MarketModel
from kaggriculture_agent.quantile import FEATURE_NAMES, configuration_signature, load_approved_model, predict_price_quantile, public_features
from scripts.train_quantile import fit_linear_quantile, pinball, predict_ratio, validate_splits
from tests.test_agent import observation


def approved_model():
    width = len(FEATURE_NAMES)
    return {"schema_version": 1, "approved": True, "feature_names": list(FEATURE_NAMES),
            "training_configuration": configuration_signature(),
            "horizon_turns": 24, "quantile_tau": .25,
            "models": {"MILK": {"approved": True, "coefficients": [2.] + [0.] * width,
                                "feature_means": [0.] * width, "feature_scales": [1.] * width,
                                "feature_min": [-100.] * width, "feature_max": [100.] * width}}}


class QuantileTests(unittest.TestCase):
    def test_features_ignore_private_stocks_and_unavailable_opponent_orders(self):
        original = observation(100)
        changed = copy.deepcopy(original)
        changed["private"] = {"shed": {"MILK": 99999}}
        changed["farms"][1]["private"] = {"seeds": {"MELON": 10000}}
        changed["opponent_action"] = {"market": [["SELL", "MILK", 10000]]}
        self.assertEqual(public_features(original, "MILK", 24), public_features(changed, "MILK", 24))

    def test_regularized_fit_learns_signal_without_validation_inputs(self):
        rows = [{"features": [float(x)], "target_ratio": 2. + .3 * x} for x in range(30)]
        fit = fit_linear_quantile(rows, l2=.001)
        self.assertAlmostEqual(fit["feature_means"][0], 14.5)
        self.assertLess(sum(pinball(row["target_ratio"], predict_ratio(fit, row["features"]), .25)
                            for row in rows) / len(rows), .05)
        self.assertLessEqual(fit["sweeps"], 80)

    def test_split_rejects_reused_episode_and_opposite_side_same_seed(self):
        with self.assertRaises(ValueError):
            validate_splits([{"sha256": "a", "seed": 1}], [{"sha256": "a", "seed": 2}])
        with self.assertRaises(ValueError):
            validate_splits([{"sha256": "a", "seed": 1}], [{"sha256": "b", "seed": 1}])
        validate_splits([{"sha256": "a", "seed": 1}], [{"sha256": "b", "seed": 2}])

    def test_runtime_gates_product_horizon_tau_and_approval(self):
        model = approved_model()
        obs = observation(100)
        self.assertEqual(predict_price_quantile(model, obs, "MILK", 24, .25), 320)
        self.assertIsNone(predict_price_quantile(model, obs, "MILK", 72, .25))
        self.assertIsNone(predict_price_quantile(model, obs, "MILK", 24, .1))
        self.assertIsNone(predict_price_quantile(model, obs, "WOOL", 24, .25))
        model["approved"] = False
        self.assertIsNone(predict_price_quantile(model, obs, "MILK", 24, .25))

    def test_only_price_quantile_uses_trained_spot_model(self):
        model = MarketModel({"quantileModel": approved_model()})
        empirical = MarketModel({"quantileModel": None})
        for market in (model, empirical):
            market.update(observation(100))
        self.assertEqual(model.price_quantile("MILK", 24, .25), 320)
        self.assertEqual(model.last_quantile_source, "validated_local_price_quantile")
        batches = [(24, 5), (48, 10)]
        self.assertEqual(model.revenue_quantile("MILK", batches), empirical.revenue_quantile("MILK", batches))
        self.assertEqual(model.price_quantile("WOOL", 24), empirical.price_quantile("WOOL", 24))
        self.assertEqual(model.last_quantile_source, "empirical_scenarios")

    def test_changed_curve_calendar_or_feature_domain_uses_fallback(self):
        payload = approved_model()
        obs = observation(100)
        self.assertIsNone(predict_price_quantile(payload, obs, "MILK", 24, .25, {"turnsPerDay": 12}))
        changed = copy.deepcopy(obs)
        changed["market"]["params"] = {"MILK": {"base": 999}}
        self.assertIsNone(predict_price_quantile(payload, changed, "MILK", 24, .25))
        obs["market"]["inventory"]["MILK"] = 10000000
        self.assertIsNone(predict_price_quantile(payload, obs, "MILK", 24, .25))

    def test_file_loader_refuses_unapproved_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.assertIsNone(load_approved_model(path))
            model = approved_model()
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertEqual(load_approved_model(path), model)
            model["approved"] = False
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertIsNone(load_approved_model(path))


if __name__ == "__main__":
    unittest.main()
