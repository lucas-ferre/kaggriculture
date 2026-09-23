"""Numerical market regressions: discrete lots, weighted quadrature and causality."""
import math
import unittest

from kaggriculture_agent.market import MarketModel, PRODUCTS, price_at, sale_quantity_at_reserve
from kaggriculture_agent.numerics import NORMAL_NODES_5, NORMAL_WEIGHTS_5, trailing_quadratic_slope, weighted_quantile
from tests.test_agent import observation


class NumericalTests(unittest.TestCase):
    def test_lot_threshold_matches_exhaustive_integer_quotes(self):
        for product in PRODUCTS:
            for inventory in (9500, 9999, 10000, 10040, 11000):
                for reserve in (0, 1, 2, 25, 60, 200, 1000):
                    expected = sum(price_at(product, inventory + i) >= reserve for i in range(200))
                    self.assertEqual(sale_quantity_at_reserve(product, inventory, 200, reserve), expected)
        self.assertEqual(sale_quantity_at_reserve("MILK", 10000, 0, 50), 0)
        with self.assertRaises(ValueError):
            sale_quantity_at_reserve("MILK", 10000, 3, math.nan)

    def test_quadrature_weights_integrate_standard_normal_moments(self):
        for power, expected in ((0, 1), (1, 0), (2, 1), (4, 3), (6, 15), (8, 105)):
            actual = sum(w * x ** power for x, w in zip(NORMAL_NODES_5, NORMAL_WEIGHTS_5))
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertEqual(weighted_quantile([30, 10, 20], [.1, .1, .8], .25), 20)
        self.assertEqual(weighted_quantile([30, 10, 20], [.1, .1, .8], 0), 10)

    def test_trailing_quadratic_derivative_uses_past_endpoint(self):
        self.assertAlmostEqual(trailing_quadratic_slope([7] * 7), 0)
        self.assertAlmostEqual(trailing_quadratic_slope([3 + 2 * i for i in range(7)]), 2)
        self.assertAlmostEqual(trailing_quadratic_slope([i * i for i in range(7)]), 12)
        with self.assertRaises(ValueError):
            trailing_quadratic_slope([1] * 6)

    def test_quadrature_forecast_uses_weights_for_prices_probabilities_and_quantiles(self):
        model = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None})
        model.update(observation(1))
        inventory = [9800, 9900, 10000, 10100, 10200]
        model._scenario_inventories = lambda product, horizon: inventory
        forecast = model.forecast("MILK", 24, 1)
        prices = [price_at("MILK", value) for value in inventory]
        self.assertAlmostEqual(forecast.expected_price, sum(p * w for p, w in zip(prices, NORMAL_WEIGHTS_5)))
        self.assertAlmostEqual(forecast.prob_flat, NORMAL_WEIGHTS_5[2])
        self.assertAlmostEqual(forecast.prob_up, sum(NORMAL_WEIGHTS_5[:2]))
        self.assertEqual(forecast.p50, prices[2])
        self.assertEqual(forecast.samples, 5)
        self.assertIn("approximation", forecast.confidence)

    def test_analytic_shop_demand_obeys_future_ticks_replacement_and_cap(self):
        model = MarketModel()
        model.update(observation(1))
        # MILK appears in 3/8 shop types. A new day-3 shop consumes at step72.
        self.assertAlmostEqual(model.expected_town_demand("MILK", 71, 73, []), 1 + 3 / 8)
        # Eight duplicate shops are legal and suppress all future unlocks.
        shops = ["PIZZA_SHOP"] * 8
        self.assertEqual(model.expected_town_demand("MILK", 71, 73, shops), 9)
        self.assertEqual(model.expected_town_demand("FERTILIZER", 0, 720, []), 0)

    def test_quadrature_batches_are_deterministic_terminal_safe_and_weighted(self):
        a = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None})
        b = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None})
        for model in (a, b):
            model.update(observation(200))
        self.assertEqual(a.forecast_batches("MILK", [(24, 10), (48, 20)]),
                         b.forecast_batches("MILK", [(24, 10), (48, 20)]))
        a.update(observation(718))
        forecast = a.forecast("MILK", 1000, 1)
        self.assertEqual(forecast.samples, 1)
        self.assertEqual(forecast.expected_price, 160)

    def test_sg_does_not_bridge_a_missing_or_censored_observation(self):
        model = MarketModel({"forecastingMethod": "quadrature", "quantileModel": None})
        for step in range(7):
            model.update(observation(step))
        self.assertEqual(model._contiguous_updates, 6)
        model.update(observation(20))
        self.assertEqual(model._contiguous_updates, 0)
        # A censored final sample must give exactly the EMA fallback.
        model._contiguous_updates = 6
        model._history["MILK"].append((999., True))
        first = model._trend("MILK")
        model.causal_trend = False
        self.assertEqual(first, model._trend("MILK"))


if __name__ == "__main__":
    unittest.main()
