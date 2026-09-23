"""Economic contract checks against the pinned source, using stdlib only."""

import ast
import copy
import math
from pathlib import Path
import unittest

from kaggriculture_agent.market import MARKET_PARAMS, MarketModel, price_at, sale_revenue


def observation(step=0, inventory=None, shops=(), opponent_tiles=None, own_stock=None):
    inv = {p: 10000 for p in MARKET_PARAMS}
    inv.update(inventory or {})
    return {"step": step, "day": step // 24, "hour": step % 24, "player": 0,
            "market": {"inventory": inv, "prices": {p: price_at(p, n) for p, n in inv.items()}},
            "town": {"unlocked_shops": list(shops)},
            "farms": [{"tiles": []}, {"tiles": opponent_tiles or []}],
            "private": {"shed": own_stock or {}, "inventories": [{}]}}


def official_price_function():
    """Load only mathematical declarations, without importing Kaggle or executing it."""
    path = Path(__file__).resolve().parents[1] / "vendor/kaggriculture/kaggriculture.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"MARKET_I0", "PRICE_FLOOR", "MARKET_PARAMS", "HINGE_GAIN"}
    nodes = [n for n in parsed.body if
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)) or
             (isinstance(n, ast.FunctionDef) and n.name in ("_shape", "market_price"))]
    scope = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["market_price"]


class PriceTests(unittest.TestCase):
    def test_matches_pinned_official_formula_all_products_both_sides(self):
        official = official_price_function()
        for product, p in MARKET_PARAMS.items():
            for inventory in (0, 8000, 9999, 10000, 10001, 10073, 10105, 11000, 100000):
                with self.subTest(product=product, inventory=inventory):
                    self.assertEqual(price_at(product, inventory), official(product, inventory))

    def test_sparse_overrides_and_every_curve_match_official(self):
        official = official_price_function()
        for shape in ("linear", "sq", "sqrt", "log", "log10", "hinge", "unknown"):
            patch = {"base": 73, "I0": 400, "T": 31, "below_func": shape,
                     "above_func": shape, "above_target": .7, "below_target": 1.2}
            resolved = {p: dict(v) for p, v in MARKET_PARAMS.items()}
            resolved["WOOL"].update(patch)
            for inv in (10, 350, 399, 400, 401, 431, 501):
                self.assertEqual(price_at("WOOL", inv, {"WOOL": patch}),
                                 official("WOOL", inv, resolved))

    def test_sequential_sale_slippage_and_floor_no_supply(self):
        inv, expected = 10025, 0
        for _ in range(100):
            price = price_at("STRAWBERRY", inv)
            expected += price
            if price > 1:
                inv += 1
        self.assertEqual(sale_revenue("STRAWBERRY", 10025, 100), expected)
        self.assertLess(expected, price_at("STRAWBERRY", 10025) * 100)
        self.assertEqual(sale_revenue("STRAWBERRY", inv, 1000000), 1000000)
        self.assertEqual(sale_revenue("WHEAT", 10000, 0), 0)


class ForecastTests(unittest.TestCase):
    def test_probabilities_quantiles_determinism_and_own_slippage(self):
        first, second = MarketModel(), MarketModel()
        obs = observation(240, shops=["PET_CAFE", "PET_CAFE"])
        first.update(obs)
        second.update(obs)
        a = first.forecast("CARROT", 96, 30)
        self.assertEqual(a, second.forecast("CARROT", 96, 30))
        self.assertIs(a, first.forecast("CARROT", 96, 30))
        self.assertAlmostEqual(a.prob_up + a.prob_flat + a.prob_down, 1.)
        self.assertLessEqual(a.p10, a.p50)
        self.assertLessEqual(a.p50, a.p90)
        self.assertLessEqual(a.expected_revenue, 30 * a.expected_price)
        self.assertEqual(a.samples, 32)
        self.assertEqual(a.confidence, "prior_uncalibrated")

    def test_duplicated_known_shops_raise_expected_demand_price(self):
        # Disable new unlocks so the only difference is observed, repeated demand.
        base = MarketModel({"townShopUnlockInterval": 1000})
        demand = MarketModel({"townShopUnlockInterval": 1000})
        base.update(observation(100))
        demand.update(observation(100, shops=["PET_CAFE"] * 4))
        self.assertGreater(demand.forecast("CARROT", 200).expected_price,
                           base.forecast("CARROT", 200).expected_price)

    def test_cached_revenue_prefix_remains_exact_across_quantities_and_floor(self):
        model = MarketModel()
        model.update(observation(100, {"STRAWBERRY": 10025}))
        for n in (10, 3, 100, 400, 5, 1000000):
            self.assertEqual(model.forecast("STRAWBERRY", 0, n).expected_revenue,
                             sale_revenue("STRAWBERRY", 10025, n))
        # A long sale at the floor must not allocate a million-entry prefix.
        self.assertLess(len(model._revenue_prefixes[("STRAWBERRY", 10025)]), 100)

    def test_residual_subtracts_own_sales_buys_and_known_town(self):
        own = MarketModel()
        obs = observation(1, own_stock={"WHEAT": 20})
        obs["private"]["inventories"] = [{"CARROT": 9}]
        own.update(obs)
        own.update(observation(2, {"WHEAT": 10008, "CARROT": 10009}),
                   [["SELL", "WHEAT", 10], ["BUY_PRODUCT", "WHEAT", 2], ["SELL", "CARROT", 9]])
        self.assertEqual(own._history["WHEAT"][-1], (0., False))
        self.assertEqual(own._history["CARROT"][-1], (0., False))
        town = MarketModel()
        town.update(observation(0, shops=["PET_CAFE", "PET_CAFE"]))
        town.update(observation(1, {p: 9995 if p == "CARROT" else 9999 for p in MARKET_PARAMS if p != "FERTILIZER"},
                                shops=["PET_CAFE", "PET_CAFE"]))
        self.assertTrue(all(samples[-1] == (0., False) for samples in town._history.values()))

    def test_floor_censorship_does_not_learn_zero_sales(self):
        model = MarketModel()
        model.update(observation(1, {"STRAWBERRY": 10200}, own_stock={"STRAWBERRY": 50}))
        model.update(observation(2, {"STRAWBERRY": 10200}), [["SELL", "STRAWBERRY", 50]])
        self.assertTrue(model._history["STRAWBERRY"][-1][1])
        self.assertEqual(model._history["STRAWBERRY"][-1][0], 0.)
        self.assertEqual(model.forecast("STRAWBERRY", 24).confidence, "prior_uncalibrated")

    def test_recent_external_supply_lowers_forecast_after_same_current_state(self):
        quiet, selling = MarketModel(), MarketModel()
        # Same final stock and step; only inferred external trading differs.
        for step in range(1, 21):
            quiet.update(observation(step, {"FERTILIZER": 10000}))
            selling.update(observation(step, {"FERTILIZER": 9800 + step * 10}))
        self.assertLess(selling.forecast("FERTILIZER", 48).expected_price,
                        quiet.forecast("FERTILIZER", 48).expected_price)

    def test_public_opponent_production_is_uncertain_supply(self):
        empty, farm = MarketModel(), MarketModel()
        empty.update(observation(200))
        tiles = [[{"kind": "PLANT", "crop": "CARROT", "planted_day": 5,
                   "yield_units": 3, "max_lifespan_step": 300}] * 20]
        farm.update(observation(200, opponent_tiles=tiles))
        self.assertLess(farm.forecast("CARROT", 24).expected_price,
                        empty.forecast("CARROT", 24).expected_price)

    def test_public_production_does_not_repeat_observed_watering_bonus(self):
        # At peak age a watered carrot has no remaining growth. An unwatered
        # carrot can still gain today's bonus, and watering remains possible
        # tomorrow for a younger carrot even when it was watered today.
        cases = ((3, 3, True, 1, 3),
                 (3, 2, False, 1, 3),
                 (2, 2, True, 24, 3),
                 (2, 1, False, 24, 3))
        for age, visible, watered, horizon, expected in cases:
            with self.subTest(age=age, watered=watered, horizon=horizon):
                model = MarketModel()
                tile = {"kind": "PLANT", "crop": "CARROT", "planted_day": 0,
                        "yield_units": visible, "watered_today": watered,
                        "max_lifespan_step": 96}
                model.update(observation(age * 24 + 1, opponent_tiles=[[tile]]))
                self.assertEqual(model._public_potential("CARROT", horizon),
                                 (float(expected), 0.0))

    def test_snapshot_is_copied_and_opponent_private_ignored(self):
        model, other = MarketModel(), MarketModel()
        obs = observation(100)
        with_secrets = copy.deepcopy(obs)
        with_secrets["farms"][1]["shed"] = {"CARROT": 1000000}
        with_secrets["farms"][1]["private"] = {"inventories": [{"MILK": 10000}]}
        model.update(obs)
        other.update(with_secrets)
        before = model.forecast("CARROT", 24)
        obs["market"]["inventory"]["CARROT"] = -100000
        self.assertEqual(before, model.forecast("CARROT", 24))
        self.assertEqual(before, other.forecast("CARROT", 24))

    def test_config_observation_override_and_end_game_horizon(self):
        model = MarketModel({"marketParams": {"CARROT": {"base": 100}}})
        obs = observation(718)
        obs["market"]["params"] = {"CARROT": {"I0": 9000}}
        model.update(obs)
        a = model.forecast("CARROT", 720, 3)
        self.assertEqual(a.samples, 1)
        self.assertEqual(a.expected_price, price_at("CARROT", 10000, {"CARROT": {"base": 100, "I0": 9000}}))
        self.assertEqual(a.expected_revenue, sale_revenue("CARROT", 10000, 3, model.params))

    def test_episode_reset_instance_isolation_and_skipped_observations(self):
        model, other = MarketModel(history_limit=4), MarketModel()
        for step in range(1, 8):
            model.update(observation(step))
        self.assertEqual(len(model._history["WHEAT"]), 4)
        model.update(observation(20))
        self.assertEqual(len(model._history["WHEAT"]), 4)
        model.update(observation(0))
        self.assertEqual(len(model._history["WHEAT"]), 0)
        other.update(observation(0))
        self.assertEqual(model.forecast("WOOL", 100), other.forecast("WOOL", 100))
        model.update(observation(1))
        changed_episode = observation(2)
        changed_episode["episode_id"] = "next"
        model.update(changed_episode)
        self.assertEqual(len(model._history["WHEAT"]), 0)


if __name__ == "__main__":
    unittest.main()
