"""Market timing and capital/feeding regressions introduced in 0.1.1."""
import unittest

from kaggriculture_agent.agent import FarmAgent
from kaggriculture_agent.domain import game_configuration
from kaggriculture_agent.economy import animal_counts, livestock_plan, purchase_cost
from kaggriculture_agent.market import MarketModel, price_at, sale_revenue
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from tests.test_agent import observation


class DatedMarketTests(unittest.TestCase):
    def test_batch_carries_earlier_own_supply_and_coalesces_same_date(self):
        model = MarketModel()
        model.update(observation(1))
        # Known fixed external path isolates sequential own sale accounting.
        model._scenario_paths = lambda product, horizons: [{h: 10000 for h in horizons}]
        expected = sale_revenue("MILK", 10000, 20)
        result = model.forecast_batches("MILK", [(0, 10), (5, 10)])
        self.assertEqual(result.expected_revenue, expected)
        self.assertEqual(model.revenue_quantile("MILK", [(0, 10), (5, 10)]), expected)
        self.assertEqual(model.forecast_batches("MILK", [(5, 10), (5, 10)]).expected_revenue, expected)

    def test_dated_sales_benefit_from_intervening_demand(self):
        model = MarketModel()
        model.update(observation(1))
        model._scenario_paths = lambda product, horizons: [{h: 10000 - h for h in horizons}]
        dated = model.forecast_batches("MILK", [(0, 10), (20, 10)])
        self.assertEqual(dated.expected_revenue,
                         sale_revenue("MILK", 10000, 10) + sale_revenue("MILK", 9990, 10))

    def test_own_floor_sales_create_no_phantom_supply_in_next_batch(self):
        model = MarketModel()
        model.update(observation(1))
        model._scenario_paths = lambda product, horizons: [{h: (10200 if h == 0 else 10000) for h in horizons}]
        result = model.forecast_batches("STRAWBERRY", [(0, 1000), (20, 10)])
        self.assertEqual(result.expected_revenue, 1000 + sale_revenue("STRAWBERRY", 10000, 10))

    def test_visible_crop_supply_waits_until_maturity(self):
        obs = observation(1)
        obs["farms"][1]["tiles"][4][4] = {
            "kind": "PLANT", "crop": "CARROT", "planted_day": 0,
            "yield_units": 1, "watered_today": True, "max_lifespan_step": 96}
        model = MarketModel()
        model.update(obs)
        self.assertEqual(model.public_pulses("CARROT", 24), [])
        self.assertEqual(model.public_pulses("CARROT", 80), [(71, 3.)])

    def test_premium_lead_uses_visible_stock_only_without_consumption_tick(self):
        obs = observation(5)
        obs["farms"][1]["tiles"][4][4] = {"kind": "PASTURE", "animal": "COW",
                                                       "placed_day": -10, "yield_units": 3}
        model = MarketModel()
        model.update(obs)
        self.assertTrue(model.should_lead_sale("MILK", 4))
        obs["step"], obs["hour"] = 24, 0
        model.update(obs)
        self.assertFalse(model.should_lead_sale("MILK", 4))
        with self.assertRaises(ValueError):
            model.revenue_quantile("MILK", [(1, 1)], 1.1)

    def test_melon_pulse_arrives_at_maturity_not_lifespan(self):
        obs = observation(241)
        obs["farms"][1]["tiles"][4][4] = {
            "kind": "PLANT", "crop": "MELON", "planted_day": 0,
            "yield_units": 6, "watered_today": True, "max_lifespan_step": 312}
        model = MarketModel()
        model.update(obs)
        self.assertEqual(model.public_pulses("MELON", 1), [(1, 6.)])


class LivestockEconomyTests(unittest.TestCase):
    def test_deposited_feed_is_not_also_counted_as_carried_and_sold(self):
        obs = observation(250)
        obs["private"]["inventories"] = [{"WHEAT": 2}]
        obs["farms"][0]["tiles"][4][4] = {
            "kind": "PASTURE", "animal": "COW", "placed_day": 0,
            "yield_units": 0, "fed_today": True, "cared_today": True,
            "fertilizer_available": False}
        bot = FarmAgent(PolicyConfig(max_plots=0, enable_expansion=False, use_forecast=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 1})
        action = bot(obs)
        self.assertEqual(action["farmer"], ["DROP"])
        self.assertFalse(any(order[0] in {"SELL", "BUY_PRODUCT"} for order in action["market"]))

    def test_post_action_feed_conservation_handles_pickup_feed_and_harvest(self):
        obs = observation(100)
        obs["private"]["shed"] = {"WHEAT": 3}
        self.assertEqual(FarmAgent._carried_feed_after_actions(obs, [["PICKUP", "WHEAT", 2]], {"WHEAT": 1}), 2)
        obs["private"]["inventories"] = [{"WHEAT": 2}]
        self.assertEqual(FarmAgent._carried_feed_after_actions(obs, [["FEED"]], {"WHEAT": 3}), 1)
        obs["farms"][0]["tiles"][4][4] = {"kind": "PLANT", "crop": "WHEAT", "yield_units": 4}
        self.assertEqual(FarmAgent._carried_feed_after_actions(obs, [["HARVEST"]], {"WHEAT": 3}), 6)

    def test_bought_and_carried_animals_are_counted_once(self):
        obs = observation()
        obs["private"]["shed"] = {"COW": 2}
        obs["private"]["inventories"] = [{"COW": 1, "WHEAT": 9}]
        obs["farms"][0]["tiles"][3][4] = {"kind": "PASTURE", "animal": "COW"}
        live, owned = animal_counts(obs)
        self.assertEqual(live["COW"], 1)
        self.assertEqual(owned["COW"], 4)

    def test_purchase_quotes_use_post_buy_stock(self):
        self.assertEqual(purchase_cost("WHEAT", 10000, 10),
                         sum(price_at("WHEAT", 9999 - i) for i in range(10)))

    def test_no_late_livestock_and_no_spending_cash_reserve(self):
        for step, money in ((650, 100000), (0, 300)):
            obs = observation(step)
            obs["farms"][0]["money"] = money
            model = MarketModel(game_configuration())
            model.update(obs)
            targets, debug = livestock_plan(obs, game_configuration(), PolicyConfig(), model)
            self.assertEqual(sum(targets.values()), 0)
            self.assertEqual(debug["reserved_cash"], 0)

    def test_feed_buffer_is_not_repeatedly_sold_and_repurchased(self):
        obs = observation(10)
        obs["private"]["shed"] = {"WHEAT": 3}
        for x in (3, 4):
            obs["farms"][0]["tiles"][3][x] = {"kind": "PASTURE", "animal": "COW"}
        bot = FarmAgent(PolicyConfig(use_forecast=False, enable_expansion=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 2})
        orders, _ = bot._market_orders(obs, game_configuration(), obs["private"]["shed"])
        self.assertFalse(any(order[0] in {"SELL", "BUY_PRODUCT"} for order in orders))

    def test_final_day_releases_feed_to_sell(self):
        obs = observation(718)
        obs["private"]["shed"] = {"WHEAT": 3}
        obs["farms"][0]["tiles"][3][4] = {"kind": "PASTURE", "animal": "COW"}
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 1})
        orders, _ = bot._market_orders(obs, game_configuration(), obs["private"]["shed"])
        self.assertIn(["SELL", "WHEAT", 3], orders)
        self.assertFalse(any(order[0].startswith("BUY") for order in orders))


if __name__ == "__main__":
    unittest.main()
