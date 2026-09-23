"""Public-only competitive state, contextual learning and economic safeguards."""
import copy
import unittest

from kaggriculture_agent.agent import FarmAgent
from kaggriculture_agent.domain import game_configuration
from kaggriculture_agent.economy import prioritize_market_orders
from kaggriculture_agent.intelligence import ContextualBandit, OpponentModel
from kaggriculture_agent.market import MarketModel
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from tests.test_agent import observation


class IntelligenceTests(unittest.TestCase):
    def test_opponent_private_fields_do_not_change_profile(self):
        obs = observation(4)
        altered = copy.deepcopy(obs)
        altered["farms"][1]["private"] = {"shed": {"MILK": 999999}, "seeds": {"MELON": 100}}
        altered["farms"][1]["market_orders"] = [["SELL", "MILK", 99999]]
        model = MarketModel()
        model.update(obs)
        self.assertEqual(OpponentModel().update(obs, model), OpponentModel().update(altered, model))

    def test_visible_harvest_and_external_sales_update_uncertain_stock(self):
        obs = observation(250)
        obs["farms"][1]["tiles"][4][4] = {"kind": "PASTURE", "animal": "COW", "placed_day": 0, "yield_units": 6}
        model, tracker = MarketModel(), OpponentModel()
        model.update(obs)
        tracker.update(obs, model)
        self.assertEqual(tracker.supply_risk("MILK"), 6)
        obs["step"] += 1
        obs["farms"][1]["tiles"][4][4]["yield_units"] = 0
        model.update(obs)
        tracker.update(obs, model)
        self.assertGreater(tracker.supply_risk("MILK"), 0)
        obs["step"] += 1
        obs["market"]["inventory"]["MILK"] += 6
        model.update(obs)
        tracker.update(obs, model)
        self.assertEqual(tracker.supply_risk("MILK"), 0)

    def test_opponent_reset_drops_previous_episode_stock(self):
        model, tracker = MarketModel(), OpponentModel()
        model.update(observation(10))
        tracker.update(observation(10), model)
        tracker.stock["MILK"] = 50
        model.update(observation())
        tracker.update(observation(), model)
        self.assertEqual(tracker.stock["MILK"], 0)

    def test_bandit_is_reproducible_updates_once_per_day_and_holds_three_days(self):
        first, second = ContextualBandit(), ContextualBandit()
        profile = {"visible_animals": {"COW": 4}, "visible_crops": {"WHEAT": 6}}
        initial = first.choose(observation(), profile)
        self.assertEqual(initial, second.choose(observation(), profile))
        for day in range(1, 4):
            obs = observation(day * 24)
            obs["farms"][0]["money"] += day * 1000
            arm = first.choose(obs, profile)
            self.assertEqual(arm, second.choose(obs, profile))
            if day < 3:
                self.assertEqual(arm, initial)
            self.assertEqual(sum(first.counts.values()), day)
            first.choose(obs, profile)
            self.assertEqual(sum(first.counts.values()), day)
        self.assertTrue(all(value > 0 for matrix in first.inverse.values()
                            for value in [matrix[i][i] for i in range(len(matrix))]))

    def test_risky_premium_sale_moves_to_first_order_slot(self):
        obs = observation(100)
        tracker = OpponentModel()
        tracker.stock["MILK"] = 100
        orders = [["HIRE"], ["SELL", "WHEAT", 4], ["SELL", "MILK", 4]]
        ranked = prioritize_market_orders(orders, obs, game_configuration(), tracker)
        self.assertEqual(ranked[0], ["SELL", "MILK", 4])
        self.assertCountEqual(ranked, orders)

    def test_urgent_feed_survives_single_market_slot_limit(self):
        obs = observation(100)
        orders = [["SELL", "MILK", 6], ["BUY_PRODUCT", "WHEAT", 2], ["HIRE"]]
        ranked = prioritize_market_orders(orders, obs,
                    game_configuration({"maxMarketOrdersPerTurn": 1}), feed_urgent=True)
        self.assertEqual(ranked, [["BUY_PRODUCT", "WHEAT", 2]])

    def test_final_day_without_saleable_work_does_not_hire_or_feed(self):
        obs = observation(696)
        obs["farms"][0]["tiles"][4][4] = {"kind": "PASTURE", "animal": "COW", "placed_day": 20,
                                                       "yield_units": 0, "fertilizer_available": False}
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 1}, workforce_target=8)
        orders, _ = bot._market_orders(obs, game_configuration(), {})
        self.assertFalse(any(o[0] in {"HIRE", "BUY_PRODUCT", "BUY_ANIMAL"} for o in orders))

    def test_full_shed_sale_must_precede_urgent_feed_purchase_in_engine(self):
        from scripts.benchmark import make_environment, load_engine
        env = make_environment(19, 720)
        obs = env.state[0].observation
        obs.private["shed"]["CARROT"] = 100
        obs.farms[0]["tiles"][4][4] = load_engine()._new_animal("COW", 0)
        obs.farms[0]["tiles"][4][4]["consecutive_unfed"] = 1
        bot = FarmAgent(PolicyConfig(max_plots=0, use_forecast=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 1}, workforce_target=1)
        orders, _ = bot._market_orders(obs, game_configuration(), obs.private["shed"])
        feed = next(i for i, order in enumerate(orders) if order[:2] == ["BUY_PRODUCT", "WHEAT"])
        sale = next(i for i, order in enumerate(orders) if order[:2] == ["SELL", "CARROT"])
        self.assertLess(sale, feed)
        action = {"farmer": ["PASS"], "hands": [], "market": orders}
        env.step([action, {"farmer": ["PASS"], "hands": [], "market": []}])
        self.assertEqual(env.state[0].observation.private["shed"]["WHEAT"], 2)

    def test_midnight_pressure_forces_shed_sale_and_releases_space(self):
        obs = observation(262)  # Hour22, returns happen after the market stage.
        obs["private"]["shed"] = {"MILK": 40}
        obs["private"]["inventories"] = [{"STRAWBERRY": 80}]
        bot = FarmAgent(PolicyConfig(use_forecast=True, enable_livestock=False))
        bot.market = MarketModel()
        bot.market.update(obs)
        bot.plan = CropPlan(None, None, 0, 0)
        orders, debug = bot._market_orders(obs, game_configuration(), {"MILK": 40}, 0, {"STRAWBERRY": 80})
        self.assertIn(["SELL", "MILK", 40], orders)
        self.assertEqual(debug["MILK"]["reason"], "midnight_shed_capacity")

    def test_cargo_conserves_manure_and_animal_placement(self):
        obs = observation(100)
        obs["private"]["inventories"] = [{"FERTILIZER": 2, "COW": 1}]
        self.assertEqual(FarmAgent._cargo_after_actions(obs, [["FERTILIZE"]], {})["FERTILIZER"], 1)
        self.assertEqual(FarmAgent._cargo_after_actions(obs, [["COLLECT_FERTILIZER"]], {})["FERTILIZER"], 3)
        obs["farms"][0]["tiles"][4][4] = {"kind": "PASTURE"}
        self.assertEqual(FarmAgent._cargo_after_actions(obs, [["PLACE", "COW"]], {})["COW"], 0)

    def test_late_animal_purchase_reserves_room_for_returning_workers(self):
        obs = observation(23)
        obs["farms"][0]["money"] = 10000
        obs["private"]["inventories"] = [{"CARROT": 98}]
        bot = FarmAgent(PolicyConfig(use_forecast=False, enable_expansion=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"COW": 3})
        orders, _ = bot._market_orders(obs, game_configuration(), {}, 0, {"CARROT": 98})
        self.assertEqual(sum(order[2] for order in orders if order[0] == "BUY_ANIMAL"), 2)


if __name__ == "__main__":
    unittest.main()
