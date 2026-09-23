"""Economic land decisions must use productive capacity, not bank balance."""
import unittest

from kaggriculture_agent.agent import FarmAgent
from kaggriculture_agent.domain import game_configuration
from kaggriculture_agent.economy import expansion_plan, livestock_plan, prioritize_market_orders
from kaggriculture_agent.market import MarketModel
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from tests.test_agent import observation


def expanded_observation(step=120, quadrants=("NW", "NE")):
    obs = observation(step)
    farm = obs["farms"][0]
    farm["money"] = 100000
    farm["unlocked_quadrants"] = list(quadrants)
    for y, row in enumerate(farm["tiles"]):
        for x in range(len(row)):
            name = ("N" if y < 5 else "S") + ("W" if x < 5 else "E")
            if name in quadrants:
                row[x] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 0, "watered_today": False} if (x + y) % 4 != 0 else None
            else:
                row[x] = "LOCKED"
    return obs


def productive_plan(score=1200):
    return CropPlan("WHEAT", "TOMATO", 8, 10,
                    scores={"WHEAT": score, "TOMATO": score},
                    animal_targets={"COW": 8, "SHEEP": 6})


class ExpansionTests(unittest.TestCase):
    def test_second_purchase_values_compact_work_area(self):
        obs = expanded_observation()
        buy, debug = expansion_plan(obs, game_configuration(), PolicyConfig(), productive_plan(), 99000)
        self.assertTrue(buy)
        self.assertEqual(debug["next_quadrant"], "SW")
        self.assertEqual(debug["cost"], 2000)
        self.assertEqual(debug["compact_owned"], 20)
        self.assertGreater(debug["projected_margin"], debug["cost"])

    def test_can_reconsider_each_successive_quadrant(self):
        # A larger feasible workforce can justify the final quadrant too.
        policy = PolicyConfig(target_workers=12, max_plots=60, max_expansions=3)
        for count, expected, cost in ((1, "NE", 1000), (2, "SW", 2000), (3, "SE", 4000)):
            obs = expanded_observation(quadrants=("NW", "NE", "SW")[:count])
            buy, debug = expansion_plan(obs, game_configuration(), policy, productive_plan(), 99000)
            self.assertTrue(buy)
            self.assertEqual((debug["next_quadrant"], debug["cost"]), (expected, cost))

    def test_rich_farm_still_preserves_cash_without_return_or_time(self):
        for step, plan in ((120, productive_plan(score=1)), (690, productive_plan())):
            obs = expanded_observation(step)
            buy, debug = expansion_plan(obs, game_configuration(), PolicyConfig(), plan, 99000)
            self.assertFalse(buy)
            self.assertIn(debug["status"], {"insufficient_projected_return", "no_recoverable_production"})

    def test_land_does_not_consume_operating_buffer_or_exceed_labor(self):
        obs = expanded_observation()
        buy, debug = expansion_plan(obs, game_configuration(), PolicyConfig(), productive_plan(), 2000)
        self.assertFalse(buy)
        self.assertEqual(debug["status"], "operating_cash_reserved")
        buy, debug = expansion_plan(obs, game_configuration(), PolicyConfig(target_workers=2),
                                    productive_plan(), 99000)
        self.assertFalse(buy)
        self.assertEqual(debug["status"], "no_spare_crop_labor")

    def test_crossing_locked_area_is_not_purchase_signal(self):
        obs = expanded_observation()
        obs["farms"][0]["farmer"] = [4, 5]
        plan = CropPlan(None, None, 0, 0)
        buy, _ = expansion_plan(obs, game_configuration(), PolicyConfig(), plan, 99000)
        self.assertFalse(buy)

    def test_land_priority_follows_feed_sales_and_hiring(self):
        obs = observation()
        orders = [["BUY_ANIMAL", "SHEEP", 1], ["BUY_SEED", "TOMATO", 2],
                  ["BUY_LAND"], ["HIRE"], ["BUY_PRODUCT", "WHEAT", 1], ["SELL", "MILK", 2]]
        ranked = prioritize_market_orders(orders, obs, game_configuration(), feed_urgent=True)
        names = [order[0] for order in ranked]
        self.assertLess(names.index("BUY_PRODUCT"), names.index("BUY_LAND"))
        self.assertLess(names.index("HIRE"), names.index("BUY_LAND"))
        self.assertLess(names.index("BUY_LAND"), names.index("BUY_SEED"))
        self.assertLess(names.index("BUY_LAND"), names.index("BUY_ANIMAL"))

    def test_land_integration_and_replan_after_purchase(self):
        obs = expanded_observation()
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = productive_plan()
        orders, _ = bot._market_orders(obs, game_configuration(), {})
        self.assertIn(["BUY_LAND"], orders)
        bot(observation())
        old_plan = bot.plan
        bot(expanded_observation(1))
        self.assertIsNot(bot.plan, old_plan)
        self.assertEqual(bot.last_land_count, 2)

    def test_initial_land_does_not_spend_committed_animal_cash(self):
        obs = observation()
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = CropPlan("WHEAT", "TOMATO", 8, 10,
                            scores={"WHEAT": 1200, "TOMATO": 1200},
                            animal_targets={"COW": 3})
        orders, _ = bot._market_orders(obs, game_configuration(), {})
        self.assertTrue(any(order[0] == "BUY_ANIMAL" for order in orders))
        self.assertNotIn(["BUY_LAND"], orders)
        self.assertEqual(bot.expansion_debug["status"], "operating_cash_reserved")

    def test_existing_herd_can_use_reserve_for_emergency_feed(self):
        obs = observation(18)
        farm = obs["farms"][0]
        farm["money"] = 300
        farm["tiles"][4][4] = {"kind": "PASTURE", "animal": "SHEEP",
                                 "fed_today": False, "consecutive_unfed": 1}
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = CropPlan(None, None, 0, 0, animal_targets={"SHEEP": 1})
        orders, _ = bot._market_orders(obs, game_configuration(), {})
        self.assertTrue(any(order[0] == "BUY_PRODUCT" and order[1] == "WHEAT" for order in orders))
        self.assertFalse(any(order[0] in {"BUY_LAND", "BUY_ANIMAL", "BUY_SEED"} for order in orders))

    def test_six_sheep_are_planned_and_owned_animals_count_towards_limit(self):
        policy = PolicyConfig(max_cows=0, max_geese=0)
        self.assertEqual(policy.max_sheep, 6)
        obs = expanded_observation(step=24)
        for x in range(4):
            obs["farms"][0]["tiles"][4][x] = {"kind": "PASTURE", "animal": "SHEEP"}
        model = MarketModel(game_configuration())
        model.update(obs)
        targets, _ = livestock_plan(obs, game_configuration(), policy, model)
        self.assertEqual(targets["SHEEP"], 6)

    def test_dynamic_workforce_ceiling_scales_with_unlocked_quadrants(self):
        policy = PolicyConfig(target_workers=10)
        bot = FarmAgent(policy)
        obs_two = expanded_observation(step=0, quadrants=("NW", "NE"))
        bot.plan = CropPlan("WHEAT", "TOMATO", 20, 20, workforce_target=10)
        orders_two, _ = bot._market_orders(obs_two, game_configuration(), {})
        self.assertLessEqual(sum(1 for o in orders_two if o[0] == "HIRE"), 7)

        obs_three = expanded_observation(step=0, quadrants=("NW", "NE", "SW"))
        bot.plan = CropPlan("WHEAT", "TOMATO", 30, 30, workforce_target=10)
        orders_three, _ = bot._market_orders(obs_three, game_configuration(), {})
        self.assertLessEqual(sum(1 for o in orders_three if o[0] == "HIRE"), 8)

        obs_four = expanded_observation(step=0, quadrants=("NW", "NE", "SW", "SE"))
        bot.plan = CropPlan("WHEAT", "TOMATO", 40, 40, workforce_target=10)
        orders_four, _ = bot._market_orders(obs_four, game_configuration(), {})
        self.assertLessEqual(sum(1 for o in orders_four if o[0] == "HIRE"), 9)

    def test_expansion_pacing_blocks_early_and_unoccupied(self):
        obs_early = observation(step=24)
        buy, debug = expansion_plan(obs_early, game_configuration(), PolicyConfig(), productive_plan(), 99000)
        self.assertFalse(buy)
        self.assertIn(debug["status"], {"early_game_restricted", "insufficient_occupancy", "operating_cash_reserved"})

        obs_empty = expanded_observation(step=120)
        for y, row in enumerate(obs_empty["farms"][0]["tiles"]):
            for x in range(len(row)):
                if row[x] != "LOCKED":
                    row[x] = None
        buy, debug = expansion_plan(obs_empty, game_configuration(), PolicyConfig(), productive_plan(), 99000)
        self.assertFalse(buy)
        self.assertEqual(debug["status"], "insufficient_occupancy")


if __name__ == "__main__":
    unittest.main()
