"""Execution regressions: resource reservation, first-night care and cash-out."""
import copy
import unittest

from kaggriculture_agent.agent import FarmAgent
from kaggriculture_agent.domain import game_configuration
from kaggriculture_agent.market import MARKET_PARAMS
from kaggriculture_agent.planner import CropPlan, PolicyConfig
from kaggriculture_agent.scheduler import TaskScheduler


def observation(step=0):
    farm = {"tiles": [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)]
                      for y in range(10)], "farmer": [4, 4], "hands": [],
            "money": 3000., "hires_today": 0, "unlocked_quadrants": ["NW"]}
    return {"step": step, "day": step // 24, "hour": step % 24, "player": 0,
            "farms": [farm, copy.deepcopy(farm)], "town": {"unlocked_shops": []},
            "market": {"inventory": {p: 10000 for p in MARKET_PARAMS},
                       "prices": {p: params["base"] for p, params in MARKET_PARAMS.items()}},
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}


class ExecutionTests(unittest.TestCase):
    def test_shared_seed_reserved_before_parallel_planting(self):
        obs = observation(2)
        obs["farms"][0]["hands"] = [[4, 4], [4, 4]]
        obs["private"]["inventories"] = [{}, {}, {}]
        obs["private"]["seeds"] = {"WHEAT": 1}
        actions, _, _ = TaskScheduler(PolicyConfig()).actions(
            obs, game_configuration(), CropPlan("WHEAT", None, 3, 0))
        self.assertEqual(sum(a[0] == "PLANT" for a in actions), 1)

    def test_first_night_water_is_not_postponed(self):
        obs = observation(23)
        obs["farms"][0]["tiles"][4][4] = {
            "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "yield_units": 1, "watered_today": False, "consecutive_unwatered": 1,
            "max_lifespan_step": 120, "fertilized_until_day": -1}
        actions, _, _ = TaskScheduler(PolicyConfig()).actions(
            obs, game_configuration(), CropPlan("WHEAT", None, 1, 0))
        self.assertEqual(actions[0], ["WATER"])

    def test_no_plant_without_same_day_water_slot(self):
        obs = observation(23)
        obs["private"]["seeds"] = {"WHEAT": 2}
        actions, _, _ = TaskScheduler(PolicyConfig()).actions(
            obs, game_configuration(), CropPlan("WHEAT", None, 2, 0))
        self.assertNotIn("PLANT", [a[0] for a in actions])

    def test_final_action_deposits_and_sells_in_same_turn_without_mutation(self):
        obs = observation(718)
        obs["private"]["inventories"] = [{"WHEAT": 3}]
        before = copy.deepcopy(obs)
        action = FarmAgent(PolicyConfig(max_plots=0, use_forecast=False))(obs)
        self.assertEqual(action["farmer"], ["DROP"])
        self.assertIn(["SELL", "WHEAT", 3], action["market"])
        self.assertEqual(obs, before)

    def test_near_full_shed_uses_partial_place_instead_of_lossy_drop(self):
        obs = observation(200)
        obs["private"]["shed"] = {"CARROT": 95}
        obs["private"]["inventories"] = [{"WHEAT": 9}]
        actions, predicted, _ = TaskScheduler(PolicyConfig()).actions(
            obs, game_configuration(), CropPlan(None, None, 0, 0))
        self.assertEqual(actions[0], ["PLACE", "WHEAT", 5])
        self.assertEqual(sum(predicted.values()), 100)

    def test_stale_target_does_not_buy_unrecoverable_late_seed(self):
        obs = observation(673)  # day 28, next wheat matures after the game ends
        bot = FarmAgent(PolicyConfig(use_forecast=False))
        bot.plan = CropPlan("WHEAT", None, 1, 0)
        orders, _ = bot._market_orders(obs, game_configuration(), {})
        self.assertFalse(any(order[0] == "BUY_SEED" for order in orders))

    def test_episode_restart_resets_memory_and_reproduces_first_action(self):
        bot = FarmAgent(PolicyConfig(max_plots=4, use_forecast=False))
        obs = observation()
        first = bot(obs)
        bot(observation(1))
        repeated = bot(observation())
        self.assertEqual(first, repeated)
        self.assertEqual(bot.last_step, 0)

    def test_final_day_liquidates_entire_shed_including_feed_and_fertilizer(self):
        bot = FarmAgent()
        bot.plan = CropPlan(None, None, 0, 0)
        obs_final = observation(step=719)
        obs_final["day"] = 29
        obs_final["hour"] = 23
        obs_final["private"]["shed"] = {"WHEAT": 20, "FERTILIZER": 15, "MELON": 5}
        obs_final["farms"][0]["money"] = 1000
        orders, _ = bot._market_orders(obs_final, game_configuration(), obs_final["private"]["shed"])
        sold = {o[1]: o[2] for o in orders if o[0] == "SELL"}
        self.assertEqual(sold.get("WHEAT"), 20)
        self.assertEqual(sold.get("FERTILIZER"), 15)
        self.assertEqual(sold.get("MELON"), 5)


if __name__ == "__main__":
    unittest.main()
