"""Regression checks against the vendored official engine and real framework."""
import copy
import unittest

from scripts.benchmark import load_engine, make_environment

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class EngineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine()

    def setUp(self):
        self.env = make_environment(7, 720, {"weedSpawnChance": 0})

    def act(self, farmer=None, hands=None, market=None):
        self.env.step([{"farmer": farmer or ["PASS"],
                        "hands": hands or [], "market": market or []}, PASS])

    @property
    def observation(self):
        return self.env.state[0].observation

    def test_framework_final_action_is_episode_steps_minus_two(self):
        seen = []
        def passive(obs, config):
            seen.append(obs["step"])
            return PASS
        self.env.run([passive, "pass"])
        self.assertEqual(len(self.env.steps), 720)
        self.assertEqual(seen, list(range(719)))
        self.assertEqual(self.env.state[0].observation.step, 719)
        self.assertEqual([p.status for p in self.env.state], ["DONE", "DONE"])
        self.assertEqual(self.env.state[0].reward, 3000)

    def test_seed_is_recorded_but_hidden_from_agent_configuration(self):
        self.assertEqual(self.env.info["seed"], 7)
        self.assertIsNone(self.env.configuration["seed"])
        other = make_environment(7, 80)
        repeated = make_environment(7, 80)
        other.run(["starter", "pass"])
        repeated.run(["starter", "pass"])
        self.assertEqual(other.steps, repeated.steps)

    def test_buying_seed_then_planting_needs_two_decisions(self):
        self.act(["PLANT", "WHEAT"], market=[["BUY_SEED", "WHEAT", 1]])
        self.assertIsNone(self.observation.farms[0]["tiles"][4][4])
        self.act(["PLANT", "WHEAT"])
        self.assertEqual(self.observation.farms[0]["tiles"][4][4]["crop"], "WHEAT")

    def test_new_seed_dies_first_night_without_water(self):
        self.act(market=[["BUY_SEED", "WHEAT", 1]])
        self.act(["PLANT", "WHEAT"])
        while self.observation.step < 24:
            self.act()
        self.assertEqual(self.observation.farms[0]["tiles"][4][4], {"kind": "WEED"})

    def test_atomic_overbooked_seeds_block_all_plants(self):
        self.act(market=[["BUY_SEED", "WHEAT", 1], ["HIRE"]])
        self.act(["PLANT", "WHEAT"], hands=[["PLANT", "WHEAT"]])
        self.assertIsNone(self.observation.farms[0]["tiles"][4][4])
        self.assertEqual(self.observation.private["seeds"]["WHEAT"], 1)

    def test_locked_tiles_are_walkable_and_shed_accessible(self):
        self.act(["EAST"])
        self.assertEqual(self.observation.farms[0]["farmer"], [5, 4])
        self.assertEqual(self.observation.farms[0]["tiles"][4][5], "LOCKED")
        self.observation.private["inventories"][0]["WHEAT"] = 2
        self.act(["PLACE", "WHEAT", 2])
        self.assertEqual(self.observation.private["shed"]["WHEAT"], 2)

    def test_sell_uses_shed_and_place_can_sell_in_same_turn(self):
        self.observation.private["inventories"][0]["WHEAT"] = 2
        self.act(market=[["SELL", "WHEAT", 2]])
        self.assertEqual(self.observation.farms[0]["money"], 3000)
        self.act(["PLACE", "WHEAT", 2], market=[["SELL", "WHEAT", 2]])
        self.assertGreater(self.observation.farms[0]["money"], 3000)
        self.assertEqual(self.observation.private["shed"].get("WHEAT", 0), 0)

    def test_place_keeps_overflow_but_drop_discards_it(self):
        self.observation.private["shed"]["CARROT"] = 99
        self.observation.private["inventories"][0]["WHEAT"] = 3
        self.act(["PLACE", "WHEAT", 3])
        self.assertEqual(self.observation.private["shed"]["WHEAT"], 1)
        self.assertEqual(self.observation.private["inventories"][0]["WHEAT"], 2)
        self.act(["DROP"])
        self.assertEqual(self.observation.private["inventories"][0], {})
        self.assertEqual(sum(self.observation.private["shed"].values()), 100)

    def test_end_of_day_deposit_occurs_after_market_orders(self):
        while self.observation.step < 23:
            self.act()
        self.observation.private["inventories"][0]["WHEAT"] = 2
        self.act(market=[["SELL", "WHEAT", 2]])
        self.assertEqual(self.observation.farms[0]["money"], 3000)
        self.assertEqual(self.observation.private["shed"]["WHEAT"], 2)
        self.act(market=[["SELL", "WHEAT", 2]])
        self.assertGreater(self.observation.farms[0]["money"], 3000)

    def test_melon_yield_before_maturity_cannot_be_harvested(self):
        farm = self.observation.farms[0]
        private = self.observation.private
        farm["tiles"][4][4] = self.engine._new_plant("MELON", 0, 24)
        self.engine._apply_unit_action(farm, private, 0, ["WATER"], 10, 6, 24)
        self.assertEqual(farm["tiles"][4][4]["yield_units"], 2)
        self.engine._apply_unit_action(farm, private, 0, ["HARVEST"], 10, 6, 24)
        self.assertEqual(private["inventories"][0], {})
        self.engine._apply_unit_action(farm, private, 0, ["HARVEST"], 10, 10, 24)
        self.assertEqual(private["inventories"][0]["MELON"], 2)

    def test_ongoing_has_four_production_events_not_infinite_lifetime(self):
        for crop, first, interval in [("TOMATO", 8, 1), ("STRAWBERRY", 10, 2)]:
            farm = self.observation.farms[0]
            tile = self.engine._new_plant(crop, 0, 24)
            farm["tiles"][4][4] = tile
            produced = []
            for day in range(first + 3 * interval + 2):
                tile["watered_today"] = True
                tile["fertilized_until_day"] = day
                before = tile["yield_units"]
                self.engine._daily_refresh_plants(farm, day, 24)
                if tile["yield_units"] > before:
                    produced.append(day + 1)
                    self.assertEqual(tile["yield_units"], 2)
                    tile["yield_units"] = 0  # Model collecting each production batch.
            self.assertEqual(produced, [first + i * interval for i in range(4)])
            self.assertEqual(tile["max_lifespan_step"], (first + 3 * interval + 1) * 24)


if __name__ == "__main__":
    unittest.main()
