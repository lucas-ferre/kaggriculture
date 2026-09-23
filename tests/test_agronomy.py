"""Marginal fertilizer and animal calendars checked against actual transitions."""
import copy
from types import SimpleNamespace
import unittest

from kaggriculture_agent.agronomy import (
    animal_care_can_pay, animal_production_days, fertilizer_candidates,
    fertilizer_reserve, nightly_animal_production, projected_crop_harvests)
from kaggriculture_agent.domain import CROPS, game_configuration
from scripts.benchmark import load_engine, make_environment


class AgronomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine()

    def setUp(self):
        self.config = game_configuration()
        self.policy = SimpleNamespace(enable_fertilizer=True, labor_cost_per_action=.6)

    def engine_harvests(self, tile, day, last_day, fertilize):
        farm = {"tiles": [[None for _ in range(10)] for _ in range(10)],
                "farmer": [4, 4], "hands": []}
        private = {"shed": {}, "seeds": {}, "inventories": [{"FERTILIZER": 1}]}
        farm["tiles"][4][4] = copy.deepcopy(tile)
        if fertilize:
            self.engine._apply_unit_action(farm, private, 0, ["FERTILIZE"], 10, day, 24)
        result = []
        for current in range(day, last_day + 1):
            plant = farm["tiles"][4][4]
            if plant is None:
                break
            rule = CROPS[plant["crop"]]
            self.engine._apply_unit_action(farm, private, 0, ["WATER"], 10, current, 24)
            age = current - plant["planted_day"]
            if age >= rule.first and plant["yield_units"] and (
                    current == last_day or not rule.ongoing and age >= rule.peak
                    or rule.ongoing and (plant["yield_units"] >= 2 or age >= rule.last)):
                quantity = plant["yield_units"]
                self.engine._apply_unit_action(farm, private, 0, ["HARVEST"], 10, current, 24)
                result.append((current, quantity))
            self.engine._daily_refresh_plants(farm, current, 24)
        return result

    def test_forward_harvests_match_engine_at_growth_and_production_boundaries(self):
        for crop, rule in CROPS.items():
            for day in sorted({0, rule.first - 1, rule.first, rule.last - 1, rule.last}):
                for units in (1, 3):
                    for watered in (False, True):
                        for fertilize in (False, True):
                            tile = self.engine._new_plant(crop, 0, 24)
                            tile.update(yield_units=units, watered_today=watered)
                            with self.subTest(crop=crop, day=day, units=units,
                                              watered=watered, fertilize=fertilize):
                                self.assertEqual(projected_crop_harvests(tile, day, 29, fertilize),
                                                 self.engine_harvests(tile, day, 29, fertilize))

    def test_well_watered_melon_has_no_imaginary_extra_units(self):
        tile = self.engine._new_plant("MELON", 0, 24)
        baseline = projected_crop_harvests(tile, 6, 29)
        fertilized = projected_crop_harvests(tile, 6, 29, True)
        self.assertEqual(baseline, [(10, 6)])
        self.assertEqual(fertilized, baseline)

    def test_existing_fertilizer_and_water_are_not_applied_twice(self):
        tile = self.engine._new_plant("WHEAT", 0, 24)
        tile.update(yield_units=3, watered_today=True, fertilized_until_day=4)
        self.assertEqual(projected_crop_harvests(tile, 2, 29, True), [(4, 6)])
        self.assertEqual(projected_crop_harvests(tile, 2, 29), [(4, 6)])

    def test_recurring_fertilizer_obeys_calendar_and_three_day_duration(self):
        tile = self.engine._new_plant("TOMATO", 0, 24)
        self.assertEqual(sum(n for _, n in projected_crop_harvests(tile, 7, 29)), 4)
        self.assertEqual(sum(n for _, n in projected_crop_harvests(tile, 7, 29, True)), 7)
        strawberry = self.engine._new_plant("STRAWBERRY", 0, 24)
        self.assertEqual(sum(n for _, n in projected_crop_harvests(strawberry, 9, 29, True)), 6)

    def test_candidate_values_sell_opportunity_and_reserves_predicted_stock(self):
        obs = make_environment(9).state[0].observation
        obs.update(step=7 * 24, day=7, hour=0)
        obs.farms[0]["tiles"][4][4] = self.engine._new_plant("TOMATO", 0, 24)
        obs.market["prices"].update(TOMATO=100, FERTILIZER=20)
        obs.private["shed"]["FERTILIZER"] = 4
        before = copy.deepcopy(obs)
        candidates = fertilizer_candidates(obs, self.config, self.policy)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].extra_units, 3)
        self.assertAlmostEqual(candidates[0].net_gain, 278.8)
        self.assertEqual(fertilizer_reserve(obs, self.config, self.policy), 1)
        self.assertEqual(obs, before)
        obs.private["inventories"][0]["FERTILIZER"] = 1
        self.assertEqual(fertilizer_reserve(obs, self.config, self.policy), 0)
        self.assertEqual(fertilizer_reserve(obs, self.config, self.policy,
                                          {"FERTILIZER": 5}, predicted_carried_fertilizer=0), 1)
        obs.market["prices"]["FERTILIZER"] = 400
        self.assertEqual(fertilizer_candidates(obs, self.config, self.policy), [])

    def test_fertilizer_requires_reachable_same_day_water_slot(self):
        obs = make_environment(9).state[0].observation
        obs.update(step=7 * 24 + 23, day=7, hour=23)
        obs.farms[0]["tiles"][4][4] = self.engine._new_plant("TOMATO", 0, 24)
        obs.private["inventories"][0]["FERTILIZER"] = 1
        obs.market["prices"].update(TOMATO=100, FERTILIZER=1)
        self.assertEqual(fertilizer_candidates(obs, self.config, self.policy), [])
        obs.farms[0]["tiles"][4][4]["watered_today"] = True
        self.assertEqual(len(fertilizer_candidates(obs, self.config, self.policy)), 1)

    def test_sheep_placement_calendar_includes_real_final_payout(self):
        self.assertEqual(animal_production_days("SHEEP", 11, 29), [17, 20, 23, 26, 29])
        self.assertEqual(animal_production_days("SHEEP", 12, 29), [18, 21, 24, 27])

    def test_care_is_banked_after_payout_and_skipped_when_terminal(self):
        tile = self.engine._new_animal("SHEEP", 11)
        tile.update(fed_today=True, cared_today=True, pending_care_bonus=2)
        self.assertEqual(nightly_animal_production(tile, 28), 3)
        self.assertTrue(animal_care_can_pay(tile, 27, 29))
        self.assertFalse(animal_care_can_pay(tile, 28, 29))
        farm = {"tiles": [[tile]]}
        self.engine._daily_refresh_animals(farm, 28)
        self.assertEqual(tile["yield_units"], 3)
        self.assertEqual(tile["pending_care_bonus"], 1)

    def test_nightly_cap_and_starvation_lose_old_bonus_exactly(self):
        tile = self.engine._new_animal("COW", 0)
        tile.update(yield_units=5, pending_care_bonus=3, fed_today=True, cared_today=True)
        self.assertEqual(nightly_animal_production(tile, 9), 4)
        self.engine._daily_refresh_animals({"tiles": [[tile]]}, 9)
        self.assertEqual(tile["yield_units"], 6)
        tile.update(yield_units=0, pending_care_bonus=3, fed_today=False)
        self.assertEqual(nightly_animal_production(tile, 11, False), 1)
        self.engine._daily_refresh_animals({"tiles": [[tile]]}, 11)
        self.assertEqual(tile["yield_units"], 1)
        self.assertEqual(tile["pending_care_bonus"], 0)


if __name__ == "__main__":
    unittest.main()
