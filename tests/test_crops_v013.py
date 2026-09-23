"""Finite recurrent production, partial payback and bounded preferences."""
from dataclasses import replace
import unittest

from kaggriculture_agent.planner import CropPlanner, PolicyConfig, minimum_cycle_day
from tests.test_planner import ForecastMarket, observation


def recurrent_market(crop="STRAWBERRY", price=120):
    return ForecastMarket({name: price if name == crop else 1
                           for name in ForecastMarket().prices})


class RecurrentInvestmentTests(unittest.TestCase):
    def test_strawberry_counts_profitable_second_partial_cycle(self):
        market = recurrent_market()
        policy = PolicyConfig(max_plots=1)
        improved = CropPlanner(policy).plan(observation(), {}, market)
        full_only = CropPlanner(replace(policy, min_recurrent_pulses=4)).plan(
            observation(), {}, recurrent_market())
        batches = [(h, q) for crop, h, q in market.calls if crop == "STRAWBERRY"]
        # Four finite pulses in the first planting and two in the second;
        # the scheduler actually collects pairs on days 12, 16 and 29.
        self.assertEqual(batches, [(289, 2), (385, 2), (697, 2)])
        self.assertEqual(sum(q for _, q in batches), 6)
        self.assertGreater(improved.scores["STRAWBERRY"], full_only.scores["STRAWBERRY"])
        self.assertEqual(improved.targets, {"STRAWBERRY": 1})

    def test_late_recurrent_investment_requires_payback_and_two_real_pulses(self):
        policy = PolicyConfig(max_plots=4)
        viable = CropPlanner(policy).plan(observation(day=17), {}, recurrent_market())
        too_late = CropPlanner(policy).plan(observation(day=18), {}, recurrent_market())
        losing = CropPlanner(policy).plan(observation(day=17), {}, recurrent_market(price=1))
        full_only = CropPlanner(replace(policy, min_recurrent_pulses=4)).plan(
            observation(day=17), {}, recurrent_market())
        self.assertEqual(viable.targets, {"STRAWBERRY": 4})
        self.assertEqual(too_late.targets, {})
        self.assertEqual(losing.targets, {})
        self.assertEqual(full_only.targets, {})
        self.assertEqual(minimum_cycle_day("STRAWBERRY", policy), 12)
        self.assertEqual(minimum_cycle_day("TOMATO", policy), 9)
        self.assertEqual(minimum_cycle_day("MELON", policy), 10)

    def test_preference_can_choose_recurrent_crop_without_forcing_a_loss(self):
        # At this late viable date carrot and tomato have comparable returns.
        # The preference only changes a close choice with a positive margin.
        obs = observation(day=20)
        prices = {"WHEAT": 1, "CARROT": 19, "TOMATO": 60,
                  "STRAWBERRY": 1, "MELON": 1}
        policy = PolicyConfig(max_plots=1)
        preferred = CropPlanner(policy).plan(obs, {}, ForecastMarket(prices))
        neutral = CropPlanner(replace(policy, ongoing_preference=0)).plan(obs, {}, ForecastMarket(prices))
        self.assertEqual(preferred.targets, {"TOMATO": 1})
        self.assertEqual(neutral.targets, {"CARROT": 1})
        losing = CropPlanner(policy).plan(obs, {}, recurrent_market("TOMATO", 1))
        self.assertEqual(losing.targets, {})

    def test_pair_collection_respects_real_batch_storage(self):
        plan = CropPlanner(PolicyConfig(max_plots=25)).plan(
            observation(), {"shedCapacity": 3}, recurrent_market())
        # Each plot deposits two units, so three empty slots support one plot.
        self.assertEqual(plan.targets, {"STRAWBERRY": 1})

    def test_mature_recurrent_plants_survive_quote_and_terminal_changes(self):
        obs = observation(day=28, plants=("TOMATO", "STRAWBERRY"))
        for tile in obs["farms"][0]["tiles"][0][:2]:
            tile.update(planted_day=13, yield_units=2)
        plan = CropPlanner().plan(obs, {}, recurrent_market(price=1))
        self.assertEqual(plan.targets, {"TOMATO": 1, "STRAWBERRY": 1})

    def test_recurrent_ablation_configuration_and_animal_caps(self):
        self.assertEqual(PolicyConfig().max_sheep, 6)
        self.assertEqual(PolicyConfig().max_expansions, 3)
        for value in (0, 1, 5, 2.5):
            with self.assertRaises(ValueError):
                PolicyConfig(min_recurrent_pulses=value)


if __name__ == "__main__":
    unittest.main()
