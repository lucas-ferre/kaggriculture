"""Behavioral checks for terminal investment, market impact and assignments."""

import itertools
import math
import random
import unittest

from kaggriculture_agent.optimization import assign, optimize_allocation
from kaggriculture_agent.planner import CropPlanner, PolicyConfig, full_cycle_day, yield_without_fertilizer


class ForecastMarket:
    def __init__(self, prices=None, impact=0.0, risk=0.0):
        self.prices = prices or {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250}
        self.impact = impact
        self.risk = risk
        self.calls = []

    def forecast(self, crop, horizon_turns, sale_quantity):
        self.calls.append((crop, horizon_turns, sale_quantity))
        quote = max(1.0, self.prices[crop] - self.impact * sale_quantity)
        return {"expected_price": quote, "expected_revenue": quote * sale_quantity,
                "p10": max(1.0, quote * (1 - self.risk)), "p50": quote, "p90": quote * 1.1}


def observation(day=0, money=3000, plants=()):
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    for index, crop in enumerate(plants):
        tiles[index // 5][index % 5] = {"kind": "PLANT", "crop": crop,
                                       "planted_day": max(0, day - 3), "yield_units": 0}
    return {"player": 0, "day": day, "hour": 0, "step": day * 24,
            "farms": [{"tiles": tiles, "money": money, "hands": [], "farmer": [4, 4]}],
            "private": {"seeds": {}}, "market": {"prices": ForecastMarket().prices}}


class AssignmentTests(unittest.TestCase):
    def test_hungarian_known_optimum_beats_greedy(self):
        matrix = [[1, 2], [2, 100]]
        exact = assign(matrix, "hungarian")
        self.assertEqual(exact, [(0, 1), (1, 0)])
        self.assertEqual(sum(matrix[i][j] for i, j in exact), 4)
        self.assertEqual(sum(matrix[i][j] for i, j in assign(matrix)), 101)

    def test_rectangular_and_infeasible(self):
        self.assertEqual(assign([[4, 1, 4], [2, 0, 5]], "hungarian"), [(0, 1), (1, 0)])
        self.assertEqual(assign([[1], [0], [2]], "hungarian"), [(1, 0)])
        self.assertEqual(assign([[math.inf, math.nan], [math.inf, 3]], "hungarian"), [(1, 1)])
        self.assertEqual(assign([[math.inf], [math.inf]], "hungarian"), [])
        self.assertEqual(assign([[], []], "hungarian"), [])
        self.assertEqual(assign([], "hungarian"), [])

    def test_feasible_cardinality_is_first_priority(self):
        self.assertEqual(assign([[0, 1000], [1, math.inf]], "hungarian"), [(0, 1), (1, 0)])

    def test_exact_matches_bruteforce_rectangular_negative_costs(self):
        rng = random.Random(4102)
        for _ in range(20):
            matrix = [[rng.randint(-20, 40) for _ in range(4)] for _ in range(3)]
            actual = sum(matrix[i][j] for i, j in assign(matrix, "hungarian"))
            expected = min(sum(matrix[i][j] for i, j in enumerate(choice))
                           for choice in itertools.permutations(range(4), 3))
            self.assertEqual(actual, expected)

    def test_rejects_ragged_matrix_and_unknown_method(self):
        with self.assertRaises(ValueError):
            assign([[1], [1, 2]])
        with self.assertRaises(ValueError):
            assign([[1]], "mystery")

    def test_allocation_methods_and_unused_land(self):
        score = lambda s, l: 100 - (s - 3) ** 2 - (l - 2) ** 2
        for method in ("enumerate", "beam"):
            self.assertEqual(optimize_allocation(10, score, method), (3, 2))
        self.assertEqual(optimize_allocation(3, lambda s, l: -s - l), (0, 0))


def legacy_policy(**overrides):
    settings = dict(strategy="dual", optimizer="enumerate", max_plots=20,
                    target_workers=6, enable_livestock=False, enable_expansion=False)
    settings.update(overrides)
    return PolicyConfig(**settings)


class CropPlannerTests(unittest.TestCase):
    def test_unfertilized_yields_and_finite_recurrent_cycles(self):
        self.assertEqual(yield_without_fertilizer("WHEAT", 4), 4)
        self.assertEqual(yield_without_fertilizer("CARROT", 3), 3)
        self.assertEqual(yield_without_fertilizer("MELON", 10), 6)
        self.assertEqual(full_cycle_day("TOMATO"), 11)
        self.assertEqual(full_cycle_day("STRAWBERRY"), 16)
        self.assertEqual(full_cycle_day("MELON"), 10)
        self.assertEqual(yield_without_fertilizer("TOMATO", 29), 4)
        self.assertEqual(yield_without_fertilizer("STRAWBERRY", 29), 4)

    def test_legacy_opens_two_profitable_cycles(self):
        market = ForecastMarket()
        plan = CropPlanner(legacy_policy()).plan(observation(), {}, market)
        self.assertIn(plan.short_crop, ("WHEAT", "CARROT"))
        self.assertIn(plan.long_crop, ("TOMATO", "STRAWBERRY"))
        self.assertGreater(plan.short_target, 0)
        self.assertGreater(plan.long_target, 0)
        self.assertLessEqual(sum(plan.targets.values()), 20)
        self.assertTrue(any(quantity > 6 for _, _, quantity in market.calls))
        self.assertTrue(all(horizon < 718 - 12 for _, horizon, _ in market.calls))

    def test_long_mode_allows_comparison_with_single_harvest_melon(self):
        market = ForecastMarket()
        recurring = CropPlanner(legacy_policy()).plan(observation(), {}, market)
        unrestricted = CropPlanner(legacy_policy(long_mode="any")).plan(observation(), {}, market)
        self.assertIn(recurring.long_crop, ("TOMATO", "STRAWBERRY"))
        self.assertEqual(unrestricted.long_crop, "MELON")
        with self.assertRaises(ValueError):
            legacy_policy(long_mode="unknown")

    def test_recurrent_mode_preserves_existing_melon_area_without_replanting_it(self):
        obs = observation(plants=("MELON",) * 4)
        before = [list(row) for row in obs["farms"][0]["tiles"]]
        plan = CropPlanner(legacy_policy()).plan(obs, {}, ForecastMarket())
        self.assertIn(plan.long_crop, ("TOMATO", "STRAWBERRY"))
        self.assertLessEqual(sum(plan.targets.values()) + 4, 20)
        self.assertNotIn("MELON", plan.targets)
        self.assertEqual(obs["farms"][0]["tiles"], before)

    def test_recurrent_mode_does_not_substitute_melon_when_recurring_crops_lose_money(self):
        market = ForecastMarket({"WHEAT": 25, "CARROT": 35, "TOMATO": 1,
                                 "STRAWBERRY": 1, "MELON": 250})
        plan = CropPlanner(legacy_policy()).plan(observation(), {}, market)
        self.assertGreater(plan.short_target, 0)
        self.assertEqual(plan.long_target, 0)

    def test_terminal_window_and_late_fast_only(self):
        late = CropPlanner(legacy_policy()).plan(observation(day=26), {}, ForecastMarket())
        self.assertGreater(late.short_target, 0)
        self.assertEqual(late.long_target, 0)
        terminal = CropPlanner(legacy_policy()).plan(observation(day=29), {}, ForecastMarket())
        self.assertEqual(terminal.targets, {})
        small_episode = CropPlanner(legacy_policy()).plan(observation(), {"episodeSteps": 48}, ForecastMarket())
        self.assertEqual(small_episode.targets, {})

    def test_budget_and_no_forced_losing_investment(self):
        self.assertEqual(CropPlanner(legacy_policy()).plan(observation(money=300), {}, ForecastMarket()).targets, {})
        cheap = ForecastMarket({crop: 1 for crop in ForecastMarket().prices})
        self.assertEqual(CropPlanner(legacy_policy()).plan(observation(), {}, cheap).targets, {})
        plan = CropPlanner(legacy_policy()).plan(observation(money=400), {}, ForecastMarket())
        costs = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
        self.assertLessEqual(sum(costs[crop] * count for crop, count in plan.targets.items()), 100)

    def test_preserves_existing_crops_when_quotes_change_and_late(self):
        obs = observation(plants=("WHEAT", "WHEAT", "TOMATO", "TOMATO"))
        plan = CropPlanner(legacy_policy()).plan(obs, {}, ForecastMarket())
        self.assertEqual(plan.short_crop, "WHEAT")
        self.assertEqual(plan.long_crop, "TOMATO")
        self.assertGreaterEqual(plan.short_target, 2)
        self.assertGreaterEqual(plan.long_target, 2)
        late = CropPlanner(legacy_policy()).plan(observation(day=28, plants=("TOMATO",) * 3), {}, ForecastMarket())
        self.assertEqual(late.long_crop, "TOMATO")
        self.assertEqual(late.long_target, 3)

    def test_respects_other_occupied_land_and_max_plots(self):
        obs = observation(plants=("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"))
        plan = CropPlanner(legacy_policy(max_plots=4)).plan(obs, {}, ForecastMarket())
        # Two secondary crops remain protected, outside the selected pair.
        self.assertLessEqual(sum(plan.targets.values()) + 2, 4)

    def test_market_impact_can_leave_land_unused(self):
        config = legacy_policy(dual_cycle=False)
        low_impact = CropPlanner(config).plan(observation(), {}, ForecastMarket())
        high_impact = CropPlanner(config).plan(observation(), {}, ForecastMarket(impact=2))
        self.assertLess(high_impact.short_target, low_impact.short_target)

    def test_risk_penalty_and_ablation(self):
        risky = ForecastMarket(risk=0.8)
        cautious = CropPlanner(legacy_policy()).plan(observation(), {}, risky)
        neutral = CropPlanner(legacy_policy(risk_aversion=0)).plan(observation(), {}, risky)
        self.assertLess(cautious.scores["WHEAT"], neutral.scores["WHEAT"])
        market = ForecastMarket()
        simple = CropPlanner(legacy_policy(use_forecast=False, dual_cycle=False)).plan(observation(), {}, market)
        self.assertEqual(market.calls, [])
        self.assertGreater(simple.short_target, 0)
        self.assertEqual(simple.long_target, 0)


class PortfolioTests(unittest.TestCase):
    def test_defaults_enable_portfolio_and_all_five_crops_can_win(self):
        config = PolicyConfig()
        self.assertEqual((config.strategy, config.optimizer), ("adaptive", "branch_bound"))
        for winner in ForecastMarket().prices:
            market = ForecastMarket({crop: 1000 if crop == winner else 1
                                     for crop in ForecastMarket().prices})
            plan = CropPlanner().plan(observation(), {}, market)
            self.assertEqual(set(plan.targets), {winner})
            self.assertGreater(plan.targets[winner], 0)

    def test_all_existing_crops_remain_in_portfolio(self):
        crops = tuple(ForecastMarket().prices)
        plan = CropPlanner(PolicyConfig(max_plots=5)).plan(observation(plants=crops), {}, ForecastMarket())
        self.assertEqual(plan.targets, dict.fromkeys(crops, 1))

    def test_recurrent_output_is_priced_at_executable_harvest_dates(self):
        market = ForecastMarket()
        CropPlanner(PolicyConfig(max_plots=1)).plan(observation(), {}, market)
        calls = [(h, q) for crop, h, q in market.calls if crop == "TOMATO"]
        self.assertEqual(calls, [(217, 2), (265, 2), (505, 2), (553, 2)])
        self.assertTrue(all(h < 706 for h, _ in calls))

    def test_aggregate_batch_api_and_empirical_quantile_are_used(self):
        class BatchMarket(ForecastMarket):
            def __init__(self):
                super().__init__()
                self.batch_calls, self.quantiles = [], []

            def forecast_batches(self, crop, batches):
                self.batch_calls.append((crop, list(batches)))
                return {"expected_revenue": sum(q for _, q in batches) * self.prices[crop]}

            def revenue_quantile(self, crop, batches, tau):
                self.quantiles.append(tau)
                return sum(q for _, q in batches) * self.prices[crop] * .5

        market = BatchMarket()
        plan = CropPlanner(PolicyConfig(max_plots=2, risk_quantile=.15)).plan(observation(), {}, market)
        self.assertGreater(len(market.batch_calls), 0)
        self.assertEqual(set(market.quantiles), {.15})
        self.assertEqual(market.calls, [])
        self.assertTrue(all(batches == sorted(batches) for _, batches in market.batch_calls))
        neutral = CropPlanner(PolicyConfig(max_plots=2, risk_aversion=0)).plan(observation(), {}, BatchMarket())
        self.assertLess(plan.scores["MELON"], neutral.scores["MELON"])

    def test_integer_repair_respects_land_cash_labor_and_storage(self):
        costs = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
        rng = random.Random(110)
        for _ in range(20):
            money = rng.randint(300, 2000)
            plots = rng.randint(1, 25)
            workers = rng.randint(1, 4)
            config = PolicyConfig(max_plots=plots, target_workers=workers)
            plan = CropPlanner(config).plan(observation(money=money), {"shedCapacity": 12}, ForecastMarket())
            self.assertTrue(all(isinstance(count, int) and count >= 0 for count in plan.targets.values()))
            self.assertLessEqual(sum(plan.targets.values()), plots)
            self.assertLessEqual(sum(costs[crop] * count for crop, count in plan.targets.items()), money - 300)
            self.assertLessEqual(sum(plan.targets.values()) * config.labor_actions_per_plot_day,
                                 workers * 24 * config.labor_utilization)
            for crop, count in plan.targets.items():
                largest_batch = 1 if crop in {"TOMATO", "STRAWBERRY"} else yield_without_fertilizer(crop, full_cycle_day(crop))
                self.assertLessEqual(count * largest_batch, 12)

    def test_stored_seed_and_animal_resource_reservations(self):
        obs = observation(money=300)
        obs["private"]["seeds"] = {"MELON": 3}
        plan = CropPlanner().plan(obs, {}, ForecastMarket())
        self.assertEqual(plan.targets, {"MELON": 3})
        blocked = CropPlanner().plan(observation(), {"reservedAnimalPlots": 25}, ForecastMarket())
        self.assertEqual(blocked.targets, {})
        no_labor = CropPlanner().plan(observation(), {"reservedAnimalActions": 200}, ForecastMarket())
        self.assertEqual(no_labor.targets, {})

    def test_liquidator_preserves_existing_without_new_investments(self):
        plan = CropPlanner().plan(observation(day=26, plants=("TOMATO", "MELON")), {}, ForecastMarket())
        self.assertEqual(plan.specialist, "liquidator")
        self.assertEqual(plan.targets, {"TOMATO": 1, "MELON": 1})

    def test_replanning_does_not_add_batches_over_existing_storage_budget(self):
        market = ForecastMarket({crop: 1000 if crop == "MELON" else 1
                                 for crop in ForecastMarket().prices})
        plan = CropPlanner().plan(observation(plants=("MELON",) * 16), {}, market)
        self.assertEqual(plan.targets, {"MELON": 16})

    def test_integer_budget_is_shared_and_workforce_cash_is_reserved(self):
        from kaggriculture_agent.domain import fib_cost
        obs = observation(money=450)
        policy = PolicyConfig(integer_node_limit=3)
        plan = CropPlanner(policy).plan(obs, {}, ForecastMarket())
        self.assertLessEqual(plan.optimizer_diagnostics["nodes"], 3)
        self.assertGreaterEqual(plan.workforce_target, 1)
        self.assertLessEqual(plan.workforce_target, policy.target_workers)
        hire_cost = sum(fib_cost(i) for i in range(plan.workforce_target - 1))
        costs = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
        seed_cost = sum(costs[crop] * count for crop, count in plan.targets.items())
        self.assertLessEqual(seed_cost + hire_cost, 150)
        self.assertLessEqual(sum(plan.targets.values()) * policy.labor_actions_per_plot_day,
                             plan.workforce_target * 24 * policy.labor_utilization)

    def test_expensive_fibonacci_hires_reduce_new_workload(self):
        cheap = CropPlanner().plan(observation(), {}, ForecastMarket())
        expensive = CropPlanner().plan(observation(), {"farmHandCostMult": 10000}, ForecastMarket())
        self.assertGreater(cheap.workforce_target, expensive.workforce_target)
        self.assertEqual(expensive.workforce_target, 1)
        self.assertGreater(sum(cheap.targets.values()), sum(expensive.targets.values()))

    def test_idle_farm_does_not_hire_to_configured_ceiling(self):
        plan = CropPlanner(PolicyConfig(max_plots=0, target_workers=12)).plan(observation(), {}, ForecastMarket())
        self.assertEqual(plan.workforce_target, 1)
        self.assertEqual(plan.targets, {})

    def test_workforce_selection_protects_existing_labor_commitments(self):
        config = PolicyConfig(target_workers=8)
        plan = CropPlanner(config).plan(observation(plants=("WHEAT",) * 20), {}, ForecastMarket())
        self.assertGreaterEqual(plan.workforce_target * 24 * config.labor_utilization,
                                20 * config.labor_actions_per_plot_day)
        self.assertGreaterEqual(plan.targets["WHEAT"], 20)

    def test_new_configuration_modes_are_validated(self):
        self.assertEqual(PolicyConfig().market_method, "quadrature")
        self.assertTrue(PolicyConfig().enable_fertilizer)
        self.assertFalse(PolicyConfig().enable_contextual_bandit)
        for kwargs in ({"integer_node_limit": 0}, {"market_method": "unknown"}):
            with self.assertRaises(ValueError):
                PolicyConfig(**kwargs)

    def test_crop_clash_penalizes_crop_flooded_by_opponent(self):
        obs_base = observation()
        obs_base["farms"].append({"tiles": [[None] * 10 for _ in range(10)], "money": 3000, "hands": [], "farmer": [4, 4]})
        plan_base = CropPlanner().plan(obs_base, {}, ForecastMarket())
        opp_tiles = [[{"kind": "PLANT", "crop": "MELON", "planted_day": 0, "yield_units": 0}
                      if x < 2 and y < 5 else None for x in range(10)] for y in range(10)]
        obs_opp = observation()
        obs_opp["farms"].append({"tiles": opp_tiles, "money": 3000, "hands": [], "farmer": [4, 4]})
        plan_opp = CropPlanner().plan(obs_opp, {}, ForecastMarket())
        self.assertLess(plan_opp.scores["MELON"], plan_base.scores["MELON"])


if __name__ == "__main__":
    unittest.main()
