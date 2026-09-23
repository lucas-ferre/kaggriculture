"""Outcome-based ratings and truthful benchmark action/production metrics."""
import math
import unittest

from scripts.arena import bradley_terry, rate_benchmark
from scripts.benchmark import _crop_statistics, episode_metrics, make_environment, opponent_config, run_match, variant_config


class RatingsTests(unittest.TestCase):
    def test_ties_are_symmetric_and_finite(self):
        result = bradley_terry([{"player_a": "a", "player_b": "b", "score_a": 0.5}])
        ratings = {row["player"]: row for row in result["ratings"]}
        self.assertAlmostEqual(ratings["a"]["rating"], ratings["b"]["rating"])
        self.assertEqual(ratings["a"]["points"], 0.5)
        self.assertEqual(ratings["b"]["draws"], 1)
        self.assertTrue(result["converged"])

    def test_unbeaten_rating_regularized_and_outcome_order_respected(self):
        result = bradley_terry(
            [{"player_a": "a", "player_b": "b", "score_a": 1}] * 12
            + [{"player_a": "b", "player_b": "c", "score_a": 1}] * 12)
        self.assertEqual([row["player"] for row in result["ratings"]], ["a", "b", "c"])
        self.assertTrue(all(math.isfinite(row["rating"]) for row in result["ratings"]))
        self.assertAlmostEqual(sum(row["log_ability"] for row in result["ratings"]), 0.0)

    def test_invalid_matches_excluded_and_disconnected_components_disclosed(self):
        result = bradley_terry([
            {"player_a": "a", "player_b": "b", "score_a": 1},
            {"player_a": "c", "player_b": "d", "score_a": 0},
            {"player_a": "bad", "player_b": "a", "score_a": 1, "valid": False},
        ])
        self.assertEqual(result["played_matches"], 2)
        self.assertEqual(result["excluded_matches"], 1)
        self.assertEqual(result["connected_components"], [["a", "b"], ["c", "d"]])

    def test_more_balanced_games_reduce_conditional_uncertainty(self):
        game = {"player_a": "a", "player_b": "b", "score_a": 0.5}
        small = bradley_terry([game])["ratings"][0]["approx_prior_conditioned_sd"]
        large = bradley_terry([game] * 100)["ratings"][0]["approx_prior_conditioned_sd"]
        self.assertLess(large, small)

    def test_match_adapter_uses_results_not_money_magnitudes(self):
        row = {"variant": "adaptive", "opponent": "starter", "valid": True,
               "outcome": "win", "money": 1}
        result = rate_benchmark([row])
        self.assertEqual(result, rate_benchmark([row | {"money": 1000000}]))
        self.assertEqual(result["played_matches"], 1)

    def test_invalid_model_parameters_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bradley_terry([], value)
        with self.assertRaises(ValueError):
            bradley_terry([{"player_a": "a", "player_b": "b", "score_a": 0.7}])
        self.assertEqual(bradley_terry([])["ratings"], [])

    def test_same_policy_selfplay_is_excluded_but_disclosed(self):
        source = {"runtime_fingerprint_sha256": "runtime-a", "version": "0.1.2"}
        policy = {"max_plots": 40, "optimizer": "branch_bound"}
        game = {"variant": "adaptive", "opponent": "selfplay", "valid": True,
                "outcome": "win", "policy": policy, "opponent_policy": dict(policy),
                "policy_source": source, "opponent_source": dict(source)}
        result = rate_benchmark([game])
        self.assertEqual(result["recorded_matches"], 1)
        self.assertEqual(result["valid_matches"], 1)
        self.assertEqual(result["played_matches"], 0)
        self.assertEqual(result["excluded_same_policy_matches"], 1)
        self.assertEqual(result["ratings"], [])
        self.assertEqual(result["policy_identities"][0]["player"], "agent:adaptive")
        self.assertEqual(result["policy_identities"][0]["aliases"], ["opponent:selfplay", "variant:adaptive"])

    def test_frozen_policy_shares_identity_across_both_roles(self):
        old_source = {"frozen": True, "sha256": "archive-011", "version": "0.1.1"}
        new_source = {"runtime_fingerprint_sha256": "runtime-012", "version": "0.1.2"}
        old_policy, new_policy = {"strategy": "adaptive", "optimizer": "simplex"}, {"strategy": "adaptive", "optimizer": "branch_bound"}
        results = rate_benchmark([
            {"variant": "baseline_011", "opponent": "starter", "outcome": "win", "valid": True,
             "policy": old_policy, "policy_source": old_source},
            {"variant": "adaptive", "opponent": "baseline_011", "outcome": "win", "valid": True,
             "policy": new_policy, "policy_source": new_source,
             "opponent_policy": old_policy, "opponent_source": old_source},
            {"variant": "baseline_011", "opponent": "baseline_011", "outcome": "draw", "valid": True,
             "policy": old_policy, "policy_source": old_source,
             "opponent_policy": old_policy, "opponent_source": old_source},
        ])
        self.assertEqual(results["played_matches"], 2)
        self.assertEqual(results["excluded_same_policy_matches"], 1)
        self.assertEqual(len(results["ratings"]), 3)
        old = next(row for row in results["ratings"] if row["player"] == "agent:baseline_011")
        self.assertEqual(old["matches"], 2)
        self.assertEqual(old["points"], 1)
        self.assertEqual(len(results["connected_components"]), 1)

    def test_different_configuration_or_source_is_not_merged(self):
        source = {"runtime_fingerprint_sha256": "runtime-a"}
        base = {"variant": "adaptive", "opponent": "selfplay", "outcome": "win", "valid": True,
                "policy": {"max_plots": 40}, "opponent_policy": {"max_plots": 20},
                "policy_source": source, "opponent_source": source}
        for game in (base, base | {"opponent_policy": base["policy"],
                                   "opponent_source": {"runtime_fingerprint_sha256": "runtime-b"}}):
            result = rate_benchmark([game])
            self.assertEqual(result["played_matches"], 1)
            self.assertEqual(result["excluded_same_policy_matches"], 0)
            self.assertEqual(len({row["player"] for row in result["ratings"]}), 2)

    def test_matching_real_policy_merges_arbitrary_aliases_but_not_profiles(self):
        source = {"runtime_fingerprint_sha256": "runtime-a"}
        result = rate_benchmark([
            {"variant": "adaptive", "opponent": "custom_same", "outcome": "draw", "valid": True,
             "policy": {"max_plots": 40}, "opponent_policy": {"max_plots": 40},
             "policy_source": source, "opponent_source": source},
            {"variant": "adaptive", "opponent": "livestock", "outcome": "win", "valid": True,
             "policy": {"max_plots": 40}, "opponent_policy": {"max_plots": 8},
             "policy_source": source, "opponent_source": source},
        ])
        self.assertEqual(result["played_matches"], 1)
        self.assertEqual(result["excluded_same_policy_matches"], 1)
        self.assertEqual({row["player"] for row in result["ratings"]}, {"agent:adaptive", "agent:livestock"})

    def test_old_report_selfplay_alias_has_explicit_fallback_identity(self):
        result = rate_benchmark([{"variant": "adaptive", "opponent": "selfplay", "valid": True, "outcome": "draw"}])
        self.assertEqual(result["excluded_same_policy_matches"], 1)
        self.assertFalse(result["policy_identities"][0]["fingerprinted"])


class ProfileTests(unittest.TestCase):
    def test_frozen_baseline_import_is_isolated_and_playable(self):
        from kaggriculture_agent.planner import PolicyConfig
        original = variant_config("baseline_010")
        self.assertEqual(type(original).__module__, "_baseline_010.planner")
        self.assertNotIsInstance(original, PolicyConfig)
        match = run_match(original, seed=7, opponent="baseline_010", steps=4)
        self.assertTrue(match["valid"])
        self.assertEqual(match["outcome"], "draw")
        self.assertTrue(match["policy_source"]["frozen"])
        self.assertEqual(len(match["policy_source"]["sha256"]), 64)

    def test_legacy_parameters_explicit(self):
        config = variant_config("legacy_policy")
        self.assertEqual((config.strategy, config.optimizer, config.max_plots, config.target_workers),
                         ("dual", "enumerate", 20, 6))
        self.assertFalse(config.enable_livestock)
        self.assertFalse(config.enable_expansion)

    def test_profiles_exercise_different_mechanisms(self):
        self.assertEqual(opponent_config("passive").max_plots, 0)
        self.assertFalse(opponent_config("crop_flood").dual_cycle)
        self.assertFalse(opponent_config("crop_flood").use_forecast)
        self.assertTrue(opponent_config("livestock").enable_livestock)
        self.assertFalse(opponent_config("expansive").enable_livestock)
        self.assertGreater(opponent_config("expansive").max_plots, 25)


class MetricsTests(unittest.TestCase):
    def test_side_one_uses_shared_replay_clock_and_own_private_stock(self):
        from contextlib import redirect_stdout
        from io import StringIO
        env = make_environment(7, 300, {"weedSpawnChance": 0})
        idle = {"farmer": ["PASS"], "hands": [], "market": []}
        env.step([idle, {"market": [["BUY_SEED", "STRAWBERRY", 1]]}])
        env.step([idle, {"farmer": ["PLANT", "STRAWBERRY"]}])
        env.step([idle, {"farmer": ["WATER"]}])
        while env.state[0].observation.step < 240:
            hour = env.state[0].observation.hour
            env.step([idle, {"farmer": ["WATER"] if hour == 0 else ["PASS"]}])
        self.assertIsNone(env.steps[-1][1].observation.get("step"))
        env.step([idle, {"farmer": ["HARVEST"]}])
        actual = env.state[1].observation.private["inventories"][0]["STRAWBERRY"]
        self.assertGreater(actual, 0)
        # Shared fields may be defaults or absent on side one in raw replay
        # records; private stocks and actions remain specific to that player.
        for state in env.steps:
            state[1].observation["farms"] = []
            state[1].observation["day"] = 0
            state[1].observation["hour"] = 0
        captured = StringIO()
        with redirect_stdout(captured):
            result = episode_metrics(env, 1)
        self.assertEqual(captured.getvalue(), "")
        self.assertEqual(result["production"]["harvested_units"], {"STRAWBERRY": actual})
        self.assertEqual(result["unit_action_audit"]["non_pass_noop_count"], 0)
        self.assertEqual(result["economy"]["net_cash_change"], -100)
        self.assertEqual(result["economy"]["final_carried_units"], {"STRAWBERRY": actual})
        self.assertEqual(_crop_statistics(env, 1)["successful_plantings"], {"STRAWBERRY": 1})

    def test_physical_replay_counts_success_noop_and_harvest_without_mutation(self):
        env = make_environment(7, 100, {"weedSpawnChance": 0})
        idle = {"farmer": ["PASS"], "hands": [], "market": []}
        # Deliberately invalid first planting: purchase resolves afterwards.
        env.step([{"farmer": ["PLANT", "WHEAT"], "market": [["BUY_SEED", "WHEAT", 1]]}, idle])
        env.step([{"farmer": ["PLANT", "WHEAT"]}, idle])
        env.step([{"farmer": ["WATER"]}, idle])
        env.step([{"farmer": ["WATER"]}, idle])
        while env.state[0].observation.step < 48:
            hour = env.state[0].observation.hour
            env.step([{"farmer": ["WATER"] if hour == 0 else ["PASS"]}, idle])
        env.step([{"farmer": ["HARVEST"]}, idle])
        import copy
        before = copy.deepcopy(env.steps)
        result = episode_metrics(env, 0)
        self.assertEqual(env.steps, before)
        self.assertTrue(result["unit_action_audit"]["available"])
        self.assertEqual(result["unit_action_audit"]["atomic_blocked_plant_requests"], {"WHEAT": 1})
        self.assertEqual(result["unit_action_audit"]["non_pass_noops"].get("WATER"), 1)
        self.assertEqual(result["production"]["successful_plant_actions"], 1)
        self.assertGreater(result["production"]["harvested_units"].get("WHEAT", 0), 0)
        self.assertEqual(result["land"]["purchased_quadrants"], 0)
        self.assertEqual(result["economy"]["net_cash_change"], -10)

    def test_disabled_audit_has_no_invented_production_counts(self):
        env = make_environment(7, 2)
        env.run(["pass", "pass"])
        result = episode_metrics(env, 0, audit_unit_actions=False)
        self.assertFalse(result["unit_action_audit"]["available"])
        self.assertNotIn("production", result)


if __name__ == "__main__":
    unittest.main()
