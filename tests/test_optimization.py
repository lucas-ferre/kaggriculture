"""Continuous LP certificates, feasibility, degeneracy and input contracts."""

import itertools
import math
import random
import unittest

from kaggriculture_agent.optimization import branch_bound, simplex


class SimplexTests(unittest.TestCase):
    def test_bounded_known_optimum(self):
        result = simplex([3, 2], [[1, 1], [1, 0], [0, 1]], [4, 2, 3])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.objective, 10)
        self.assertEqual(result.variables, (2, 2))

    def test_negative_bound_requires_phase_one(self):
        result = simplex([1, 2], [[-1, -1], [1, 1], [1, 0]], [-2, 5, 3])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.objective, 10)
        self.assertAlmostEqual(result.variables[1], 5)

    def test_infeasible_and_unbounded_are_distinct(self):
        self.assertEqual(simplex([1], [[1], [-1]], [1, -2]).status, "infeasible")
        self.assertEqual(simplex([1], [[-1]], [-2]).status, "unbounded")
        self.assertEqual(simplex([1], [], []).status, "unbounded")
        self.assertEqual(simplex([1], [[0]], [-1]).status, "infeasible")

    def test_redundant_equalities_and_degenerate_origin(self):
        result = simplex([1, 1], [[1, 1], [-1, -1], [2, 2], [0, 0]], [3, -3, 6, 0])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(sum(result.variables), 3)
        self.assertEqual(simplex([-1], [[1]], [0]).variables, (0.0,))

    def test_classic_cycling_example_terminates(self):
        result = simplex([10, -57, -9, -24],
                         [[.5, -5.5, -2.5, 9], [.5, -1.5, -.5, 1], [1, 0, 0, 0]],
                         [0, 0, 1], max_iterations=100)
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.objective, 1)

    def test_row_scaling_and_fractional_solution(self):
        result = simplex([1, 1], [[1e6, 2e6], [3, 1]], [4e6, 3])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.variables[0], .4)
        self.assertAlmostEqual(result.variables[1], 1.8)

    def test_small_coefficients_preserve_scale_and_exact_empty_cases(self):
        result = simplex([1e-12, 1e-12], [[1e-12, 2e-12], [3e-12, 1e-12]], [4e-12, 3e-12])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.variables[0], .4)
        self.assertAlmostEqual(result.variables[1], 1.8)
        self.assertEqual(simplex([1e-12], [], []).status, "unbounded")
        self.assertEqual(simplex([], [[]], [-1e-12]).status, "infeasible")
        self.assertEqual(simplex([1], [[0]], [-1e-12]).status, "infeasible")

    def test_input_validation_and_iteration_limit(self):
        for objective, constraints, bounds in [([1], [[1, 2]], [1]),
                                                ([math.nan], [[1]], [1]),
                                                ([1], [[math.inf]], [1]),
                                                ([1], [[1]], [])]:
            with self.assertRaises(ValueError):
                simplex(objective, constraints, bounds)
        self.assertEqual(simplex([3, 2], [[1, 1], [1, 0], [0, 1]], [4, 2, 3],
                                 max_iterations=1).status, "iteration_limit")
        self.assertEqual(simplex([], [[]], [0]).status, "optimal")
        self.assertEqual(simplex([], [[]], [-1]).status, "infeasible")

    def test_random_two_dimensional_programs_match_vertex_enumeration(self):
        rng = random.Random(71011)
        for _ in range(50):
            rows = [[1., 0.], [0., 1.]] + [[rng.uniform(-2, 5), rng.uniform(-2, 5)]
                                                   for _ in range(4)]
            bounds = [rng.uniform(2, 10), rng.uniform(2, 10)] + [rng.uniform(0, 15) for _ in range(4)]
            objective = [rng.uniform(-2, 5), rng.uniform(-2, 5)]
            vertices = [(0., 0.)]
            boundaries = list(zip(rows + [[-1., 0.], [0., -1.]], bounds + [0., 0.]))
            for (a, b), (c, d) in itertools.combinations(boundaries, 2):
                det = a[0] * c[1] - a[1] * c[0]
                if abs(det) > 1e-10:
                    x, y = (b * c[1] - a[1] * d) / det, (a[0] * d - b * c[0]) / det
                    if x >= -1e-8 and y >= -1e-8 and all(r[0] * x + r[1] * y <= v + 1e-8
                                                                         for r, v in zip(rows, bounds)):
                        vertices.append((x, y))
            expected = max(objective[0] * x + objective[1] * y for x, y in vertices)
            result = simplex(objective, rows, bounds)
            self.assertEqual(result.status, "optimal")
            self.assertAlmostEqual(result.objective, expected, places=6)


class BranchBoundTests(unittest.TestCase):
    def test_fractional_relaxation_has_integer_certificate_when_tree_closes(self):
        result = branch_bound([3, 2], [[2, 2], [1, 0], [0, 1]], [7, 3, 3])
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.variables, (3, 0))
        self.assertEqual(result.objective, 9)
        self.assertEqual(result.best_bound, 9)
        self.assertEqual(result.gap, 0)

    def test_small_random_integer_programs_match_bruteforce(self):
        rng = random.Random(1202)
        for _ in range(40):
            objective = [rng.randint(-2, 10) for _ in range(3)]
            rows = [[float(i == j) for i in range(3)] for j in range(3)]
            rows += [[rng.randint(-2, 5) for _ in range(3)] for _ in range(3)]
            bounds = [4, 4, 4] + [rng.randint(-3, 15) for _ in range(3)]
            feasible = [x for x in itertools.product(range(5), repeat=3)
                        if all(sum(v * z for v, z in zip(row, x)) <= bound
                               for row, bound in zip(rows, bounds))]
            result = branch_bound(objective, rows, bounds, node_limit=500)
            if not feasible:
                self.assertEqual(result.status, "infeasible")
            else:
                optimum = max(sum(v * z for v, z in zip(objective, x)) for x in feasible)
                self.assertEqual(result.status, "optimal")
                self.assertAlmostEqual(result.objective, optimum)
                self.assertEqual(result.gap, 0)

    def test_mixed_integer_leaves_other_variables_continuous(self):
        result = branch_bound([4, 3], [[2, 1], [1, 0], [0, 1]], [4.5, 2, 3], integer_indices=[0])
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.variables, (1, 2.5))
        self.assertAlmostEqual(result.objective, 11.5)
        continuous = branch_bound([1], [[1]], [1.5], integer_indices=[])
        self.assertEqual(continuous.variables, (1.5,))

    def test_limit_preserves_incumbent_and_valid_upper_bound(self):
        args = ([3, 2], [[2, 2], [1, 0], [0, 1]], [7, 3, 3])
        limited = branch_bound(*args, node_limit=1, incumbent=[2, 1])
        exact = branch_bound(*args, node_limit=100)
        self.assertEqual(limited.status, "node_limit")
        self.assertEqual(limited.nodes, 1)
        self.assertGreaterEqual(limited.objective, 8)
        self.assertGreaterEqual(limited.best_bound, exact.objective)
        self.assertGreater(limited.gap, 0)
        self.assertEqual(limited, branch_bound(*args, node_limit=1, incumbent=[2, 1]))
        stopped = branch_bound(*args, time_limit_seconds=0, incumbent=[2, 1])
        self.assertEqual(stopped.status, "time_limit")
        self.assertEqual(stopped.variables, (2, 1))
        self.assertEqual(stopped.nodes, 0)

    def test_integer_infeasible_and_unbounded_relaxation_statuses(self):
        impossible = branch_bound([1], [[2], [-2]], [1, -1])
        self.assertEqual(impossible.status, "infeasible")
        self.assertIsNone(impossible.gap)
        self.assertEqual(impossible.objective, -math.inf)
        self.assertEqual(branch_bound([1], [], []).status, "relaxation_unbounded")
        limited = branch_bound([1], [[2], [-2]], [1, -1], node_limit=0)
        self.assertEqual(limited.status, "node_limit")
        self.assertEqual(limited.variables, ())

    def test_validation_rejects_invalid_incumbent_and_controls(self):
        for kwargs in ({"incumbent": [2]}, {"incumbent": [.5]}, {"integer_indices": [1]},
                       {"node_limit": -1}, {"node_limit": 1.5}, {"time_limit_seconds": -1}):
            with self.assertRaises(ValueError):
                branch_bound([1], [[1]], [1], **kwargs)
        with self.assertRaises(ValueError):
            branch_bound([1], [[1, 2]], [1])
        with self.assertRaises(ValueError):
            branch_bound([math.nan], [[1]], [1])

    def test_scaled_objective_and_rows_do_not_create_false_certificate(self):
        optimum = branch_bound([1e-12], [[1e-12]], [1.5e-12])
        self.assertEqual(optimum.status, "optimal")
        self.assertEqual(optimum.variables, (1.,))
        self.assertEqual(optimum.objective, 1e-12)
        impossible = branch_bound([1], [[2e-12], [-2e-12]], [1e-12, -1e-12])
        self.assertEqual(impossible.status, "infeasible")


if __name__ == "__main__":
    unittest.main()
