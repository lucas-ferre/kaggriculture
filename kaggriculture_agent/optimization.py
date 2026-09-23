"""Small, dependency-free alternatives for controlled policy experiments.

Assignments maximize the number of feasible matches, then minimize cost when
``method='hungarian'``. Greedy is a faster heuristic and can miss both goals.
Non-finite matrix entries mean that a worker cannot perform a task.
"""

from __future__ import annotations

import math
import heapq
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearSolution:
    """Result of a continuous LP; this is not an integer-program certificate."""

    status: str
    variables: tuple[float, ...] = ()
    objective: float = 0.0
    iterations: int = 0


@dataclass(frozen=True)
class IntegerSolution:
    """Bounded mixed-integer search result for the supplied linear model.

    ``gap`` is (upper bound - incumbent) / max(1, abs(incumbent)). No incumbent
    is represented by objective=-inf, variables=(), gap=None. Optimality is
    numerical, within tolerance, and is reported only after closing the tree.
    """

    status: str
    variables: tuple[float, ...] = ()
    objective: float = -math.inf
    best_bound: float = math.inf
    gap: float | None = None
    nodes: int = 0
    iterations: int = 0


def branch_bound(objective: list[float], constraints: list[list[float]],
                 bounds: list[float], *, integer_indices=None, node_limit: int = 64,
                 time_limit_seconds: float | None = None, incumbent=None,
                 tolerance: float = 1e-7) -> IntegerSolution:
    """Maximize a bounded-work MILP using two-phase simplex relaxations.

    By default all variables are integer; ``integer_indices=[]`` leaves a LP.
    The node budget and tie rules are deterministic. Optional wall time is an
    emergency stop between LPs, not a deterministic or hard real-time promise.
    A feasible incumbent survives node/time/numerical limits. LP unboundedness
    alone is reported as relaxation_unbounded, not a MILP unbounded certificate.
    """
    n = len(objective)
    c = [float(v) for v in objective]
    a = [[float(v) for v in row] for row in constraints]
    b = [float(v) for v in bounds]
    if len(a) != len(b) or any(len(row) != n for row in a):
        raise ValueError("MILP constraints must match objective and bounds dimensions")
    if any(not math.isfinite(v) for v in c + b + [v for row in a for v in row]):
        raise ValueError("MILP coefficients must be finite")
    if not isinstance(node_limit, int) or node_limit < 0:
        raise ValueError("node_limit must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("MILP tolerance must be finite and positive")
    if time_limit_seconds is not None and (not math.isfinite(time_limit_seconds) or time_limit_seconds < 0):
        raise ValueError("time limit must be finite and nonnegative")
    indices = tuple(range(n)) if integer_indices is None else tuple(sorted(set(integer_indices)))
    if any(not isinstance(i, int) or i < 0 or i >= n for i in indices):
        raise ValueError("integer indices must refer to objective variables")
    objective_tolerance = tolerance * (max(map(abs, c), default=0.) or 1.)

    def feasible(x, rows=a, rhs=b):
        if len(x) != n or any(not math.isfinite(v) or v < -tolerance for v in x):
            return False
        if any(abs(x[i] - round(x[i])) > tolerance for i in indices):
            return False
        for row, bound in zip(rows, rhs):
            scale = max(map(abs, row), default=0.) or 1.
            if not any(row) and bound < 0:
                return False
            terms = [v / scale * z for v, z in zip(row, x)]
            bound = bound / scale
            if math.fsum(terms) - bound > tolerance * (1 + abs(bound) + math.fsum(map(abs, terms))):
                return False
        return True

    best_x, best_value = (), -math.inf
    if incumbent is not None:
        candidate = tuple(float(v) for v in incumbent)
        if not feasible(candidate):
            raise ValueError("incumbent must be feasible and integral on integer_indices")
        best_x, best_value = candidate, math.fsum(v * x for v, x in zip(c, candidate))
    elif feasible((0.0,) * n):
        best_x, best_value = (0.0,) * n, 0.0
    # Pending nodes retain the parent relaxation's valid upper bound. This
    # preserves a useful global bound even when budget ends before child LPs.
    pending = [(-math.inf, 0, (), ())]
    sequence, nodes, iterations = 0, 0, 0
    started = time.perf_counter()
    status, unresolved = "optimal", -math.inf
    while pending:
        if pending[0][0] >= -best_value - objective_tolerance:
            pending.clear()
            break
        if nodes >= node_limit:
            status = "node_limit"
            break
        if time_limit_seconds is not None and time.perf_counter() - started >= time_limit_seconds:
            status = "time_limit"
            break
        negative_bound, _, extra_rows, extra_bounds = heapq.heappop(pending)
        rows, rhs = a + list(extra_rows), b + list(extra_bounds)
        result = simplex(c, rows, rhs)
        nodes += 1
        iterations += result.iterations
        if result.status == "infeasible":
            continue
        if result.status != "optimal":
            status = "relaxation_unbounded" if result.status == "unbounded" else result.status
            unresolved = -negative_bound
            break
        upper = result.objective
        if upper <= best_value + objective_tolerance:
            continue
        fractional = [i for i in indices if abs(result.variables[i] - round(result.variables[i])) > tolerance]
        candidate = list(result.variables)
        for i in indices:
            candidate[i] = float(math.floor(candidate[i])) if fractional else float(round(candidate[i]))
        if feasible(candidate, rows, rhs):
            value = math.fsum(v * x for v, x in zip(c, candidate))
            if value > best_value:
                best_x, best_value = tuple(candidate), value
        if not fractional:
            if not feasible(candidate, rows, rhs):
                status, unresolved = "numerical_error", upper
                break
            continue
        if upper <= best_value + objective_tolerance:
            continue
        index = max(fractional, key=lambda i: (min(result.variables[i] % 1, 1 - result.variables[i] % 1), -i))
        floor = math.floor(result.variables[index])
        for sign, bound in ((1., float(floor)), (-1., float(-floor - 1))):
            row = tuple(sign if j == index else 0.0 for j in range(n))
            sequence += 1
            heapq.heappush(pending, (-upper, sequence, extra_rows + (row,), extra_bounds + (bound,)))
    remaining = max((-item[0] for item in pending if -item[0] > best_value + objective_tolerance), default=-math.inf)
    if remaining == -math.inf and unresolved == -math.inf:
        status = "optimal" if best_value != -math.inf else "infeasible"
    upper = max(best_value, remaining, unresolved)
    gap = None if best_value == -math.inf else max(0.0, upper - best_value) / max(1., abs(best_value))
    return IntegerSolution(status, best_x, best_value, upper, gap, nodes, iterations)


def simplex(objective: list[float], constraints: list[list[float]],
            bounds: list[float], *, tolerance: float = 1e-9,
            max_iterations: int = 10000) -> LinearSolution:
    """Maximize ``objective @ x`` subject to ``constraints @ x <= bounds, x >= 0``.

    Two-phase primal simplex accepts negative right-hand sides and returns an
    explicit optimal, infeasible, unbounded, iteration_limit or numerical_error
    status. Bland's entering/leaving tie rules avoid exact-arithmetic cycling.
    Coefficients must be finite; row/objective scaling and a final primal check
    mitigate floating-point error without promising exact arithmetic.
    """
    n, m = len(objective), len(bounds)
    if len(constraints) != m or any(len(row) != n for row in constraints):
        raise ValueError("LP constraints must match objective and bounds dimensions")
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("LP tolerance and iteration limit must be positive")
    c = [float(v) for v in objective]
    a = [[float(v) for v in row] for row in constraints]
    b = [float(v) for v in bounds]
    if any(not math.isfinite(v) for v in c + b + [v for row in a for v in row]):
        raise ValueError("LP coefficients must be finite")
    if not n:
        return LinearSolution("optimal" if all(v >= 0 for v in b) else "infeasible")
    if not m:
        return LinearSolution("unbounded" if any(v > 0 for v in c) else "optimal",
                              (0.0,) * n)
    if any(not any(row) and bound < 0 for row, bound in zip(a, b)):
        return LinearSolution("infeasible")
    basic, nonbasic = list(range(n, n + m)), list(range(n)) + [-1]
    tableau = [[0.0] * (n + 2) for _ in range(m + 2)]
    for i in range(m):
        scale = max(map(abs, a[i])) or 1.0
        tableau[i][:n] = [v / scale for v in a[i]]
        tableau[i][n], tableau[i][-1] = -1.0, b[i] / scale
    cscale = max(map(abs, c)) or 1.0
    tableau[m][:n] = [-v / cscale for v in c]
    tableau[m + 1][n] = 1.0
    iterations = 0

    def pivot(row: int, col: int) -> bool:
        nonlocal iterations
        if iterations >= max_iterations:
            return False
        divisor = tableau[row][col]
        if not math.isfinite(divisor) or abs(divisor) < tolerance * tolerance:
            raise ArithmeticError("unstable LP pivot")
        for i in range(m + 2):
            if i == row:
                continue
            factor = tableau[i][col] / divisor
            for j in range(n + 2):
                if j != col:
                    tableau[i][j] -= tableau[row][j] * factor
            tableau[i][col] = -factor
        for j in range(n + 2):
            if j != col:
                tableau[row][j] /= divisor
        tableau[row][col] = 1.0 / divisor
        basic[row], nonbasic[col] = nonbasic[col], basic[row]
        iterations += 1
        if any(not math.isfinite(v) for line in tableau for v in line):
            raise ArithmeticError("nonfinite LP tableau")
        return True

    def phase(which: int) -> str:
        cost_row = m + 1 if which == 1 else m
        while True:
            entering = [j for j in range(n + 1)
                        if (which == 1 or nonbasic[j] != -1)
                        and tableau[cost_row][j] < -tolerance]
            if not entering:
                return "optimal"
            col = min(entering, key=lambda j: nonbasic[j])
            leaving = [i for i in range(m) if tableau[i][col] > tolerance]
            if not leaving:
                return "unbounded"
            ratio = min(tableau[i][-1] / tableau[i][col] for i in leaving)
            row = min((i for i in leaving
                       if tableau[i][-1] / tableau[i][col] <= ratio + tolerance),
                      key=lambda i: basic[i])
            if not pivot(row, col):
                return "iteration_limit"

    try:
        row = min(range(m), key=lambda i: tableau[i][-1])
        if tableau[row][-1] < -tolerance:
            if not pivot(row, n):
                return LinearSolution("iteration_limit", iterations=iterations)
            status = phase(1)
            if status == "iteration_limit":
                return LinearSolution(status, iterations=iterations)
            if tableau[m + 1][-1] < -tolerance:
                return LinearSolution("infeasible", iterations=iterations)
            if status != "optimal" or abs(tableau[m + 1][-1]) > tolerance * 10:
                return LinearSolution("numerical_error", iterations=iterations)
            for i in range(m):
                if basic[i] == -1:
                    eligible = [j for j in range(n + 1) if nonbasic[j] != -1
                                and abs(tableau[i][j]) > tolerance]
                    if eligible and not pivot(i, min(eligible, key=lambda j: nonbasic[j])):
                        return LinearSolution("iteration_limit", iterations=iterations)
        status = phase(2)
        if status != "optimal":
            return LinearSolution(status, iterations=iterations)
        solution = [0.0] * n
        for i, index in enumerate(basic):
            if 0 <= index < n:
                solution[index] = tableau[i][-1]
        if any(v < -tolerance * 20 or not math.isfinite(v) for v in solution):
            return LinearSolution("numerical_error", iterations=iterations)
        solution = [max(0.0, v) for v in solution]
        for row, bound in zip(a, b):
            terms = [v * x for v, x in zip(row, solution)]
            if math.fsum(terms) - bound > tolerance * 20 * (1 + abs(bound) + math.fsum(map(abs, terms))):
                return LinearSolution("numerical_error", iterations=iterations)
        value = math.fsum(v * x for v, x in zip(c, solution))
        if not math.isfinite(value):
            return LinearSolution("numerical_error", iterations=iterations)
        return LinearSolution("optimal", tuple(solution), value, iterations)
    except (ArithmeticError, OverflowError):
        return LinearSolution("numerical_error", iterations=iterations)


def assign(costs: list[list[float]], method: str = "greedy") -> list[tuple[int, int]]:
    """Return unique (worker, task) pairs; rectangular matrices are supported."""
    if method not in {"greedy", "hungarian"}:
        raise ValueError("assignment method must be 'greedy' or 'hungarian'")
    if not costs:
        return []
    width = len(costs[0])
    if any(len(row) != width for row in costs):
        raise ValueError("cost matrix must be rectangular")
    if width == 0:
        return []
    edges = [(float(value), i, j) for i, row in enumerate(costs)
             for j, value in enumerate(row) if math.isfinite(value)]
    if not edges:
        return []
    if method == "greedy":
        used_workers, used_tasks, result = set(), set(), []
        for _, worker, task in sorted(edges):
            if worker not in used_workers and task not in used_tasks:
                result.append((worker, task))
                used_workers.add(worker)
                used_tasks.add(task)
        return sorted(result)

    # Dummy tasks permit partial matchings. Normalization keeps finite costs in
    # [0, 1]; a dummy penalty > worker count makes cardinality the first goal.
    n = len(costs)
    low, high = min(e[0] for e in edges), max(e[0] for e in edges)
    scale = max(1.0, abs(low), abs(high))
    low_scaled, high_scaled = low / scale, high / scale
    span = max(1e-15, high_scaled - low_scaled)
    dummy = float(n + 1)
    forbidden = dummy * (n + 2)
    matrix = [[((v / scale - low_scaled) / span if math.isfinite(v) else forbidden)
               for v in row] + [dummy] * n for row in costs]
    m = width + n
    # Shortest augmenting path formulation of the Hungarian algorithm.
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for worker in range(1, n + 1):
        p[0], j0 = worker, 0
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    reduced = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if reduced < minimum[j]:
                        minimum[j], way[j] = reduced, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((p[j] - 1, j - 1) for j in range(1, width + 1)
                  if p[j] and math.isfinite(costs[p[j] - 1][j - 1]))


def enumerate_allocations(capacity: int, minimum: tuple[int, int] = (0, 0)) -> Iterator[tuple[int, int]]:
    """Enumerate all two-crop area allocations, including unused land."""
    short_min, long_min = minimum
    if capacity < 0 or min(minimum) < 0:
        raise ValueError("capacity and minimum counts must be nonnegative")
    for short in range(short_min, capacity - long_min + 1):
        for long in range(long_min, capacity - short + 1):
            yield short, long


def optimize_allocation(capacity: int, score: Callable[[int, int], float],
                        method: str = "enumerate", minimum: tuple[int, int] = (0, 0),
                        beam_width: int = 16) -> tuple[int, int]:
    """Maximize an evaluator under an area cap; -inf marks infeasible choices.

    Enumeration is exact over this small discrete planning model. Beam search
    is explicitly approximate and retains at most ``beam_width`` states per
    expansion depth. Neither method claims optimal play in the full game.
    """
    if capacity < 0 or min(minimum) < 0 or sum(minimum) > capacity:
        raise ValueError("minimum allocation exceeds the area capacity")
    if method not in {"enumerate", "beam"}:
        raise ValueError("allocation method must be 'enumerate' or 'beam'")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if method == "enumerate":
        return max(enumerate_allocations(capacity, minimum),
                   key=lambda pair: (score(*pair), -sum(pair), -pair[1]))
    frontier = [minimum]
    best, best_score = minimum, score(*minimum)
    for _ in range(capacity - sum(minimum)):
        expanded = {(s + ds, l + dl) for s, l in frontier
                    for ds, dl in ((1, 0), (0, 1)) if s + l < capacity}
        ranked = sorted(((score(*pair), pair) for pair in expanded), reverse=True)
        if not ranked:
            break
        frontier = [pair for _, pair in ranked[:beam_width]]
        if ranked[0][0] > best_score:
            best_score, best = ranked[0]
    return best
