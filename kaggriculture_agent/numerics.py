"""Small, bounded numerical tools; no optional runtime dependencies."""
import math

# Standard-normal nodes/weights derived from five-point Gauss-Hermite, with
# nodes multiplied by sqrt(2) and weights divided by sqrt(pi).
NORMAL_NODES_5 = (-2.8569700138728056, -1.355626179974266, 0.0,
                  1.355626179974266, 2.8569700138728056)
NORMAL_WEIGHTS_5 = (0.011257411327720688, 0.22207592200561264,
                    0.5333333333333333, 0.22207592200561264, 0.011257411327720688)


def weighted_quantile(values, weights, tau):
    """Inverse CDF of a discrete weighted distribution, including point masses."""
    if not 0 <= tau <= 1 or not values or len(values) != len(weights):
        raise ValueError("Invalid weighted quantile input")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive sum")
    pairs = sorted((value, weight) for value, weight in zip(values, weights) if weight > 0)
    target, cumulative = tau * sum(weights), 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def trailing_quadratic_slope(values):
    """Endpoint derivative of a quadratic fitted to the last seven samples.

    Equal time spacing is required. Unlike a centered Savitzky-Golay filter,
    every coefficient multiplies an already observed sample. This is a local
    trend estimate, not a causal identification of another player's actions.
    """
    if len(values) < 7:
        raise ValueError("Seven trailing samples are required")
    samples = values[-7:]
    if not all(math.isfinite(value) for value in samples):
        raise ValueError("Samples must be finite")
    coefficients = (7, -2, -7, -8, -5, 2, 13)
    # Subtract the endpoint first to limit cancellation for large inventories.
    return sum(weight * (value - samples[-1]) for weight, value in
               zip(coefficients, samples)) / 28.0
