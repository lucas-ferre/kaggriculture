"""Optional multiple linear price regression with held-out residual calibration.

The regression predicts a price ratio, never a probability directly. Independent
calibration episodes supply an empirical residual CDF; a third episode split
must pass the probability gate before its model is eligible at runtime.
"""
from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path

from .quantile import FEATURE_NAMES, configuration_signature, public_features, within_training_domain

SCHEMA_VERSION = 1
MODEL_KIND = "mrlm_price_residual_cdf"
CLASS_NAMES = ("up", "flat", "down")


def predict_ratio(fit, features):
    """Unbounded linear prediction; the residual CDF handles the price floor."""
    return fit["coefficients"][0] + sum(
        beta * (value - mean) / scale for beta, value, mean, scale in zip(
            fit["coefficients"][1:], features, fit["feature_means"], fit["feature_scales"]))


def residual_probabilities(fit, features, current_price, base):
    """Map residuals to rounded integer price events, with a fixed weak prior.

    One pseudo-count per event avoids unsupported exact 0/1 probabilities.
    At the official price floor, all lower latent prices also count as flat.
    These are probabilities of future spot direction, not profit or winning.
    """
    predicted = predict_ratio(fit, features)
    residuals = fit["calibration_residuals"]
    lower = (current_price - .5) / base - predicted
    upper = (current_price + .5) / base - predicted
    # The engine uses Python's round(), including ties to even. Residuals
    # exactly on a half-integer quote boundary must follow the same rule.
    even = current_price % 2 == 0
    down = (bisect_left if even else bisect_right)(residuals, lower) if current_price > 1 else 0
    below_upper = (bisect_right if even else bisect_left)(residuals, upper)
    flat = below_upper - down
    up = len(residuals) - below_upper
    down_prior = float(current_price > 1)
    total = len(residuals) + 2. + down_prior
    return ((up + 1.) / total, (flat + 1.) / total, (down + down_prior) / total)


def _valid_payload(model):
    return (isinstance(model, dict) and model.get("approved") is True
            and model.get("schema_version") == SCHEMA_VERSION
            and model.get("model_kind") == MODEL_KIND
            and model.get("feature_names") == list(FEATURE_NAMES)
            and model.get("class_names") == list(CLASS_NAMES)
            and isinstance(model.get("models"), dict))


def load_approved_model(path=None):
    model_path = Path(path) if path is not None else Path(__file__).with_name("mrlm_coefficients.json")
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return model if _valid_payload(model) else None


def predict_probabilities(model, obs, product, horizon, configuration=None):
    """Return (up, flat, down) only for approved, compatible, in-domain fits."""
    from .market import _params, price_at
    if (not _valid_payload(model) or horizon != model.get("horizon_turns")
            or model.get("training_configuration") != configuration_signature(configuration, obs.get("market"))):
        return None
    fit = model["models"].get(product.upper(), {})
    if not isinstance(fit, dict) or fit.get("approved") is not True:
        return None
    features = public_features(obs, product, horizon, configuration)
    coefficients, means, scales, residuals = (fit.get("coefficients"), fit.get("feature_means"),
                                             fit.get("feature_scales"), fit.get("calibration_residuals"))
    if (not all(isinstance(values, list) for values in (coefficients, means, scales, residuals))
            or len(coefficients) != len(features) + 1 or len(means) != len(features)
            or len(scales) != len(features) or len(residuals) < 30):
        return None
    if not all(isinstance(fit.get(key), list) for key in ("feature_min", "feature_max")):
        return None
    if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in [*coefficients, *means, *scales, *residuals])
            or any(scale <= 0 for scale in scales)
            or any(a > b for a, b in zip(residuals, residuals[1:]))
            or not within_training_domain(fit, features)):
        return None
    market, cfg = obs.get("market", {}), configuration or {}
    params = _params(product.upper(), market.get("params") or cfg.get("marketParams"))
    current = market.get("prices", {}).get(product.upper(), price_at(
        product.upper(), market.get("inventory", {}).get(product.upper(), params["I0"]), params))
    if (not isinstance(current, (int, float)) or not math.isfinite(current)
            or current < 1 or int(current) != current):
        return None
    prediction = predict_ratio(fit, features)
    if not math.isfinite(prediction):
        return None
    return residual_probabilities(fit, features, current, max(1., params["base"]))
