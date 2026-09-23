"""Public-state price quantile features and optional validated coefficients.

Training lives in scripts/train_quantile.py. Runtime uses only scalar arithmetic
and a bundled local JSON file; unavailable/unapproved models return None so the
market model can retain its empirical uncertainty distribution.
"""
import json
import math
from pathlib import Path

FEATURE_NAMES = ("inventory_deviation", "spot_price_ratio", "day_fraction",
                 "hour_fraction", "known_shop_units", "future_shop_units",
                 "opponent_crop_tiles", "opponent_animals", "visible_units")
SCHEMA_VERSION = 1


def configuration_signature(configuration=None, market=None):
    """Normalize the public rules that define feature meaning and price targets."""
    from .market import PRODUCTS, _params
    cfg, current = configuration or {}, market or {}
    defaults = {"turnsPerDay": 24, "episodeSteps": 720, "townShopUnlockInterval": 3,
                "townShopSellInterval": 4, "townCenterSellInterval": 24}
    result = {key: int(cfg.get(key, value)) for key, value in defaults.items()}
    curves = current.get("params") or cfg.get("marketParams")
    result["marketParams"] = {product: _params(product, curves) for product in PRODUCTS}
    return result


def within_training_domain(fit, features):
    lower, upper = fit.get("feature_min", []), fit.get("feature_max", [])
    if len(lower) != len(features) or len(upper) != len(features):
        return False
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*features, *lower, *upper]):
        return False
    return all(lo - .25 * max(1., hi - lo) <= value <= hi + .25 * max(1., hi - lo)
               for value, lo, hi in zip(features, lower, upper))


def public_features(obs, product, horizon, configuration=None):
    """Features at decision time; never inspect actions, private stocks or future states."""
    from .market import SHOPS, _params, _shop_units, price_at
    cfg = configuration or {}
    tpd = max(1, int(cfg.get("turnsPerDay", 24)))
    step = int(obs.get("step", int(obs.get("day", 0)) * tpd + int(obs.get("hour", 0))))
    product = product.upper()
    market = obs.get("market", {})
    params = _params(product, market.get("params") or cfg.get("marketParams"))
    inventory = market.get("inventory", {}).get(product, params["I0"])
    spot = market.get("prices", {}).get(product, price_at(product, inventory, params))
    shops = tuple(obs.get("town", {}).get("unlocked_shops", []))
    interval = max(1, int(cfg.get("townShopUnlockInterval", 3))) * tpd
    new_shops = min(max(0, 8 - len(shops)), max(0, (step + int(horizon)) // interval - step // interval))
    average_shop = sum(_shop_units(product, (shop,)) for shop in SHOPS) / len(SHOPS)
    farms, player = obs.get("farms", []), int(obs.get("player", 0))
    other = farms[1 - player] if len(farms) > 1 else {}
    products = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
    crop_tiles = animals = visible = 0.0
    for row in other.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop_tiles += tile.get("kind") == "PLANT" and tile.get("crop") == product
            animals += tile.get("animal") in products
            if tile.get("crop") == product or products.get(tile.get("animal")) == product:
                visible += max(0, tile.get("yield_units", 0))
            elif product == "FERTILIZER" and tile.get("animal") in products:
                visible += bool(tile.get("fertilizer_available"))
    return [(inventory - params["I0"]) / max(1, params["T"]), spot / max(1, params["base"]),
            step / max(1, int(cfg.get("episodeSteps", 720)) - 1), (step % tpd) / tpd,
            float(_shop_units(product, shops)), new_shops * average_shop,
            crop_tiles, animals, visible]


def load_approved_model(path=None):
    """Load only a compatible payload that passed its recorded validation gate."""
    model_path = Path(path) if path is not None else Path(__file__).with_name("quantile_coefficients.json")
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (not isinstance(payload, dict) or payload.get("approved") is not True
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("feature_names") != list(FEATURE_NAMES)):
        return None
    return payload


def predict_price_quantile(model, obs, product, horizon, tau, configuration=None):
    """Return a validated scalar price prediction for its trained horizon/tau only."""
    from .market import _params
    if (not isinstance(model, dict) or model.get("approved") is not True
            or model.get("schema_version") != SCHEMA_VERSION
            or model.get("feature_names") != list(FEATURE_NAMES)
            or int(horizon) != model.get("horizon_turns")
            or not isinstance(model.get("quantile_tau"), (int, float))
            or abs(float(tau) - model["quantile_tau"]) > 1e-9
            or model.get("training_configuration") != configuration_signature(configuration, obs.get("market"))):
        return None
    fit = model.get("models", {}).get(product.upper(), {})
    if fit.get("approved") is not True:
        return None
    features = public_features(obs, product, horizon, configuration)
    coefficients, means, scales = (fit.get("coefficients", []), fit.get("feature_means", []),
                                    fit.get("feature_scales", []))
    if len(coefficients) != len(features) + 1 or len(means) != len(features) or len(scales) != len(features):
        return None
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [*coefficients, *means, *scales]):
        return None
    if any(scale <= 0 for scale in scales):
        return None
    standardized = [(value - mean) / scale for value, mean, scale in zip(features, means, scales)]
    # Reject extrapolation far outside the training feature range. This is an
    # eligibility check, not a claim that in-range predictions are calibrated.
    if not within_training_domain(fit, features):
        return None
    prediction = coefficients[0] + sum(beta * value for beta, value in zip(coefficients[1:], standardized))
    cfg = configuration or {}
    market = obs.get("market", {})
    base = _params(product.upper(), market.get("params") or cfg.get("marketParams"))["base"]
    return max(1.0, prediction * base) if math.isfinite(prediction) else None
