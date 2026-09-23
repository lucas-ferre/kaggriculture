"""Public-information market forecasts and the pinned official price curve.

Price quotes and isolated sale proceeds reproduce vendor/kaggriculture exactly.
Forecasts use bounded, deterministic scenarios or quadrature; eligible models
can replace spot-direction probabilities with held-out MRLM calibration.
Opponent sales timing, hidden stock, future planting and care remain unknown.
Public tiles provide a production prior; recent inventory changes,
less town consumption and our known orders, update that prior. No opponent
private inventory or random episode seed is read.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import random
from statistics import fmean

from .numerics import NORMAL_NODES_5, NORMAL_WEIGHTS_5, trailing_quadratic_slope, weighted_quantile


_ROWS = {
    "WHEAT": (25, 400, "sqrt", .80, "log", .20),
    "CARROT": (35, 450, "hinge", 1., "sqrt", .70),
    "TOMATO": (60, 200, "hinge", .40, "sqrt", .60),
    "STRAWBERRY": (120, 100, "sqrt", .70, "linear", 1.60),
    "MELON": (250, 300, "log", .20, "sq", 3.60),
    "EGG": (50, 332, "hinge", .40, "log", .20),
    "MILK": (160, 122, "sqrt", .60, "linear", 1.60),
    "WOOL": (200, 105, "log", .20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", .40, "linear", .40),
}
MARKET_PARAMS = {
    item: dict(base=r[0], I0=10000, T=r[1], below_func=r[2],
               below_target=r[3], above_func=r[4], above_target=r[5])
    for item, r in _ROWS.items()
}
PRODUCTS = tuple(MARKET_PARAMS)
SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_CROPS = {
    "WHEAT": (2, 4, 0, 6), "CARROT": (2, 3, 0, 4),
    "TOMATO": (8, 8, 1, 4), "STRAWBERRY": (10, 10, 2, 4),
    "MELON": (10, 12, 0, 6),
}
_ANIMALS = {"GOOSE": ("EGG", 4, 1), "COW": ("MILK", 8, 2),
            "SHEEP": ("WOOL", 6, 3)}


def _params(product, overrides=None):
    result = MARKET_PARAMS[product].copy()
    if overrides:
        # Accept both a marketParams-style product map and a single curve patch.
        patch = overrides.get(product, overrides)
        if isinstance(patch, dict):
            result.update({k: v for k, v in patch.items() if k in result})
    return result


def _shape(func, x, anchor):
    x = max(0., x)
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1. + x)
    if func == "log10":
        return math.log10(1. + x)
    if func == "hinge" and anchor and anchor > 0:
        u = x / anchor
        return u + 8. * max(0., u - 1.) ** 2
    return x


def _price(inventory, p):
    side = "below" if inventory < p["I0"] else "above"
    fn = p[side + "_func"]
    denom = _shape(fn, p["T"], p["T"])
    amp = p[side + "_target"] * p["base"] / denom
    sign = 1 if side == "below" else -1
    return max(1, int(round(p["base"] + sign * amp *
                            _shape(fn, abs(inventory - p["I0"]), p["T"]))))


def price_at(product, inventory, overrides=None):
    """Official pre-sale integer quote, including Python rounding and floor 1.

    ``overrides`` may be the sparse configuration ``marketParams`` mapping,
    observation ``market.params``, or a single product's curve patch.
    """
    return _price(inventory, _params(product.upper(), overrides))


def _sell(inventory, quantity, params):
    revenue = 0
    remaining = max(0, int(quantity))
    while remaining:
        quote = _price(inventory, params)
        if quote == 1:
            # The official engine pays, but does not add supply at the floor.
            return revenue + remaining, inventory
        revenue += quote
        inventory += 1
        remaining -= 1
    return revenue, inventory


def sale_revenue(product, inventory, quantity, overrides=None):
    """Exact proceeds for our isolated sequential sale, including own slippage.

    Concurrent opponent orders can alter proceeds; this function deliberately
    quotes only the specified sale. A nonpositive quantity yields zero.
    """
    return _sell(inventory, quantity, _params(product.upper(), overrides))[0]


def sale_quantity_at_reserve(product, inventory, max_quantity, reserve_price, overrides=None):
    """Exact largest integer lot whose marginal unit meets a reserve price.

    Official quotes are rounded and floored: their derivative is undefined or
    zero, so discrete bisection is safer than Newton iteration. The monotone
    official curves make this O(log(max_quantity)) quote evaluations. Ties are
    sold. At reserve <= 1, all units qualify even though floor sales add no
    inventory. This optimizes an isolated lot, not an adversarial whole game.
    """
    if not math.isfinite(reserve_price):
        raise ValueError("reserve_price must be finite")
    quantity = max(0, int(max_quantity))
    if reserve_price <= 1:
        return quantity
    params = _params(product.upper(), overrides)
    low, high = 0, quantity
    while low < high:
        middle = (low + high + 1) // 2
        if _price(inventory + middle - 1, params) >= reserve_price:
            low = middle
        else:
            high = middle - 1
    return low


def _add_supply(inventory, quantity, params):
    """Apply unit-sale inventory effects without a per-unit simulation."""
    n = max(0, int(round(quantity)))
    if not n or _price(inventory, params) == 1:
        return inventory
    if _price(inventory + n - 1, params) > 1:
        return inventory + n
    # Find the first floor quote. Positive targets yield monotone curves.
    low, high = 0, n
    while low < high:
        middle = (low + high) // 2
        if _price(inventory + middle, params) > 1:
            low = middle + 1
        else:
            high = middle
    return inventory + low


def _ticks(start, stop, interval):
    """Consumption events in [start, stop), including the action at step zero."""
    return 0 if stop <= start else ((stop - 1) // interval -
                                  (start - 1) // interval)


def _shop_units(product, shops):
    return sum((2 if len(SHOPS[name]) == 1 else 1)
               for name in shops if name in SHOPS and product in SHOPS[name])


@dataclass(frozen=True)
class Forecast:
    """Spot price scenarios; revenue additionally includes our sale's slippage.

    p10/p50/p90 are empirical quantiles; directional probabilities compare each
    scenario's integer spot quote to the currently observed quote. ``samples``
    is the scenario count, not the number of historical observations. Confidence
    strings disclose whether probabilities use empirical scenarios or an
    approved MRLM model calibrated on independent local episodes.
    """

    expected_price: float
    p10: float
    p50: float
    p90: float
    prob_up: float
    prob_flat: float
    prob_down: float
    expected_revenue: float
    samples: int
    confidence: str


class MarketModel:
    """Instance-local market state with at most 32 deterministic scenarios.

    Known shops consume on exact engine tick boundaries. Future shop choices
    use the official uniform-with-replacement rule and eight-instance cap.
    Opponent public production, possible unseen stock and sales realization
    are explicit heuristic priors. Residual flow is recency weighted and shrunk
    toward these priors; floor-censored transitions do not teach a zero supply
    rate. Future supply/town interactions are resolved in day/event blocks,
    so intraday opponent order timing and simultaneous quote effects remain
    approximate. No per-turn Monte Carlo rollout is used.

    Call update once for each observation. ``own_previous_orders`` describes
    reserved, executable orders from the immediately preceding observation,
    not the current action. For SELL the prior own shed plus carried inventory
    allows same-turn deposits. Failed orders, unexpected pickups, and opponent
    interleaving can make that correction approximate; omit invalid orders.
    A step/day rollback, changed episode id, or changed player resets history.
    A new game with an indistinguishable identical step-zero observation is
    equivalent to the existing empty history. A fresh instance is recommended.
    """

    def __init__(self, config=None, history_limit=96):
        self.config = dict(config or {})
        self.forecasting_method = self.config.get("forecastingMethod", "scenarios")
        if self.forecasting_method not in {"scenarios", "quadrature"}:
            raise ValueError("forecastingMethod must be scenarios or quadrature")
        self.causal_trend = bool(self.config.get("causalTrend", self.forecasting_method == "quadrature"))
        if "quantileModel" not in self.config:
            from .quantile import load_approved_model
            self.config["quantileModel"] = load_approved_model()
        if "mrlmModel" not in self.config:
            from .mrlm import load_approved_model
            self.config["mrlmModel"] = load_approved_model()
        self.turns_per_day = max(1, int(self.config.get("turnsPerDay", 24)))
        self.shop_interval = max(1, int(self.config.get("townShopSellInterval", 4)))
        self.center_interval = max(1, int(self.config.get("townCenterSellInterval", 24)))
        self.unlock_days = max(1, int(self.config.get("townShopUnlockInterval", 3)))
        self.last_action = max(0, int(self.config.get("episodeSteps", 720)) - 2)
        self.history_limit = max(1, int(history_limit))
        self._config_params = {p: _params(p, self.config.get("marketParams"))
                               for p in PRODUCTS}
        self.params = {p: d.copy() for p, d in self._config_params.items()}
        self._history = {p: deque(maxlen=self.history_limit) for p in PRODUCTS}
        self._snapshot = None
        self._contiguous_updates = 0
        self._cache = {}
        self._inventory_cache = {}
        self._revenue_prefixes = {}
        self._batch_cache = {}
        self._seed = 0

    def _demand(self, product, start, stop, shops):
        shops_demand = _shop_units(product, shops) * _ticks(
            start, stop, self.shop_interval)
        center = 0 if product == "FERTILIZER" else _ticks(
            start, stop, self.center_interval)
        return shops_demand + center

    def expected_town_demand(self, product, start, stop, shops=None):
        """Exact mean consumption under uniform-with-replacement future shops.

        Known shops are retained; each future unlock adds the mean per-shop
        consumption, up to eight instances. This is expected demand, not the
        price expectation through the nonlinear rounded market curve.
        """
        product = product.upper()
        shops = tuple(shops if shops is not None else self._snapshot["shops"])
        expected = float(self._demand(product, start, stop, shops))
        interval = self.unlock_days * self.turns_per_day
        future = range((start // interval + 1) * interval, stop, interval)
        mean_units = sum(_shop_units(product, (shop,)) for shop in SHOPS) / len(SHOPS)
        for _, unlock in zip(range(max(0, 8 - len(shops))), future):
            expected += mean_units * _ticks(unlock, stop, self.shop_interval)
        return expected

    def _own_flow(self, previous, orders):
        inventories = previous["inventory"].copy()
        available = previous["own_stock"].copy()
        max_orders = max(1, int(self.config.get("maxMarketOrdersPerTurn", 10)))
        for order in (orders or [])[:max_orders]:
            if not isinstance(order, (list, tuple)) or len(order) < 3:
                continue
            op, product = order[:2]
            if product not in inventories:
                continue
            try:
                n = max(0, int(order[2]))
            except (ValueError, TypeError):
                continue
            if op == "SELL":
                n = min(n, available.get(product, 0))
                inventories[product] = _add_supply(
                    inventories[product], n, previous["params"][product])
                available[product] = available.get(product, 0) - n
            elif op == "BUY_PRODUCT" and product in ("WHEAT", "FERTILIZER"):
                inventories[product] -= n
                available[product] = available.get(product, 0) + n
        return {p: inventories[p] - previous["inventory"][p] for p in PRODUCTS}

    def update(self, obs, own_previous_orders=None):
        """Copy only relevant public state and this player's own order capacity."""
        day = int(obs.get("day", 0))
        step = int(obs.get("step", day * self.turns_per_day + obs.get("hour", 0)))
        player = int(obs.get("player", 0))
        episode = obs.get("episode_id", obs.get("episodeId"))
        market = obs.get("market", {}) or {}
        params = {p: d.copy() for p, d in self._config_params.items()}
        for p, patch in (market.get("params", {}) or {}).items():
            if p in params and isinstance(patch, dict):
                params[p].update(patch)
        inventory = {p: int(market.get("inventory", {}).get(p, params[p]["I0"]))
                     for p in PRODUCTS}
        prices = {p: int(market.get("prices", {}).get(p, _price(inventory[p], params[p])))
                  for p in PRODUCTS}
        shops = tuple(obs.get("town", {}).get("unlocked_shops", []))
        farms = obs.get("farms", []) or []
        opponent = farms[1 - player] if len(farms) > 1 else {}
        tile_fields = ("kind", "crop", "animal", "planted_day", "placed_day",
                       "yield_units", "max_lifespan_step", "fertilized_until_day",
                       "consecutive_unwatered", "consecutive_unfed", "watered_today",
                       "fed_today", "pending_care_bonus", "fertilizer_available")
        tiles = [{k: tile[k] for k in tile_fields if k in tile}
                 for row in opponent.get("tiles", []) for tile in row
                 if isinstance(tile, dict)]
        private = obs.get("private", {}) or {}
        own_stock = dict(private.get("shed", {}) or {})
        for carried in private.get("inventories", []) or []:
            for p, n in carried.items():
                own_stock[p] = own_stock.get(p, 0) + n
        snapshot = dict(step=step, day=day, player=player, episode=episode,
                        inventory=inventory, prices=prices, shops=shops,
                        tiles=tiles, own_stock=own_stock, params=params)
        previous = self._snapshot
        reset = previous is not None and (
            step < previous["step"] or day < previous["day"] or
            player != previous["player"] or episode != previous["episode"] or
            params != previous["params"] or len(shops) < len(previous["shops"]))
        if reset:
            for history in self._history.values():
                history.clear()
            previous = None
        if previous is None or step - previous["step"] != 1:
            self._contiguous_updates = 0
        else:
            self._contiguous_updates += 1
        if previous is not None and step > previous["step"]:
            gap = step - previous["step"]
            # We know only the last issued order list, so missing observations
            # cannot yield an identifiable external-flow sample.
            if gap == 1:
                own_flow = self._own_flow(previous, own_previous_orders)
                for p in PRODUCTS:
                    censored = previous["prices"][p] <= 1 or prices[p] <= 1
                    residual = (inventory[p] - previous["inventory"][p] +
                                self._demand(p, previous["step"], step,
                                             previous["shops"]) - own_flow[p])
                    # Non-buyable products cannot leave the market except via
                    # the already removed town demand. Negative residuals mean
                    # an uncertain own-order correction, not opponent demand.
                    uncertain = p not in ("WHEAT", "FERTILIZER") and residual < 0
                    self._history[p].append((float(residual), censored or uncertain))
        self.params = params
        self._snapshot = snapshot
        self._cache.clear()
        self._inventory_cache.clear()
        self._revenue_prefixes.clear()
        self._batch_cache.clear()
        seed_state = {k: snapshot[k] for k in ("step", "player", "inventory", "shops", "tiles")}
        packed = json.dumps(seed_state, sort_keys=True, separators=(",", ":"))
        self._seed = int.from_bytes(hashlib.blake2b(packed.encode(), digest_size=8).digest(), "big")

    def _public_potential(self, product, horizon):
        """Approximate saleable public production under continued good care.

        This is a supply prior, not knowledge that an opponent will harvest,
        transport, sell, replant or retain livestock for the whole horizon.
        """
        s = self._snapshot
        now = s["day"]
        end = (s["step"] + horizon) // self.turns_per_day
        days = max(0, end - now)
        units, animals = 0., 0
        for tile in s["tiles"]:
            animal = tile.get("animal")
            if animal in _ANIMALS:
                animals += 1
                item, first, interval = _ANIMALS[animal]
                if product == "FERTILIZER":
                    units += int(tile.get("fertilizer_available", False)) + days
                if product == item:
                    start_day = int(tile.get("placed_day", now)) + first
                    produced = max(0, (end - start_day) // interval + 1)
                    already = max(0, (now - start_day) // interval + 1)
                    units += max(0, tile.get("yield_units", 0)) + produced - already
                    if produced > already:
                        units += max(0, tile.get("pending_care_bonus", 0))
                continue
            if tile.get("kind") != "PLANT" or tile.get("crop") != product:
                continue
            first, peak, interval, cap = _CROPS[product]
            planted = int(tile.get("planted_day", now))
            age = now - planted
            end_age = end - planted
            if end_age < first:
                continue
            visible = max(0, tile.get("yield_units", 0))
            if interval:
                before = min(4, max(0, (age - first) // interval + 1))
                after = min(4, max(0, (end_age - first) // interval + 1))
                units += visible + after - before
            else:
                mls = tile.get("max_lifespan_step", -1)
                if mls >= 0 and s["step"] >= mls and visible == 0:
                    continue
                # The visible yield already includes today's WATER bonus.
                # A watered plant cannot gain that same bonus a second time.
                next_watering_age = age + int(bool(tile.get("watered_today", False)))
                bonus_days = max(0, min(end_age, peak) -
                                 max(next_watering_age, (peak + 1) // 2) + 1)
                units += min(cap, max(1, visible) + bonus_days)
        # Wheat bought as feed is uncertain: some opponents grow their own.
        feed = .4 * animals * horizon / self.turns_per_day if product == "WHEAT" else 0.
        return max(0., units), feed

    def _trend(self, product):
        valid = [rate for rate, censored in self._history[product] if not censored]
        if not valid:
            return 0., 0., 0.
        weights = [.97 ** (len(valid) - i - 1) for i in range(len(valid))]
        total = sum(weights)
        mean = sum(w * rate for w, rate in zip(weights, valid)) / total
        variance = sum(w * (rate - mean) ** 2 for w, rate in zip(weights, valid)) / total
        # Keep uncertainty for quiet recent windows: hidden inventory is not
        # observable, and a sale can happen after a long silent period.
        deviation = math.sqrt(variance)
        recent = list(self._history[product])[-6:]
        if (self.causal_trend and self._contiguous_updates >= 6 and len(recent) == 6
                and not any(censored for _, censored in recent)):
            # Integrate corrected residual flows into seven causal cumulative
            # supply levels. Never bridge floor-censored/missing observations.
            cumulative = [0.0]
            for rate, _ in recent:
                cumulative.append(cumulative[-1] + rate)
            slope = trailing_quadratic_slope(cumulative)
            rates = [rate for rate, _ in recent]
            slope = max(min(rates) - deviation, min(max(rates) + deviation, slope))
            mean = .75 * mean + .25 * slope
        return mean, deviation, min(.85, len(valid) / (len(valid) + 12.))

    def public_pulses(self, product, horizon):
        """Potential sale batches from visible maturity, never hidden holdings.

        Times are offsets from this observation. Collection, transport and care
        are uncertain; scenario realization and delay are applied separately.
        """
        s, tpd = self._snapshot, self.turns_per_day
        end_day = (s["step"] + horizon) // tpd
        pulses = {}
        def add(day, quantity):
            offset = max(1, day * tpd - s["step"])
            if offset <= horizon and quantity > 0:
                pulses[offset] = pulses.get(offset, 0.) + quantity
        for tile in s["tiles"]:
            animal = tile.get("animal")
            if animal in _ANIMALS:
                item, first, interval = _ANIMALS[animal]
                if product == "FERTILIZER":
                    if tile.get("fertilizer_available"):
                        add(s["day"], 1)
                    for day in range(s["day"] + 1, end_day + 1):
                        add(day, 1)
                elif product == item:
                    add(s["day"], max(0, tile.get("yield_units", 0)))
                    first_day = int(tile.get("placed_day", s["day"])) + first
                    bonus = max(0, tile.get("pending_care_bonus", 0))
                    for day in range(max(s["day"] + 1, first_day), end_day + 1):
                        if (day - first_day) % interval == 0:
                            add(day, 1 + bonus + .5 * interval)
                            bonus = 0
                continue
            if tile.get("kind") != "PLANT" or tile.get("crop") != product:
                continue
            first, peak, interval, cap = _CROPS[product]
            planted = int(tile.get("planted_day", s["day"]))
            visible = max(0, tile.get("yield_units", 0))
            if interval:
                if s["day"] >= planted + first:
                    add(s["day"], visible)
                for n in range(4):
                    day = planted + first + n * interval
                    if day > s["day"]:
                        add(day, 1)
            else:
                if tile.get("max_lifespan_step", -1) >= 0 and tile["max_lifespan_step"] < s["step"] and not visible:
                    continue
                # Melon reaches its six-unit unfertilized cap at age ten;
                # max_yield_day=12 is its lifespan, not a reason to wait.
                harvest_age = first if product == "MELON" else max(first, peak)
                harvest_day = max(s["day"], planted + harvest_age)
                age = s["day"] - planted
                next_water = age + int(bool(tile.get("watered_today", False)))
                growth = max(0, min(peak, max(age, harvest_age)) - max(next_water, (peak + 1) // 2) + 1)
                add(harvest_day, min(cap, max(1, visible) + growth))
        return sorted(pulses.items())

    def _scenario_paths(self, product, horizons):
        """Coherent exogenous paths shared by every dated batch in a plan."""
        horizons = tuple(sorted(set(horizons)))
        key = (product, horizons)
        if key in self._inventory_cache:
            return self._inventory_cache[key]
        s = self._snapshot
        horizon = max(horizons, default=0)
        if not horizon:
            return [{0: s["inventory"][product]}]
        if self.forecasting_method == "quadrature":
            result = self._quadrature_paths(product, horizons)
            self._inventory_cache[key] = result
            return result
        seed = self._seed ^ int.from_bytes(hashlib.blake2b(
            f"{product}:{horizon}".encode(), digest_size=8).digest(), "big")
        rng = random.Random(seed)
        start, stop = s["step"], s["step"] + horizon
        unlock_interval = self.unlock_days * self.turns_per_day
        unlocks = list(range((start // unlock_interval + 1) * unlock_interval,
                             stop, unlock_interval))[:max(0, 8 - len(s["shops"]))]
        # At most 31 daily blocks over the maximum 720-turn forecast, plus
        # unlock events. The event count scales with days, not action turns.
        block = max(24, self.turns_per_day)
        base_boundaries = {start, stop, *range((start // block + 1) * block, stop, block),
                           *unlocks, *(start + h for h in horizons)}
        pulses = self.public_pulses(product, horizon)
        _, feed = self._public_potential(product, horizon)
        mean, deviation, weight = self._trend(product)
        result = []
        for _ in range(32):
            shops = list(s["shops"])
            # Broad, explicitly uncalibrated realization/hidden-production prior.
            hidden = rng.gammavariate(1.2, .02 * horizon)
            realized = rng.betavariate(2., 1.5)
            # Keep public crop maturity as discrete pulses even with history.
            visible_weight = max(.35, 1. - weight)
            events = {}
            for offset, quantity in pulses:
                at = start + offset + rng.randint(0, min(6, self.turns_per_day // 2))
                if at <= stop:
                    events[at] = events.get(at, 0.) + quantity * realized * visible_weight
            flow = ((1. - weight) * (hidden - feed * rng.uniform(0., 2.)) + weight * mean * horizon +
                    rng.gauss(0., 1.) * weight * max(.05, deviation) * math.sqrt(horizon * 4.))
            if product not in ("WHEAT", "FERTILIZER"):
                flow = max(0., flow)
            inv = s["inventory"][product]
            cumulative_flow = 0
            path = {0: inv}
            boundaries = sorted(base_boundaries | set(events))
            for left, right in zip(boundaries, boundaries[1:]):
                if left in unlocks:
                    shops.append(rng.choice(tuple(sorted(SHOPS))))
                demand = self._demand(product, left, right, shops)
                target_flow = round(flow * (right - start) / horizon)
                supply = target_flow - cumulative_flow
                cumulative_flow = target_flow
                # Sampling within-block order represents timing uncertainty;
                # floor sales never create invisible excess public inventory.
                before = rng.randint(0, demand) if demand else 0
                inv -= before
                inv = (_add_supply(inv, supply, self.params[product])
                       if supply > 0 else inv + supply)
                inv -= demand - before
                inv = _add_supply(inv, events.get(right, 0), self.params[product])
                if right - start in horizons:
                    path[right - start] = inv
            result.append(path)
        self._inventory_cache[key] = result
        return result

    def _quadrature_paths(self, product, horizons):
        """Five weighted paths from a one-factor normal external-flow proxy.

        Shops contribute their analytic expected demand; visible harvests stay
        as dated pulses. This is deterministic numerical integration of an
        approximate supply model, not exact game integration or 32 independent
        scenarios. Shop/outcome multimodality and sale timing remain uncertain.
        """
        s, horizon = self._snapshot, max(horizons)
        start, stop = s["step"], s["step"] + horizon
        pulses = self.public_pulses(product, horizon)
        _, feed = self._public_potential(product, horizon)
        mean, deviation, weight = self._trend(product)
        visible_weight = max(.35, 1. - weight)
        realized = 2.0 / 3.5
        events = {}
        for offset, quantity in pulses:
            at = start + offset + min(3, self.turns_per_day // 2)
            if at <= stop:
                events[at] = events.get(at, 0.0) + quantity * realized * visible_weight
        mu = (1. - weight) * (.024 * horizon - feed) + weight * mean * horizon
        variance = ((1. - weight) ** 2 * (1.2 * (.02 * horizon) ** 2 + feed ** 2 / 3.)
                    + weight ** 2 * max(.05, deviation) ** 2 * horizon * 4.)
        sigma = math.sqrt(max(0.0, variance))
        block = max(24, self.turns_per_day)
        boundaries = sorted({start, stop, *range((start // block + 1) * block, stop, block),
                             *(start + h for h in horizons), *events})
        expected_demand = {at: self.expected_town_demand(product, start, at, s["shops"])
                           for at in boundaries}
        result = []
        for node in NORMAL_NODES_5:
            flow = mu + sigma * node
            if product not in ("WHEAT", "FERTILIZER"):
                flow = max(0., flow)
            inv, cumulative = s["inventory"][product], 0
            path = {0: inv}
            for left, right in zip(boundaries, boundaries[1:]):
                target = round(flow * (right - start) / horizon)
                supply, cumulative = target - cumulative, target
                demand = expected_demand[right] - expected_demand[left]
                inv -= demand / 2.0
                inv = _add_supply(inv, supply, self.params[product]) if supply > 0 else inv + supply
                inv -= demand / 2.0
                inv = _add_supply(inv, events.get(right, 0), self.params[product])
                if right - start in horizons:
                    path[right - start] = inv
            result.append(path)
        return result

    def _weights(self, count):
        if self.forecasting_method == "quadrature" and count == 5:
            return NORMAL_WEIGHTS_5
        return (1.0 / count,) * count

    def _weighted_mean(self, values):
        return sum(value * weight for value, weight in zip(values, self._weights(len(values))))

    def _distribution_quantile(self, values, tau):
        if self.forecasting_method == "quadrature":
            return weighted_quantile(values, self._weights(len(values)), tau)
        return self._quantile(values, tau)

    def _scenario_inventories(self, product, horizon):
        return [path[horizon] for path in self._scenario_paths(product, (horizon,))]

    @staticmethod
    def _quantile(values, tau):
        ordered = sorted(values)
        rank = (len(ordered) - 1) * tau
        low = int(rank)
        high = min(len(ordered) - 1, low + 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)

    def _batch_revenues(self, product, batches):
        if self._snapshot is None:
            self.update({})
        product = product.upper()
        if product not in self.params:
            raise KeyError(product)
        limit = min(720, max(0, self.last_action - self._snapshot["step"]))
        combined = {}
        for horizon, quantity in batches:
            h = min(limit, max(0, int(horizon)))
            combined[h] = combined.get(h, 0) + max(0, int(quantity))
        batches = tuple((h, q) for h, q in sorted(combined.items()) if q)
        key = product, batches
        if key in self._batch_cache:
            return self._batch_cache[key]
        total = sum(q for _, q in batches)
        revenues = []
        for path in self._scenario_paths(product, tuple(h for h, _ in batches) or (0,)):
            added, revenue = 0, 0
            for horizon, quantity in batches:
                inv = path[horizon] + added
                revenue += self._revenue_at(product, inv, quantity)
                # Prior own sales persist in stock. Demand is already in the
                # exogenous path; the floor still prevents phantom own supply.
                added += _add_supply(inv, quantity, self.params[product]) - inv
            revenues.append(revenue)
        self._batch_cache[key] = revenues, total
        return revenues, total

    def revenue_quantile(self, product, batches, tau=.25):
        """Empirical scenario quantile, not a fitted quantile regression."""
        if not 0 <= tau <= 1:
            raise ValueError("tau must lie in [0, 1]")
        revenues, _ = self._batch_revenues(product, batches)
        return self._distribution_quantile(revenues, tau)

    def _public_observation(self):
        """Reconstruct only the public features retained by update()."""
        s = self._snapshot
        return {"step": s["step"], "day": s["day"], "player": 0,
                "market": {"inventory": s["inventory"], "prices": s["prices"], "params": s["params"]},
                "town": {"unlocked_shops": s["shops"]}, "farms": [{}, {"tiles": [s["tiles"]]}]}

    def price_quantile(self, product, horizon_turns, tau=.25):
        """Validated price-only model when eligible, otherwise empirical prices.

        A learned future spot quantile is never substituted for the quantile of
        multi-unit or multi-date sale revenue. Those have distinct targets.
        ``last_quantile_source`` makes the selected source inspectable.
        """
        from .quantile import predict_price_quantile
        if not 0 <= tau <= 1:
            raise ValueError("tau must lie in [0, 1]")
        if self._snapshot is None:
            self.update({})
        product = product.upper()
        horizon = min(720, max(0, int(horizon_turns)),
                      max(0, self.last_action - self._snapshot["step"]))
        public_obs = self._public_observation()
        predicted = predict_price_quantile(self.config.get("quantileModel"), public_obs,
                                           product, horizon, tau, self.config)
        if predicted is not None:
            self.last_quantile_source = "validated_local_price_quantile"
            return predicted
        self.last_quantile_source = "empirical_" + self.forecasting_method
        inventories = self._scenario_inventories(product, horizon)
        return self._distribution_quantile([_price(inv, self.params[product]) for inv in inventories], tau)

    def forecast_batches(self, product, batches):
        """Chronological lot proceeds with persistent own market impact.

        Quantiles here describe aggregate revenue per unit, unlike forecast's
        single-date spot quantiles. Exogenous floor interactions with prior own
        sales remain approximate; this is not a full competitive game rollout.
        """
        product = product.upper()
        revenues, quantity = self._batch_revenues(product, batches)
        unit = [value / max(1, quantity) for value in revenues]
        current = self._snapshot["prices"][product]
        n = len(unit)
        return Forecast(self._weighted_mean(unit), self._distribution_quantile(unit, .1),
                        self._distribution_quantile(unit, .5), self._distribution_quantile(unit, .9),
                        self._weighted_mean([float(x > current) for x in unit]),
                        self._weighted_mean([float(x == current) for x in unit]),
                        self._weighted_mean([float(x < current) for x in unit]),
                        self._weighted_mean(revenues), n,
                        "dated_quadrature_normal_approximation_uncalibrated" if self.forecasting_method == "quadrature"
                        else "dated_batches_uncalibrated")

    def should_lead_sale(self, product, quantity):
        """Sell available premium stock before a plausible imminent supply pulse.

        Both agents' same-turn units are quoted in lockstep by the engine. This
        only compares selling now to a later turn; it confers no execution priority.
        """
        if product not in {"MELON", "MILK", "WOOL", "STRAWBERRY"} or quantity <= 0:
            return False
        s = self._snapshot
        if self._demand(product, s["step"], s["step"] + 1, s["shops"]):
            return False
        current = sale_revenue(product, s["inventory"][product], quantity, self.params)
        upcoming = self.forecast(product, 1, quantity)
        visible = bool(self.public_pulses(product, 1))
        return visible or upcoming.expected_revenue < current

    def _revenue_at(self, product, inventory, quantity):
        """Reuse unit-price prefixes when the planner tries many sale volumes."""
        prefix = self._revenue_prefixes.setdefault((product, inventory), [0])
        params = self.params[product]
        while len(prefix) <= quantity:
            sold = len(prefix) - 1
            quote = _price(inventory + sold, params)
            if quote == 1:
                return prefix[-1] + quantity - sold
            prefix.append(prefix[-1] + quote)
        return prefix[quantity]

    def forecast(self, product, horizon_turns, sale_quantity=1):
        """Forecast up to the final actionable turn, with cached sale scenarios.

        Probabilities use an approved MRLM residual model when eligible and
        otherwise empirical scenarios; neither estimates winning the game.
        ``expected_revenue`` assumes our sale executes at
        the horizon, after other modeled net flows, and includes own slippage.
        """
        product = product.upper()
        if product not in MARKET_PARAMS:
            raise KeyError(product)
        if self._snapshot is None:
            self.update({})
        horizon = min(720, max(0, int(horizon_turns)),
                      max(0, self.last_action - self._snapshot["step"]))
        quantity = max(0, int(sale_quantity))
        key = (product, horizon, quantity)
        if key in self._cache:
            return self._cache[key]
        inventories = self._scenario_inventories(product, horizon)
        p = self.params[product]
        prices = [_price(inv, p) for inv in inventories]
        current = self._snapshot["prices"][product]
        size = len(prices)
        def quantile(q):
            return self._distribution_quantile(prices, q)
        valid = sum(not censored for _, censored in self._history[product])
        confidence = ("exact_current_quote" if not horizon else
                      "quadrature_normal_approximation_uncalibrated" if self.forecasting_method == "quadrature" else
                      "prior_uncalibrated" if valid < 3 else
                      "limited_history_uncalibrated" if valid < 24 else
                      "observed_history_uncalibrated")
        revenues = [self._revenue_at(product, inv, quantity) for inv in inventories]
        probabilities = (self._weighted_mean([float(x > current) for x in prices]),
                         self._weighted_mean([float(x == current) for x in prices]),
                         self._weighted_mean([float(x < current) for x in prices]))
        if self.config.get("mrlmModel") is not None and horizon:
            from .mrlm import predict_probabilities
            calibrated = predict_probabilities(self.config["mrlmModel"], self._public_observation(),
                                               product, horizon, self.config)
            if calibrated is not None:
                probabilities = calibrated
                confidence = "validated_mrlm_spot_probabilities_empirical_revenue"
        result = Forecast(self._weighted_mean(prices), quantile(.1), quantile(.5), quantile(.9),
                          *probabilities,
                          self._weighted_mean(revenues), size, confidence)
        self._cache[key] = result
        return result
