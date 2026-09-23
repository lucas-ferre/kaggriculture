"""Staged livestock investment from public prices and feasible local resources.

The small LP prices marginal animal tranches. It is a rolling approximation,
not a joint integer optimum for land, routes and the whole remaining season.
"""
from collections import Counter
import math

from .domain import ANIMALS, CROPS, LAND_PRICES, distance, fib_cost, nearest_shed
from .market import price_at, sale_revenue
from .optimization import simplex, branch_bound


def expansion_plan(obs, config, policy, plan, budget):
    """Price the next *fixed-order* quadrant as productive, compact capacity.

    Raw unlocked area overstates usable capacity when daily routes already
    consume the labor allowance. Count plots within a short shed trip, matching
    the scheduler's nearest-shed planting order. This radius and the margin
    haircut are policy approximations; LOCKED tiles are legally walkable and
    walking through them alone is never a reason to buy land.
    """
    from .planner import minimum_cycle_day

    debug = {"status": "disabled", "cost": 0, "projected_margin": 0.0}
    if not policy.enable_expansion:
        return False, debug
    farm = obs["farms"][obs["player"]]
    expansions = max(0, len(farm.get("unlocked_quadrants", ["NW"])) - 1)
    if expansions >= min(len(LAND_PRICES), policy.max_expansions):
        debug["status"] = "limit_reached"
        return False, debug
    tpd = max(1, int(config["turnsPerDay"]))
    size = len(farm["tiles"])
    margin = max(policy.logistics_margin, size + 2)
    # Reserve a whole day for opening and stocking the new area. Investment
    # must still finish the same minimum cycle used by crop/seed planning.
    last_day = (int(config["episodeSteps"]) - 3 - margin) // tpd
    available_days = last_day - int(obs["day"]) - 1
    candidates = [(crop, float(plan.scores.get(crop, 0)), count)
                  for crop, count in plan.targets.items()
                  if crop in CROPS and count > 0
                  and available_days >= minimum_cycle_day(crop, policy)
                  and math.isfinite(float(plan.scores.get(crop, 0)))
                  and float(plan.scores.get(crop, 0)) > 0]
    if not candidates:
        debug["status"] = "no_recoverable_production"
        return False, debug
    _, owned = animal_counts(obs)
    animals = sum(max(owned[a], int(plan.animal_targets.get(a, 0))) for a in ANIMALS)
    unlocked_quads = len(farm.get("unlocked_quadrants", ["NW"]))
    effective_workers = min(policy.target_workers, 8 if unlocked_quads <= 2 else (9 if unlocked_quads == 3 else 10))
    crop_capacity = min(policy.max_plots, max(0, int(
        (effective_workers * tpd * policy.labor_utilization - 6 * animals)
        / policy.labor_actions_per_plot_day)))
    if not crop_capacity:
        debug["status"] = "no_spare_crop_labor"
        return False, debug
    productive_demand = crop_capacity + animals
    # Three moves per leg on the default board leaves time for feeding,
    # watering and cargo handling. Larger boards retain a bounded local core.
    radius = max(1, min(3, size // 2 - 1))
    quadrant = ("NE", "SW", "SE")[expansions]
    compact_owned = compact_new = 0
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if distance((x, y), nearest_shed((x, y), size)) > radius:
                continue
            if tile != "LOCKED":
                compact_owned += 1
            else:
                name = ("N" if y < size // 2 else "S") + ("W" if x < size // 2 else "E")
                compact_new += name == quadrant
    extra = min(compact_new, max(0, productive_demand - compact_owned))
    cost = LAND_PRICES[expansions]
    # Use the selected crop mix, never a high score for an unselected crop.
    # Discount marginal forecasts and delayed occupancy instead of treating
    # every newly unlocked tile as an immediately working production unit.
    per_plot = sum(score * count for _, score, count in candidates) / sum(count for _, _, count in candidates)
    projected_margin = .65 * extra * per_plot * available_days / max(1, available_days + 1)
    private = obs["private"]
    counts = Counter(tile.get("crop") for row in farm["tiles"] for tile in row
                     if isinstance(tile, dict) and tile.get("kind") == "PLANT")
    seed_buffer = sum(max(0, count - counts[crop] - private.get("seeds", {}).get(crop, 0))
                      * CROPS[crop].seed_cost for crop, _, count in candidates)
    params = obs["market"].get("params") or config.get("marketParams")
    feed_buffer = purchase_cost("WHEAT", obs["market"]["inventory"]["WHEAT"], animals, params)
    wage_buffer = sum(fib_cost(i, int(config["farmHandCostMult"]))
                      for i in range(max(0, effective_workers - 1)))
    operating_buffer = seed_buffer + feed_buffer + wage_buffer
    debug.update(cost=cost, next_quadrant=quadrant, compact_owned=compact_owned,
                 compact_new=compact_new, productive_demand=productive_demand,
                 useful_new_plots=extra, projected_margin=round(projected_margin, 2),
                 operating_buffer=round(operating_buffer, 2), radius=radius)
    occupied = sum(isinstance(tile, dict) and (tile.get("kind") == "PLANT" or "animal" in tile)
                   for row in farm["tiles"] for tile in row)
    unlocked_total = sum(tile != "LOCKED" for row in farm["tiles"] for tile in row)
    if not extra:
        debug["status"] = "sufficient_compact_area"
    elif projected_margin < cost * 1.15:
        debug["status"] = "insufficient_projected_return"
    elif budget < cost + operating_buffer:
        debug["status"] = "operating_cash_reserved"
    elif int(obs["day"]) < 4:
        debug["status"] = "early_game_restricted"
    elif unlocked_total > 0 and occupied < unlocked_total * 0.70:
        debug["status"] = "insufficient_occupancy"
    else:
        debug["status"] = "buy"
    return debug["status"] == "buy", debug


def animal_counts(obs):
    farm = obs["farms"][obs["player"]]
    live = Counter(tile["animal"] for row in farm["tiles"] for tile in row
                   if isinstance(tile, dict) and tile.get("animal") in ANIMALS)
    owned = live.copy()
    private = obs["private"]
    for inventory in [private.get("shed", {}), *private.get("inventories", [])]:
        for animal in ANIMALS:
            owned[animal] += max(0, int(inventory.get(animal, 0)))
    return live, owned


def purchase_cost(product, inventory, quantity, params=None):
    """Exact isolated BUY_PRODUCT quote: each unit is priced after removal."""
    return sum(price_at(product, inventory - i - 1, params) for i in range(max(0, int(quantity))))


def livestock_plan(obs, config, policy, market):
    live, owned = animal_counts(obs)
    targets = dict(owned)
    debug = {"status": "disabled", "margins": {}, "reserved_cash": 0., "new_animals": 0}
    if not policy.enable_livestock:
        return targets, debug
    tpd = max(1, int(config["turnsPerDay"]))
    step = int(obs.get("step", obs["day"] * tpd + obs["hour"]))
    last_day = (int(config["episodeSteps"]) - 3) // tpd
    # Two calendar days cover buying, building, placement and initial care.
    place_day = int(obs["day"]) + 2
    days = max(0, last_day - place_day)
    farm = obs["farms"][obs["player"]]
    params = obs["market"].get("params") or config.get("marketParams")
    feed_quote = price_at("WHEAT", obs["market"]["inventory"]["WHEAT"] - 2 * (sum(owned.values()) + 3), params)
    # Keep a daily operating buffer and never use anticipated sale proceeds.
    budget = max(0., farm["money"] - policy.cash_reserve - feed_quote * (sum(owned.values()) + 3))
    free = sum(tile is None or isinstance(tile, dict) and tile.get("kind") in {"WEED", "COOP", "PASTURE"}
               and "animal" not in tile for row in farm["tiles"] for tile in row)
    staged = sum(owned.values()) - sum(live.values())
    free = max(0, free - staged)
    unlocked_quads = len(farm.get("unlocked_quadrants", ["NW"]))
    effective_workers = min(policy.target_workers, 8 if unlocked_quads <= 2 else (9 if unlocked_quads == 3 else 10))
    slots = min(free, max(0, 4 - staged), 3,
                max(0, int(effective_workers * tpd * policy.labor_utilization / 6) - sum(owned.values())))
    caps = {"COW": policy.max_cows, "SHEEP": policy.max_sheep, "GOOSE": policy.max_geese}
    columns, values, costs = [], [], []
    for animal, spec in ANIMALS.items():
        if days < spec["first"] + 2:
            continue
        for addition in range(1, min(slots, max(0, caps[animal] - owned[animal])) + 1):
            count = owned[animal] + addition
            def proceeds(n):
                batches = []
                for day in range(place_day + spec["first"], last_day + 1, spec["interval"]):
                    units = min(spec["cap"], spec["first"] if not batches else spec["interval"] + 1)
                    batches.append((max(0, day * tpd - step + 4), n * units))
                expected = market.forecast_batches(spec["product"], batches).expected_revenue
                quantile = market.revenue_quantile(spec["product"], batches, policy.risk_quantile)
                return expected - policy.risk_aversion * max(0., expected - quantile)
            total_animals = sum(owned.values()) + addition
            fertilizer = [(max(0, day * tpd - step + 6), total_animals)
                          for day in range(place_day + 1, last_day + 1)]
            before = [(h, total_animals - 1) for h, _ in fertilizer]
            manure = (market.forecast_batches("FERTILIZER", fertilizer).expected_revenue -
                      market.forecast_batches("FERTILIZER", before).expected_revenue)
            margin = (proceeds(count) - proceeds(count - 1) + manure - spec["cost"] -
                      days * feed_quote - days * 6 * policy.labor_cost_per_action)
            debug["margins"][f"{animal}:{count}"] = round(margin, 2)
            if margin <= 0 or not math.isfinite(margin):
                break
            columns.append(animal)
            values.append(margin)
            costs.append(spec["cost"])
    if not columns or slots <= 0:
        debug["status"] = "no_profitable_capacity"
        return targets, debug
    n = len(columns)
    rows = [[1.] * n, list(costs)] + [[float(i == j) for i in range(n)] for j in range(n)]
    bounds = [slots, budget] + [1.] * n
    result = (branch_bound(values, rows, bounds, node_limit=policy.integer_node_limit, incumbent=[0.] * n)
              if policy.optimizer == "branch_bound" else simplex(values, rows, bounds))
    debug["status"] = result.status
    debug["gap"] = getattr(result, "gap", None)
    debug["nodes"] = getattr(result, "nodes", 0)
    if not result.variables:
        return targets, debug
    selected = [i for i, value in enumerate(result.variables) if value >= 1 - 1e-7]
    spent = sum(costs[i] for i in selected)
    for i in sorted(range(n), key=lambda i: (-values[i] / costs[i], i)):
        if i not in selected and len(selected) < slots and spent + costs[i] <= budget:
            selected.append(i)
            spent += costs[i]
    for i in selected:
        targets[columns[i]] = targets.get(columns[i], 0) + 1
    debug.update(reserved_cash=spent, new_animals=len(selected))
    return targets, debug


def prioritize_market_orders(orders, obs, config, opponent=None, feed_urgent=False, current_shed=None):
    """Rank actual available sales by exposure to a shared-book price drop.

    Earlier order slots can lead different later slots. Same-slot simultaneous
    unit quotes remain symmetric; no private rival order queue is assumed.
    """
    params = obs["market"].get("params") or config.get("marketParams")
    shed = obs["private"].get("shed", {}) if current_shed is None else current_shed
    feed_quantity = sum(order[2] for order in orders if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT")
    feed_needs_sale_space = (feed_urgent and feed_quantity > max(0, int(config["shedCapacity"]) - sum(shed.values())))
    def key(pair):
        index, order = pair
        if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT" and feed_urgent:
            return (1 if feed_needs_sale_space else 0, 0., index)
        if order[0] == "SELL":
            _, product, quantity = order
            inventory = obs["market"]["inventory"][product]
            risk = max(0, int(opponent.supply_risk(product))) if opponent else 0
            revenue = sale_revenue(product, inventory, quantity, params)
            exposed = max(0, revenue - sale_revenue(product, inventory + risk, quantity, params))
            return (0 if feed_needs_sale_space else 1, -exposed, -revenue, index)
        return ({"BUY_PRODUCT": 2, "HIRE": 2, "BUY_LAND": 3,
                 "BUY_SEED": 4, "BUY_ANIMAL": 5}.get(order[0], 6), 0., index)
    ranked = sorted(enumerate(orders), key=key)
    return [order for _, order in ranked[:max(1, int(config["maxMarketOrdersPerTurn"]))]]
