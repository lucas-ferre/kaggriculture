"""Small, bounded forward models for observable crops and animal calendars.

Transitions match the pinned engine. Future watering and same-day harvesting
are execution assumptions, and valuation uses current public quotes, not a
claim that future market prices or available labor are known.
"""
from dataclasses import dataclass

from .domain import ANIMALS, CROPS, distance, nearest_shed


CROP_CAPS = {"WHEAT": 6, "CARROT": 4, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}
CROP_INTERVALS = {"TOMATO": 1, "STRAWBERRY": 2}


def animal_production_days(animal, placed_day, last_day):
    """Calendar days with harvestable production, including the last day."""
    rule = ANIMALS[animal]
    return list(range(int(placed_day) + rule["first"], int(last_day) + 1, rule["interval"]))


def next_animal_production(tile, after_day):
    """First production calendar day strictly after ``after_day``."""
    rule = ANIMALS[tile["animal"]]
    first = int(tile["placed_day"]) + rule["first"]
    if first > after_day:
        return first
    return first + ((int(after_day) - first) // rule["interval"] + 1) * rule["interval"]


def nightly_animal_production(tile, day, assume_fed=True):
    """Tonight's raw output, before the tile cap. Today's CARE is not included.

The engine pays the old bonus first, then adds today's fed-and-cared bonus to
the next production cycle. Without feeding the old bonus is lost on payout.
"""
    if next_animal_production(tile, day) != day + 1:
        return 0
    return 1 + (int(tile.get("pending_care_bonus", 0)) if assume_fed else 0)


def animal_care_can_pay(tile, day, last_day):
    # Today's newly banked bonus cannot be paid in tonight's production.
    if next_animal_production(tile, day + 1) > last_day:
        return False
    rule = ANIMALS[tile["animal"]]
    # Existing bonus already fills the next batch; extra care would be lost.
    return (next_animal_production(tile, day) == day + 1
            or int(tile.get("pending_care_bonus", 0)) < rule["cap"] - 1)


def projected_crop_harvests(tile, day, last_day, fertilize_today=False):
    """Project one observed plant, watered daily and harvested by our policy.

Existing yield is already inclusive of today's actions. One-time crop bonus
is applied by WATER; recurring crop output occurs at the following midnight.
Recurring caps bound held stock, while the four scheduled production events
remain finite. No replacement planting or invented fifth event is included.
"""
    crop = tile["crop"]
    rule, cap = CROPS[crop], CROP_CAPS[crop]
    planted = int(tile["planted_day"])
    units = max(0, int(tile.get("yield_units", 0)))
    until = int(tile.get("fertilized_until_day", -1))
    if fertilize_today:
        until = max(until, day + 2)
    harvests = []
    # Lifetimes are at most 17 days; never simulate an unbounded episode.
    stop = min(int(last_day), max(day, planted + rule.last + 1))
    for current in range(int(day), stop + 1):
        age = current - planted
        watered = bool(tile.get("watered_today")) if current == day else False
        if not rule.ongoing:
            window = (rule.last + 1) // 2
            if not watered and window <= age <= rule.last:
                units = min(cap, units + (2 if until >= current else 1))
            if units and age >= rule.first and (age >= rule.peak or current == last_day):
                harvests.append((current, units))
                break
        else:
            if units and age >= rule.first and (units >= 2 or age >= rule.last or current == last_day):
                harvests.append((current, units))
                units = 0
            production_age = current + 1 - planted - rule.first
            interval = CROP_INTERVALS[crop]
            if (current < last_day and production_age >= 0 and production_age % interval == 0
                    and production_age // interval < cap):
                units = min(cap, units + (2 if until >= current else 1))
    return harvests


@dataclass(frozen=True)
class FertilizerCandidate:
    position: tuple
    crop: str
    extra_units: int
    net_gain: float
    worker: int
    actions_needed: int
    deadline: int


def fertilizer_candidates(obs, config, policy):
    """Profitable, reachable applications, independent of current stock.

Computing opportunities without requiring shed stock also lets market policy
reserve fertilizer newly deposited/collected by this turn's physical actions.
Each candidate consumes exactly one fertilizer and one unique plant cell.
"""
    if not getattr(policy, "enable_fertilizer", True):
        return []
    tpd = max(1, int(config["turnsPerDay"]))
    day, hour = int(obs["day"]), int(obs["hour"])
    terminal = int(config["episodeSteps"]) - 2
    last_day = terminal // tpd
    if day >= last_day:
        return []
    farm = obs["farms"][obs["player"]]
    size = len(farm["tiles"])
    workers = [tuple(farm["farmer"]), *map(tuple, farm.get("hands", []))]
    inventories = obs["private"].get("inventories", [])
    prices = obs["market"]["prices"]
    fertilizer_price = float(prices.get("FERTILIZER", 0))
    labor = float(getattr(policy, "labor_cost_per_action", .6))
    results = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            if not tile.get("watered_today") and int(tile["planted_day"]) == day:
                continue  # New seedlings retain unconditional first-night care.
            position = x, y
            baseline = projected_crop_harvests(tile, day, last_day)
            improved = projected_crop_harvests(tile, day, last_day, True)
            harvest_today = any(d == day for d, _ in baseline + improved)
            # Simulation may assume a harvest clears the tile before tonight's
            # capped output. Reserve its action as well as today's WATER.
            deadline = tpd - hour - (0 if tile.get("watered_today") else 1) - int(harvest_today)
            routes = []
            for worker, origin in enumerate(workers):
                carried = inventories[worker] if worker < len(inventories) else {}
                if carried.get("FERTILIZER", 0):
                    cost = distance(origin, position) + 1
                else:
                    shed = nearest_shed(origin, size)
                    cost = distance(origin, shed) + 1 + distance(shed, position) + 1
                if cost <= deadline:
                    routes.append((cost, worker))
            if not routes:
                continue
            cost, worker = min(routes)
            return_actions = distance(position, nearest_shed(position, size)) + 2
            baseline_units = sum(n for d, n in baseline if d * tpd + return_actions <= terminal)
            improved_units = sum(n for d, n in improved if d * tpd + return_actions <= terminal)
            extra = improved_units - baseline_units
            gain = extra * float(prices.get(tile["crop"], 0)) - fertilizer_price - cost * labor
            if extra > 0 and gain > 0:
                results.append(FertilizerCandidate(position, tile["crop"], extra, gain,
                                                   worker, cost, deadline))
    return sorted(results, key=lambda c: (-c.net_gain / c.actions_needed, c.position))


def fertilizer_reserve(obs, config, policy, predicted_shed=None, *,
                       predicted_carried_fertilizer=None):
    """Shed units worth retaining, net of carried supply, without double count."""
    candidates = fertilizer_candidates(obs, config, policy)
    shed = obs["private"].get("shed", {}) if predicted_shed is None else predicted_shed
    carried = (sum(inv.get("FERTILIZER", 0) for inv in obs["private"].get("inventories", []))
               if predicted_carried_fertilizer is None else predicted_carried_fertilizer)
    return min(max(0, int(shed.get("FERTILIZER", 0))), max(0, len(candidates) - int(carried)))
