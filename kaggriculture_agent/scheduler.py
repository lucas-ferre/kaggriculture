"""Deadline-aware crop/livestock work, shared stock and next-turn water locks."""
from collections import Counter
from dataclasses import dataclass
from .domain import ANIMALS, CROPS, distance, move_towards, nearest_shed
from .agronomy import (CROP_CAPS, animal_care_can_pay, fertilizer_candidates,
                       nightly_animal_production)
from .optimization import assign

ANIMAL_STRUCTURES = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
ANIMAL_PRODUCTS = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}


@dataclass(frozen=True)
class Task:
    kind: str
    position: tuple
    priority: float
    crop: str = ""
    owner: int = -1
    deadline: int = 100000

    @property
    def key(self):
        return (self.kind, self.position, self.crop, self.owner)


def counts_and_positions(farm):
    counts, plants = Counter(), []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                counts[tile["crop"]] += 1
                plants.append(((x, y), tile))
    return counts, plants


class TaskScheduler:
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.targets = {}
        self.water_reservations = {}
        self.animal_sites = {}
        self.last_step = -1
        self.last_day = None
        self.last_player = None

    @staticmethod
    def _zone(worker, size):
        """Worker-index preferences remain stable through daily re-hiring."""
        h = size // 2
        offsets = ((-2, -2), (1, -2), (-2, 1), (1, 1),
                   (-4, -1), (-1, -4), (3, -1), (-1, 3))
        dx, dy = offsets[worker % len(offsets)]
        return max(0, min(size - 1, h + dx)), max(0, min(size - 1, h + dy))

    def _livestock_sites(self, farm, plan, private, inventories, size):
        occupied, candidates = Counter(), []
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("animal") in ANIMAL_STRUCTURES:
                    occupied[tile["animal"]] += 1
                elif tile is None or isinstance(tile, dict) and tile.get("kind") in {
                        "WEED", "PASTURE", "COOP"}:
                    candidates.append(((x, y), tile))
        desired = getattr(plan, "animal_targets", {}) or {}
        staged = Counter({a: max(0, int(private.get("shed", {}).get(a, 0)))
                          for a in ANIMAL_STRUCTURES})
        for inv in inventories:
            staged.update({a: max(0, int(inv.get(a, 0))) for a in ANIMAL_STRUCTURES})
        if any(desired.values()) and getattr(self.policy, "enable_livestock", False):
            # Cash affordability ramps up the herd over several days. Reserve
            # its eventual compact footprint before crops occupy that core.
            desired = {a: max(int(desired.get(a, 0)), int(getattr(self.policy, attr, 0)))
                       for a, attr in (("COW", "max_cows"), ("SHEEP", "max_sheep"),
                                       ("GOOSE", "max_geese"))}
        # A lowered target never strands livestock already purchased.
        slots = {a: max(staged[a], int(desired.get(a, 0)) - occupied[a])
                 for a in ANIMAL_STRUCTURES}
        candidates.sort(key=lambda p: (
            0 if isinstance(p[1], dict) and p[1].get("kind") in {"PASTURE", "COOP"} else 1,
            distance(p[0], nearest_shed(p[0], size)), p[0][1], p[0][0]))
        sites = {}
        for position, tile in candidates:
            animal = self.animal_sites.get(position)
            if animal and slots.get(animal, 0) > 0:
                if isinstance(tile, dict) and tile.get("kind") in {"PASTURE", "COOP"} \
                        and tile["kind"] != ANIMAL_STRUCTURES[animal]:
                    continue
                sites[position] = animal
                slots[animal] -= 1
        for position, tile in candidates:
            if position in sites:
                continue
            options = [a for a, n in slots.items() if n > 0 and (
                not isinstance(tile, dict) or tile.get("kind") == "WEED"
                or tile.get("kind") == ANIMAL_STRUCTURES[a])]
            if options:
                animal = max(options, key=lambda a: (staged[a], slots[a], a))
                sites[position] = animal
                slots[animal] -= 1
        self.animal_sites = sites
        tasks = []
        for position, animal in sites.items():
            if staged[animal] > 0:
                tasks.append(Task("INSTALL", position, 480, animal))
                staged[animal] -= 1
        return sites, tasks

    def actions(self, obs, config, plan):
        farm = obs["farms"][obs["player"]]
        private = obs["private"]
        size = len(farm["tiles"])
        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        raw_inventories = private.get("inventories", [])
        inventories = [dict(raw_inventories[i]) if i < len(raw_inventories) else {}
                       for i in range(len(positions))]
        day, hour = int(obs["day"]), int(obs["hour"])
        tpd = int(config["turnsPerDay"])
        step = int(obs.get("step", day * tpd + hour))
        player = int(obs["player"])
        if step <= self.last_step or self.last_player not in (None, player):
            self.reset()
        if self.last_day not in (None, day):
            self.targets.clear()
            self.water_reservations.clear()
        self.last_step, self.last_day, self.last_player = step, day, player
        terminal = int(config["episodeSteps"]) - 2
        remaining = max(0, terminal - step + 1)
        day_left = tpd - hour
        final_day = day == terminal // tpd
        counts, plants = counts_and_positions(farm)
        tasks, prices = [], obs["market"]["prices"]
        actions = [["PASS"] for _ in positions]
        committed, reserved_cells = set(), set()
        for worker, (position, crop, planted_day, due_step) in self.water_reservations.items():
            if worker >= len(positions) or due_step // tpd != day or step != due_step:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            if (positions[worker] == position and isinstance(tile, dict)
                    and tile.get("kind") == "PLANT" and tile.get("crop") == crop
                    and tile.get("planted_day") == planted_day and not tile.get("watered_today")):
                actions[worker] = ["WATER"]
                committed.add(worker)
                reserved_cells.add(position)
        self.water_reservations.clear()
        animals = [((x, y), tile) for y, row in enumerate(farm["tiles"])
                   for x, tile in enumerate(row) if isinstance(tile, dict)
                   and tile.get("animal") in ANIMAL_STRUCTURES]
        unfed = sum(not tile.get("fed_today") for _, tile in animals)
        sites, install_tasks = self._livestock_sites(farm, plan, private, inventories, size)
        if not final_day:
            tasks.extend(install_tasks)
        for position, tile in animals:
            animal, units = tile["animal"], int(tile.get("yield_units", 0))
            return_trip = distance(position, nearest_shed(position, size)) + 1
            if not final_day and not tile.get("fed_today"):
                urgency = 700 if tile.get("consecutive_unfed", 0) else 410
                tasks.append(Task("FEED", position, urgency + max(0, 10 - day_left) * 55,
                                  animal, deadline=day_left))
            elif units and return_trip + 1 <= remaining:
                value = units * prices.get(ANIMAL_PRODUCTS[animal], 1)
                tonight = nightly_animal_production(tile, day, bool(tile.get("fed_today")))
                overflow = max(0, units + tonight - ANIMALS[animal]["cap"])
                priority = 320 + min(250, value * .25) + (600 if final_day else 0)
                deadline = max(0, remaining - return_trip)
                if overflow:
                    # Feeding precedes this task; tonight uses yesterday's
                    # banked care, not today's CARE action.
                    priority += 350 + min(400, overflow * prices.get(ANIMAL_PRODUCTS[animal], 1))
                    deadline = min(deadline, day_left)
                tasks.append(Task("HARVEST", position, priority, animal, deadline=deadline))
            elif (not final_day and not tile.get("cared_today")
                  and animal_care_can_pay(tile, day, terminal // tpd)):
                tasks.append(Task("CARE", position, 260 + max(0, 8 - day_left) * 20,
                                  animal, deadline=day_left))
            elif tile.get("fertilizer_available") and return_trip + 1 <= remaining:
                tasks.append(Task("COLLECT_FERTILIZER", position,
                                  170 + (600 if final_day else 0), animal,
                                  deadline=min(day_left, max(0, remaining - return_trip))))
        fertilizer_stock = int(private["shed"].get("FERTILIZER", 0)) + sum(
            inv.get("FERTILIZER", 0) for inv in inventories)
        def reachable_application(candidate):
            for worker, position in enumerate(positions):
                if worker in committed:
                    continue
                if inventories[worker].get("FERTILIZER", 0):
                    cost = distance(position, candidate.position) + 1
                elif private["shed"].get("FERTILIZER", 0):
                    shed = nearest_shed(position, size)
                    cost = distance(position, shed) + 1 + distance(shed, candidate.position) + 1
                else:
                    continue
                if cost <= candidate.deadline:
                    return True
            return False
        candidates = fertilizer_candidates(obs, config, self.policy) if fertilizer_stock else []
        applications = [candidate for candidate in candidates
                        if candidate.position not in reserved_cells and reachable_application(candidate)][:fertilizer_stock]
        applications = {candidate.position: candidate for candidate in applications}
        for position, tile in plants:
            if position in reserved_cells:
                continue
            if position in applications:
                candidate = applications[position]
                tasks.append(Task("FERTILIZE", position, 230 + min(250, candidate.net_gain),
                                  tile["crop"], deadline=candidate.deadline))
                continue
            crop = tile["crop"]
            cd = CROPS[crop]
            age, units = day - tile["planted_day"], int(tile.get("yield_units", 0))
            mls = int(tile.get("max_lifespan_step", -1))
            decays_soon = mls >= 0 and mls - step < day_left
            ripe = age >= cd.first and units > 0
            return_trip = distance(position, nearest_shed(position, size)) + 1
            can_cash = return_trip + 1 <= remaining
            needs_peak_water = (not cd.ongoing and not tile["watered_today"]
                                and age <= cd.last and units < CROP_CAPS[crop])
            harvest = ripe and can_cash and (
                final_day or decays_soon or
                (not cd.ongoing and (units >= CROP_CAPS[crop] or (age >= cd.peak and not needs_peak_water))) or
                (cd.ongoing and (units >= 2 or age >= cd.last)))
            if harvest:
                value = units * prices.get(crop, 1)
                priority = 130 + min(150, value * .25)
                if decays_soon or final_day:
                    priority += 500
                tasks.append(Task("HARVEST", position, priority, crop,
                                  deadline=max(0, remaining - return_trip)))
                continue
            if cd.ongoing and age >= cd.last and units == 0:
                if not final_day:
                    tasks.append(Task("DIG", position, 45, crop, deadline=day_left))
                continue
            if not tile["watered_today"] and not final_day:
                urgent = tile.get("consecutive_unwatered", 0) >= 1
                priority = (550 if urgent else 150) + max(0, 12 - day_left) * 35
                if needs_peak_water and age >= cd.peak:
                    priority += 70
                if tile["planted_day"] == day:
                    priority += 1000  # Repair a reservation after a cold start.
                tasks.append(Task("WATER", position, priority, crop, deadline=day_left))
        targets = {c: max(0, int(n)) for c, n in plan.targets.items() if c in CROPS}
        deficits = {c: max(0, n - counts[c]) for c, n in targets.items()}
        free_budget = max(0, int(self.policy.max_plots) - len(plants))
        seeds = dict(private.get("seeds", {}))
        available = []
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if (x, y) not in sites and (tile is None or isinstance(tile, dict)
                                                       and tile.get("kind") == "WEED"):
                    available.append(((x, y), tile))
        available.sort(key=lambda p: (distance(p[0], nearest_shed(p[0], size)), p[0][1], p[0][0]))
        previous_planting = {key[1]: key[2] for key in self.targets.values()
                             if key[0] in {"PLANT", "DIG"}}
        if not final_day:
            for position, tile in available:
                if free_budget <= 0:
                    break
                options = [c for c in deficits if deficits[c] > 0 and seeds.get(c, 0) > 0]
                if not options:
                    break
                crop = previous_planting.get(position)
                if crop not in options:
                    crop = max(options, key=lambda c: (
                        deficits[c] / max(1, targets[c]), CROPS[c].ongoing, -CROPS[c].seed_cost))
                first_step = (day + CROPS[crop].first) * tpd
                if first_step + distance(position, nearest_shed(position, size)) + 2 > terminal:
                    deficits[crop] = 0
                    continue
                if tile is not None:
                    tasks.append(Task("DIG", position, 35, crop, deadline=max(0, day_left - 2)))
                else:
                    # PLANT must leave a complete next turn for the same worker.
                    tasks.append(Task("PLANT", position, 85, crop, deadline=max(0, day_left - 1)))
                deficits[crop] -= 1
                seeds[crop] -= 1
                free_budget -= 1
        capacity = int(config["shedCapacity"])
        shed_total = sum(private["shed"].values())
        carried_total = sum(sum(inv.values()) for inv in inventories)
        for worker, (position, inv) in enumerate(zip(positions, inventories)):
            depositable = {item: n for item, n in inv.items() if item not in ANIMAL_STRUCTURES
                           and (item != "WHEAT" or not unfed or final_day)
                           and (item != "FERTILIZER" or not applications)}
            load = sum(depositable.values())
            if not load:
                continue
            target = nearest_shed(position, size)
            value = sum(prices.get(item, 0) * quantity for item, quantity in depositable.items())
            pressure = shed_total + carried_total > capacity * .7
            priority = 25 + min(100, value * .1)
            if position == target or load >= 8 or pressure:
                priority += 180
            if final_day:
                priority += 700
            deadline = remaining
            if not final_day and 1 <= day_left - distance(position, target) <= 3:
                # Only the carrier can deliver goods: there is no unit-to-unit
                # transfer action. Arrive while a sale can free overnight room.
                priority += min(400, value * .5)
                deadline = min(deadline, day_left)
            if shed_total + carried_total > capacity and day_left <= 6:
                priority += 400
            tasks.append(Task("DEPOSIT", target, priority, owner=worker, deadline=deadline))
        zones = [self._zone(w, size) for w in range(len(positions))]
        zone_minima = {t.position: min(distance(z, t.position) for z in zones) for t in tasks}
        costs = []
        for worker, position in enumerate(positions):
            row = []
            for task in tasks:
                dist = distance(position, task.position)
                feasible = worker not in committed and task.owner in (-1, worker)
                inv = inventories[worker]
                if task.kind in {"FEED", "INSTALL", "FERTILIZE"}:
                    item = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}.get(task.kind, task.crop)
                    if inv.get(item, 0) <= 0:
                        feasible = feasible and private["shed"].get(item, 0) > 0
                        shed = nearest_shed(position, size)
                        dist = distance(position, shed) + 1 + distance(shed, task.position)
                feasible = feasible and dist + 1 <= task.deadline
                if task.kind in {"HARVEST", "COLLECT_FERTILIZER"}:
                    feasible = feasible and dist + 2 + distance(
                        task.position, nearest_shed(task.position, size)) <= remaining
                if not feasible:
                    row.append(float("inf"))
                    continue
                extra_zone = distance(zones[worker], task.position) - zone_minima[task.position]
                utility = task.priority / (1 + dist + .35 * extra_zone)
                if self.targets.get(worker) == task.key:
                    utility *= 1.55
                if position == task.position:
                    utility *= 1.25
                if task.kind == "INSTALL" and inv.get(task.crop, 0):
                    utility *= 3
                row.append(-utility)
            row.extend([0.0] * len(positions))
            costs.append(row)
        assignment = assign(costs, method=self.policy.scheduler)
        predicted_shed = dict(private["shed"])
        reserved_seeds = dict(private.get("seeds", {}))
        new_targets = {}
        feed_held = sum(inv.get("WHEAT", 0) for inv in inventories)
        # The interpreter processes workers in index order before market orders.
        for worker, task_index in sorted(assignment):
            if worker in committed or task_index >= len(tasks) or costs[worker][task_index] == float("inf"):
                continue
            task = tasks[task_index]
            new_targets[worker] = task.key
            position, inv = positions[worker], inventories[worker]
            if task.kind in {"FEED", "INSTALL", "FERTILIZE"}:
                item = {"FEED": "WHEAT", "FERTILIZE": "FERTILIZER"}.get(task.kind, task.crop)
                if not inv.get(item, 0):
                    shed = nearest_shed(position, size)
                    if position != shed:
                        actions[worker] = move_towards(position, shed)
                    else:
                        available_stock = max(0, predicted_shed.get(item, 0))
                        amount = min(available_stock, max(1, min(3, unfed - feed_held))) \
                            if item == "WHEAT" else min(1, available_stock)
                        if amount:
                            actions[worker] = ["PICKUP", item, amount]
                            predicted_shed[item] = available_stock - amount
                            if item == "WHEAT":
                                feed_held += amount
                    continue
            if position != task.position:
                actions[worker] = move_towards(position, task.position)
                continue
            if task.kind == "PLANT":
                if reserved_seeds.get(task.crop, 0) > 0 and day_left >= 2 and remaining >= 2:
                    actions[worker] = ["PLANT", task.crop]
                    reserved_seeds[task.crop] -= 1
                    self.water_reservations[worker] = (position, task.crop, day, step + 1)
            elif task.kind == "INSTALL":
                tile = farm["tiles"][position[1]][position[0]]
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    actions[worker] = ["DIG"]
                elif tile is None:
                    actions[worker] = ["BUILD_" + ANIMAL_STRUCTURES[task.crop]]
                elif isinstance(tile, dict) and tile.get("kind") == ANIMAL_STRUCTURES[task.crop] \
                        and not tile.get("animal"):
                    actions[worker] = ["PLACE", task.crop]
            elif task.kind == "DEPOSIT":
                room = max(0, capacity - sum(predicted_shed.values()))
                safe = {item: n for item, n in inv.items() if item not in ANIMAL_STRUCTURES
                        and (item != "WHEAT" or not unfed or final_day)
                        and (item != "FERTILIZER" or not applications)}
                if safe and room >= sum(inv.values()) and len(safe) == len(inv):
                    actions[worker] = ["DROP"]
                    for item, quantity in inv.items():
                        predicted_shed[item] = predicted_shed.get(item, 0) + quantity
                elif safe and room > 0:
                    item = max(safe, key=lambda k: prices.get(k, 0))
                    quantity = min(safe[item], room)
                    actions[worker] = ["PLACE", item, quantity]
                    predicted_shed[item] = predicted_shed.get(item, 0) + quantity
            else:
                actions[worker] = [task.kind]
                if task.kind == "FERTILIZE":
                    tile = farm["tiles"][position[1]][position[0]]
                    if not tile.get("watered_today"):
                        self.water_reservations[worker] = (
                            position, task.crop, int(tile["planted_day"]), step + 1)
        self.targets = new_targets
        return actions, predicted_shed, {"tasks": len(tasks), "assignment": len(assignment),
                                         "water_reservations": len(self.water_reservations),
                                         "animals": len(animals), "unfed": unfed,
                                         "animal_sites": len(sites),
                                         "fertilizer_candidates": len(applications)}
