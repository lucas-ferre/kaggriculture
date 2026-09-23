"""Composition root: bounded forecasting, crop planning and task execution."""
from collections import Counter
from dataclasses import asdict, replace
from .domain import ANIMALS, CROPS, PRODUCTS, fib_cost, game_configuration
from .economy import animal_counts, expansion_plan, livestock_plan, purchase_cost, prioritize_market_orders
from .agronomy import fertilizer_reserve
from .intelligence import ContextualBandit, OpponentModel
from .market import MarketModel, sale_revenue, sale_quantity_at_reserve
from .planner import CropPlanner, PolicyConfig, minimum_cycle_day
from .quantile import load_approved_model
from .scheduler import TaskScheduler, counts_and_positions


class FarmAgent:
    """Create one instance per player/episode for local evaluation.

    Only supplied observations and configuration are used. Module-memory is an
    optimization: a newly created instance remains a valid cold-start policy.
    """
    def __init__(self, policy=None):
        self.policy = policy or PolicyConfig()
        self.planner = CropPlanner(self.policy)
        self.scheduler = TaskScheduler(self.policy)
        self.reset()

    def reset(self):
        self.market = None
        self.plan = None
        self.last_step = -1
        self.last_player = None
        self.previous_orders = []
        self.stock_since = {}
        self.last_debug = {}
        self.economy_debug = {}
        self.expansion_debug = {}
        self.last_land_count = None
        self.opponent = OpponentModel()
        self.bandit = ContextualBandit()
        self.scheduler.reset()

    def __call__(self, observation, configuration=None):
        obs = observation
        config = game_configuration(configuration)
        step = int(obs.get("step", int(obs["day"]) * config["turnsPerDay"] + int(obs["hour"])))
        player = int(obs["player"])
        if step <= self.last_step or (self.last_player is not None and player != self.last_player):
            self.reset()
        if self.market is None:
            market_config = dict(config, forecastingMethod=self.policy.market_method)
            if not getattr(self.policy, "enable_mrlm", True):
                market_config["mrlmModel"] = None
            calibrated = load_approved_model()
            if calibrated is not None:
                market_config["quantileModel"] = calibrated
            self.market = MarketModel(market_config)
        self.market.update(obs, own_previous_orders=self.previous_orders)
        if self.policy.enable_opponent_model:
            self.opponent.update(obs, self.market)
        land_count = len(obs["farms"][player].get("unlocked_quadrants", ["NW"]))
        if self.plan is None or obs["hour"] == 0 or step % 6 == 0 or land_count != self.last_land_count:
            animal_targets, self.economy_debug = livestock_plan(obs, config, self.policy, self.market)
            live, owned = animal_counts(obs)
            farm = obs["farms"][player]
            empty_structures = sum(isinstance(tile, dict) and tile.get("kind") in {"COOP", "PASTURE"}
                                   and "animal" not in tile for row in farm["tiles"] for tile in row)
            pending = max(0, sum(animal_targets.values()) - sum(live.values()))
            core = sum(animal_targets.values())
            if any(animal_targets.values()) and self.policy.enable_livestock:
                core = max(core, self.policy.max_cows + self.policy.max_sheep + self.policy.max_geese)
            reserved_land = max(0, core - sum(live.values()) - empty_structures)
            planning_config = dict(config, reservedAnimalPlots=reserved_land,
                                   reservedAnimalCash=self.economy_debug["reserved_cash"],
                                   reservedAnimalActions=6 * pending)
            if self.policy.enable_contextual_bandit and self.policy.strategy == "adaptive":
                strategy = self.bandit.choose(obs, self.opponent.profile)
                if int(config["episodeSteps"]) - 2 - step <= 4 * int(config["turnsPerDay"]):
                    strategy = "liquidator"
                self.planner.config = replace(self.policy, strategy=strategy)
            self.plan = self.planner.plan(obs, planning_config, self.market)
            self.plan.animal_targets = animal_targets
            self.last_land_count = land_count
        actions, predicted_shed, schedule_debug = self.scheduler.actions(obs, config, self.plan)
        cargo_after = self._cargo_after_actions(obs, actions, predicted_shed)
        orders, sale_debug = self._market_orders(obs, config, predicted_shed,
                                               cargo_after.get("WHEAT", 0), cargo_after)
        self.previous_orders = [list(order) for order in orders]
        self.last_step, self.last_player = step, player
        self.last_debug = {"step": step, "plan": asdict(self.plan),
                           "sales": sale_debug, "livestock": self.economy_debug,
                           "expansion": self.expansion_debug,
                           "opponent": self.opponent.profile,
                           "bandit_updates": dict(self.bandit.counts), **schedule_debug}
        return {"farmer": actions[0], "hands": actions[1:], "market": orders}

    @staticmethod
    def _carried_feed_after_actions(obs, actions, predicted_shed):
        return FarmAgent._cargo_after_actions(obs, actions, predicted_shed).get("WHEAT", 0)

    @staticmethod
    def _cargo_after_actions(obs, actions, predicted_shed):
        """Conserve wheat across unit actions before the market stage.

        The scheduler's shed ledger already contains pickups and deposits.
        Subtract it from total wheat, then apply harvest/feed transformations.
        This avoids counting a deposited lot again as worker inventory.
        """
        private = obs["private"]
        total = Counter(private["shed"])
        for inventory in private.get("inventories", []):
            total.update(inventory)
        farm = obs["farms"][obs["player"]]
        positions = [farm["farmer"], *farm["hands"]]
        for action, (x, y) in zip(actions, positions):
            if action[0] == "FEED":
                total["WHEAT"] -= 1
            elif action[0] == "FERTILIZE":
                total["FERTILIZER"] -= 1
            elif action[0] == "COLLECT_FERTILIZER":
                total["FERTILIZER"] += 1
            elif action[0] == "HARVEST":
                tile = farm["tiles"][y][x]
                if isinstance(tile, dict):
                    product = ANIMALS[tile["animal"]]["product"] if tile.get("animal") in ANIMALS else tile.get("crop")
                    if product in PRODUCTS:
                        total[product] += max(0, int(tile.get("yield_units", 0)))
            elif action[0] == "PLACE" and action[1] in ANIMALS:
                tile = farm["tiles"][y][x]
                if isinstance(tile, dict) and tile.get("kind") in {"PASTURE", "COOP"} and not tile.get("animal"):
                    total[action[1]] -= 1
        return {item: max(0, quantity - predicted_shed.get(item, 0)) for item, quantity in total.items()}

    def _market_orders(self, obs, config, predicted_shed, predicted_carried_feed=None, predicted_cargo=None):
        farm = obs["farms"][obs["player"]]
        private = obs["private"]
        tpd = int(config["turnsPerDay"])
        step = int(obs.get("step", obs["day"] * tpd + obs["hour"]))
        remaining = int(config["episodeSteps"]) - 2 - step + 1
        capacity = int(config["shedCapacity"])
        # Build a funded candidate queue before choosing the executable slots.
        limit = max(20, int(config["maxMarketOrdersPerTurn"]))
        orders, sale_debug = [], {}
        # Deliberately do not spend anticipated sales: concurrent opponent orders
        # can move the realised price before our own purchase executes.
        budget = max(0.0, farm["money"] - self.policy.cash_reserve)
        params = obs["market"].get("params") or config.get("marketParams")
        live, owned = animal_counts(obs)
        # Existing animals still need care after acquisitions have stopped.
        animals = sum(owned.values())
        carried_feed = (sum(inv.get("WHEAT", 0) for inv in private.get("inventories", []))
                        if predicted_carried_feed is None else predicted_carried_feed)
        desired_feed = animals + max(1, animals // 2) if animals else 0
        if self.policy.tactical_wheat_reserve and animals and self.opponent.profile.get("visible_animals"):
            # A bounded own-feed hedge; market wheat is unlimited, so this is
            # not a mechanism that can guarantee starvation of the rival.
            desired_feed = min(capacity // 3, max(desired_feed, 2 * animals))
        feed_reserve = max(0, desired_feed - carried_feed) if remaining > tpd - obs["hour"] else 0
        cargo = (Counter(predicted_cargo) if predicted_cargo is not None else
                 sum((Counter(inv) for inv in private.get("inventories", [])), Counter()))
        fertilizer_hold = fertilizer_reserve(obs, config, self.policy, predicted_shed,
                                             predicted_carried_fertilizer=cargo.get("FERTILIZER", 0))
        midnight_pressure = (obs["hour"] >= max(0, tpd - 2) and
                             sum(predicted_shed.values()) + sum(cargo.values()) > capacity)
        returning = sum(cargo.values()) if obs["hour"] >= max(0, tpd - 2) else 0
        available_shed = dict(predicted_shed)
        for product in PRODUCTS:
            quantity = int(predicted_shed.get(product, 0))
            if product == "WHEAT":
                quantity = max(0, quantity - feed_reserve)
            elif product == "FERTILIZER" and not midnight_pressure:
                quantity = max(0, quantity - fertilizer_hold)
            if quantity <= 0:
                self.stock_since.pop(product, None)
                continue
            self.stock_since.setdefault(product, step)
            current = sale_revenue(product, obs["market"]["inventory"][product], quantity, params)
            sell = True
            lead = (self.policy.strategy == "adaptive" and self.market is not None and
                    self.market.should_lead_sale(product, quantity))
            lead = lead or bool(self.policy.enable_opponent_model and self.opponent.supply_risk(product) >= 3
                               and product in {"MELON", "MILK", "WOOL", "STRAWBERRY"})
            if lead:
                sale_debug[product] = {"decision": "sell", "reason": "one_turn_supply_risk"}
            if (not lead and not midnight_pressure and self.policy.use_forecast and remaining > 2 * tpd
                    and sum(predicted_shed.values()) < capacity * 0.55
                    and farm["money"] > self.policy.cash_reserve * 2
                    and step - self.stock_since[product] < tpd):
                prediction = self.market.forecast(product, tpd, quantity)
                risk_adjusted = prediction.expected_revenue - self.policy.risk_aversion * max(
                    0.0, current - quantity * prediction.p10)
                sell = not (prediction.prob_up >= 0.65 and risk_adjusted > current * 1.04)
                if not sell:
                    reserve_price = self.market.price_quantile(product, tpd, self.policy.risk_quantile)
                    immediate = sale_quantity_at_reserve(product, obs["market"]["inventory"][product],
                                                         quantity, reserve_price, params)
                    if immediate:
                        quantity, sell = immediate, True
                sale_debug[product] = {"decision": "sell" if sell else "hold",
                                       "prob_up": prediction.prob_up,
                                       "expected_revenue": prediction.expected_revenue,
                                       "confidence": prediction.confidence}
            if midnight_pressure:
                sale_debug[product] = {"decision": "sell", "reason": "midnight_shed_capacity"}
            if sell and len(orders) < limit:
                orders.append(["SELL", product, quantity])
                available_shed[product] = available_shed.get(product, 0) - quantity
                self.stock_since.pop(product, None)

        counts, plants = counts_and_positions(farm)
        # Feeding has priority over discretionary investment. Purchases account
        # for post-buy quotes and shed space after this turn's planned deposits.
        if animals and feed_reserve and len(orders) < limit:
            feed_stock = available_shed.get("WHEAT", 0) + carried_feed
            need = max(0, desired_feed - feed_stock) if feed_stock < animals else 0
            need = min(need, max(0, capacity - sum(available_shed.values()) - returning))
            inv = obs["market"]["inventory"]["WHEAT"]
            # The cash reserve is a discretionary-investment floor, not a
            # reason to let an existing animal die. Emergency feed may use it.
            feed_budget = max(0.0, float(farm["money"]))
            while need and purchase_cost("WHEAT", inv, need, params) > feed_budget:
                need -= 1
            if need:
                orders.append(["BUY_PRODUCT", "WHEAT", need])
                budget = max(0.0, budget - purchase_cost("WHEAT", inv, need, params))
                available_shed["WHEAT"] = available_shed.get("WHEAT", 0) + need

        expected_plots = max(len(plants), sum(self.plan.targets.values()))
        planned_animals = sum(self.plan.animal_targets.values())
        work_left = expected_plots > 0 or animals or planned_animals or any(sum(inv.values()) for inv in private.get("inventories", []))
        final_day = int(obs["day"]) == (int(config["episodeSteps"]) - 2) // tpd
        final_jobs = sum(tile.get("yield_units", 0) > 0 and int(obs["day"]) - tile["planted_day"] >= CROPS[tile["crop"]].first
                         for _, tile in plants)
        final_jobs += sum(isinstance(tile, dict) and tile.get("animal") in ANIMALS and
                          (tile.get("yield_units", 0) > 0 or bool(tile.get("fertilizer_available", False)))
                          for row in farm["tiles"] for tile in row)
        if final_day:
            work_left = bool(final_jobs or sum(cargo.values()))
        if work_left and obs["hour"] < min(6, tpd // 2):
            unlocked_quads = len(farm.get("unlocked_quadrants", ["NW"]))
            effective_target = min(self.policy.target_workers, 8 if unlocked_quads <= 2 else (9 if unlocked_quads == 3 else 10))
            desired = min(effective_target, max(1, (expected_plots + 2 * max(animals, planned_animals) + 2) // 3))
            if self.plan.workforce_target:
                desired = min(effective_target, self.plan.workforce_target)
            if final_day:
                desired = min(desired, max(1, final_jobs))
            hires = int(farm.get("hires_today", len(farm["hands"])))
            present = 1 + len(farm["hands"])
            while present < desired and len(orders) < limit:
                cost = fib_cost(hires, int(config["farmHandCostMult"]))
                if cost > budget or (cost >= 210 and budget < cost * 2.0):
                    break
                orders.append(["HIRE"])
                budget -= cost
                hires += 1
                present += 1

        if self.policy.enable_livestock and remaining > 8 * tpd:
            for animal, target in self.plan.animal_targets.items():
                if animal not in ANIMALS or len(orders) >= limit:
                    continue
                # Protect grain for the whole herd for one additional day.
                feed_buffer = purchase_cost("WHEAT", obs["market"]["inventory"]["WHEAT"], animals + 1, params)
                n = min(max(0, target - owned[animal]), max(0, int((budget - feed_buffer) // ANIMALS[animal]["cost"])),
                        max(0, capacity - sum(available_shed.values()) - returning))
                if n:
                    orders.append(["BUY_ANIMAL", animal, n])
                    budget -= n * ANIMALS[animal]["cost"]
                    available_shed[animal] = available_shed.get(animal, 0) + n

        targets = self.plan.targets.items()
        for crop, target in targets:
            if not crop or remaining <= tpd or len(orders) >= limit:
                continue
            # Targets may preserve old plants or be up to five turns old. They
            # are not permission to buy seeds for a cycle that cannot finish.
            margin = max(self.policy.logistics_margin, int(config["boardSize"]) + 2)
            available_age = (int(config["episodeSteps"]) - 3 - margin) // tpd - int(obs["day"])
            required_age = minimum_cycle_day(crop, self.policy)
            if available_age < required_age:
                continue
            quantity = max(0, int(target) - counts[crop] - private["seeds"].get(crop, 0))
            quantity = min(quantity, max(0, int(budget // CROPS[crop].seed_cost)))
            if quantity:
                orders.append(["BUY_SEED", crop, quantity])
                budget -= quantity * CROPS[crop].seed_cost

        # All animal and seed commitments must be funded before pricing land.
        # Otherwise the opening-day land quote protects a cash buffer which
        # later acquisitions immediately consume, leaving no grain for care.
        buy_land, self.expansion_debug = expansion_plan(obs, config, self.policy, self.plan, budget)
        if buy_land and len(orders) < limit:
            orders.append(["BUY_LAND"])
            budget -= self.expansion_debug["cost"]

        unfed_urgent = any(isinstance(tile, dict) and tile.get("animal") in ANIMALS and
                          not tile.get("fed_today") and tile.get("consecutive_unfed", 0) >= 1
                          for row in farm["tiles"] for tile in row)
        orders = prioritize_market_orders(orders, obs, config,
                                          self.opponent if self.policy.enable_opponent_model else None,
                                          feed_urgent=unfed_urgent and carried_feed == 0,
                                          current_shed=predicted_shed)
        return orders, sale_debug
