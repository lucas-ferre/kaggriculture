"""Risk-aware crop portfolios using public market forecasts.

Crop data and yield rules follow vendor/kaggriculture/kaggriculture.py.
Labor costs, area ratios, and yield execution are policy assumptions to measure
in simulations, not extra game rules. Linear tranches approximate returns;
bounded integer optimization does not guarantee optimal gameplay.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .optimization import branch_bound, optimize_allocation, simplex
from .agronomy import projected_crop_harvests
from .domain import fib_cost
from .market import sale_revenue


@dataclass(frozen=True)
class PolicyConfig:
    fast_share: float = 0.6
    risk_aversion: float = 0.35
    target_workers: int = 10
    max_plots: int = 40
    cash_reserve: float = 300.0
    scheduler: str = "greedy"
    optimizer: str = "branch_bound"
    integer_node_limit: int = 64
    market_method: str = "quadrature"
    enable_fertilizer: bool = True
    enable_contextual_bandit: bool = False
    enable_opponent_model: bool = True
    # Runtime uses coefficients only when their validation artifact is approved.
    enable_mrlm: bool = True
    tactical_wheat_reserve: bool = False
    strategy: str = "adaptive"
    risk_quantile: float = 0.25
    enable_livestock: bool = True
    enable_expansion: bool = True
    max_cows: int = 8
    max_sheep: int = 6
    max_geese: int = 2
    max_expansions: int = 3
    use_forecast: bool = True
    dual_cycle: bool = True
    # New long-cycle investments default to crops with repeated production.
    # Existing plants outside this set remain protected until harvested.
    long_mode: str = "recurrent"
    # Opportunity cost and travel allowance are tunable approximations.
    labor_cost_per_action: float = 0.6
    labor_actions_per_plot_day: float = 2.2
    labor_utilization: float = 0.65
    logistics_margin: int = 12
    allocation_slack: float = 0.15
    # A bounded policy preference, not an estimated financial return. Setting
    # this to zero removes the recurrent-crop preference for ablation runs.
    ongoing_preference: float = 0.10
    # Partial recurrent cycles can pay back before the terminal logistics
    # window; four restores the previous full-cycle investment restriction.
    min_recurrent_pulses: int = 2
    beam_width: int = 16

    def __post_init__(self):
        if not 0 <= self.fast_share <= 1 or not 0 <= self.risk_aversion <= 1:
            raise ValueError("fast_share and risk_aversion must lie in [0, 1]")
        if self.target_workers < 1 or self.max_plots < 0 or self.cash_reserve < 0:
            raise ValueError("worker, plot and reserve settings must be nonnegative")
        if self.scheduler not in {"greedy", "hungarian"}:
            raise ValueError("scheduler must be greedy or hungarian")
        if self.optimizer not in {"enumerate", "beam", "simplex", "branch_bound"}:
            raise ValueError("optimizer must be enumerate, beam, simplex or branch_bound")
        if not isinstance(self.integer_node_limit, int) or self.integer_node_limit < 1:
            raise ValueError("integer_node_limit must be a positive integer")
        if self.market_method not in {"quadrature", "scenarios"}:
            raise ValueError("market_method must be quadrature or scenarios")
        if self.strategy not in {"adaptive", "dual", "melon", "cashflow", "contrarian", "liquidator"}:
            raise ValueError("unknown strategy")
        if not 0 <= self.risk_quantile <= 0.5:
            raise ValueError("risk_quantile must lie in [0, .5]")
        if min(self.max_cows, self.max_sheep, self.max_geese, self.max_expansions) < 0:
            raise ValueError("animal and expansion limits must be nonnegative")
        if self.long_mode not in {"recurrent", "any"}:
            raise ValueError("long_mode must be recurrent or any")
        if not isinstance(self.min_recurrent_pulses, int) or not 2 <= self.min_recurrent_pulses <= 4:
            raise ValueError("min_recurrent_pulses must be an integer in [2, 4]")
        if (self.labor_actions_per_plot_day <= 0 or self.labor_cost_per_action < 0
                or not 0 < self.labor_utilization <= 1 or self.logistics_margin < 0
                or not 0 <= self.allocation_slack <= 1 or self.ongoing_preference < 0
                or self.beam_width < 1):
            raise ValueError("invalid labor, logistics, allocation or beam setting")


@dataclass(frozen=True)
class CropSpec:
    seed: int
    first_day: int
    max_day: int
    interval: int
    cap: int
    ongoing: bool


CROPS = {
    "WHEAT": CropSpec(10, 2, 4, 0, 6, False),
    "CARROT": CropSpec(20, 2, 3, 0, 4, False),
    "TOMATO": CropSpec(50, 8, 8, 1, 4, True),
    "STRAWBERRY": CropSpec(100, 10, 10, 2, 4, True),
    "MELON": CropSpec(80, 10, 12, 0, 6, False),
}
SHORT_CROPS = ("WHEAT", "CARROT")
RECURRENT_CROPS = ("TOMATO", "STRAWBERRY")
LONG_CROPS = ("TOMATO", "STRAWBERRY", "MELON")


@dataclass
class CropPlan:
    short_crop: str | None
    long_crop: str | None
    short_target: int
    long_target: int
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    allocation: dict[str, int] = field(default_factory=dict)
    animal_targets: dict[str, int] = field(default_factory=dict)
    specialist: str = ""
    workforce_target: int = 0
    optimizer_diagnostics: dict = field(default_factory=dict)

    @property
    def targets(self) -> dict[str, int]:
        """Total desired plots of selected crops, including existing plants."""
        if self.allocation:
            return {crop: count for crop, count in self.allocation.items() if count > 0}
        return {crop: target for crop, target in
                ((self.short_crop, self.short_target), (self.long_crop, self.long_target))
                if crop is not None and target > 0}


def yield_without_fertilizer(crop: str, age_days: int) -> int:
    """Cumulative yield when watered daily and harvested before decay."""
    spec = CROPS[crop]
    if age_days < spec.first_day:
        return 0
    if spec.ongoing:
        return min(spec.cap, 1 + (age_days - spec.first_day) // spec.interval)
    window = (spec.max_day + 1) // 2
    return min(spec.cap, 1 + max(0, min(age_days, spec.max_day) - window + 1))


def full_cycle_day(crop: str) -> int:
    """Earliest age for all unfertilized output; ongoing output is finite."""
    spec = CROPS[crop]
    if spec.ongoing:
        return spec.first_day + spec.interval * (spec.cap - 1)
    final_yield = yield_without_fertilizer(crop, spec.max_day)
    return next(age for age in range(spec.first_day, spec.max_day + 1)
                if yield_without_fertilizer(crop, age) == final_yield)


def minimum_cycle_day(crop: str, policy: PolicyConfig) -> int:
    """Earliest investable cycle, shared by planning and actual seed orders.

    Two recurring production events are a conservative minimum; the planner
    still requires forecast revenue to pay seed and labor costs. Melons and
    the legacy dual strategy retain the full long-cycle requirement.
    """
    spec = CROPS[crop]
    if spec.ongoing and policy.strategy != "dual":
        return spec.first_day + spec.interval * (policy.min_recurrent_pulses - 1)
    return full_cycle_day(crop) if crop in LONG_CROPS else spec.first_day


def _field(value, name: str, default=0.0):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class CropPlanner:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def plan(self, obs: dict, game_config: dict, market_model) -> CropPlan:
        if self.config.strategy != "dual":
            return self._portfolio(obs, game_config, market_model)
        return self._dual(obs, game_config, market_model)

    def _dual(self, obs: dict, game_config: dict, market_model) -> CropPlan:
        """Plan new investments while keeping planted crops in place.

        Future harvests are priced as one aggregate lot at their production-
        weighted mean date. This intentionally pessimistic self-impact proxy
        avoids valuing every parcel at the same pre-sale quote. It is a planning
        approximation, not a simulated cash-flow or a calibrated probability.
        """
        config = self.config
        method = config.optimizer if config.optimizer in {"enumerate", "beam"} else "enumerate"
        farms = obs.get("farms", [])
        if not farms:
            return CropPlan(None, None, 0, 0, reason="Sem estado de fazenda.")
        farm = farms[int(obs.get("player", 0))]
        tiles = [tile for row in farm.get("tiles", []) for tile in row]
        existing = Counter(tile["crop"] for tile in tiles
                           if isinstance(tile, dict) and tile.get("kind") == "PLANT"
                           and tile.get("crop") in CROPS)
        # Animal structures and locked land cannot host new crops. Weeds can be
        # cleared, but count as available only with an extra execution task.
        cultivable = sum(tile is None or isinstance(tile, dict)
                         and tile.get("kind") in {"PLANT", "WEED"} for tile in tiles)
        tpd = max(1, int(game_config.get("turnsPerDay", 24)))
        step = int(obs.get("step", int(obs.get("day", 0)) * tpd + int(obs.get("hour", 0))))
        last_action = int(game_config.get("episodeSteps", 720)) - 2
        margin = max(config.logistics_margin, int(game_config.get("boardSize", 10)) + 2)
        harvest_deadline = last_action - margin
        day = step // tpd
        # Daily hands are rehired by the execution policy. Limit plots by the
        # configured sustainable workforce, reserving capacity for logistics.
        labor_cap = int(config.target_workers * tpd * config.labor_utilization
                        / config.labor_actions_per_plot_day)
        capacity = min(config.max_plots, cultivable, max(0, labor_cap))
        cash = max(0.0, float(farm.get("money", 0)) - config.cash_reserve)
        seeds = obs.get("private", {}).get("seeds", {})
        prices = obs.get("market", {}).get("prices", {})
        profiles = {}
        for crop, spec in CROPS.items():
            # Yield starts on the relevant day boundary; allow two actions to
            # plant/water initially and one to harvest at the final boundary.
            age_available = (harvest_deadline - 1) // tpd - day
            full_age = full_cycle_day(crop)
            viable = age_available >= spec.first_day
            if crop in LONG_CROPS:
                viable = age_available >= full_age
            age = min(full_age, age_available)
            if not viable or age < 0:
                profiles[crop] = (0, 0, 0, 0.0)
                continue
            # Only one cycle is promised for existing area; potential replants
            # in free area are modeled on complete day-separated cycles.
            harvests, start_day = [], day
            while True:
                remaining_age = (harvest_deadline - 1) // tpd - start_day
                chosen_age = min(full_age, remaining_age)
                if chosen_age < spec.first_day or (crop in LONG_CROPS and chosen_age < full_age):
                    break
                quantity = yield_without_fertilizer(crop, chosen_age)
                harvest_step = (start_day + chosen_age) * tpd + 1
                harvests.append((max(1, harvest_step - step), quantity, chosen_age))
                start_day += chosen_age + 1
            quantity = sum(q for _, q, _ in harvests)
            horizon = int(sum(h * q for h, q, _ in harvests) / max(1, quantity))
            actions = sum((age + 1) * config.labor_actions_per_plot_day + 2
                          for _, _, age in harvests)
            profiles[crop] = quantity, horizon, len(harvests), actions

        forecast_cache = {}

        def value(crop: str, count: int) -> float:
            if count == 0:
                return 0.0
            quantity, horizon, cycles, actions = profiles[crop]
            if not quantity:
                return -math.inf
            key = crop, count
            if key in forecast_cache:
                return forecast_cache[key]
            total = quantity * count
            if config.use_forecast:
                forecast = market_model.forecast(crop, horizon_turns=horizon, sale_quantity=total)
                expected = float(_field(forecast, "expected_revenue", total * _field(forecast, "expected_price")))
                # Price quantile is an approximate conservative lot valuation;
                # clamp so this penalty cannot reward downside uncertainty.
                downside = min(expected, total * float(_field(forecast, "p10", 0)))
                revenue = expected - config.risk_aversion * max(0.0, expected - downside)
            else:
                # Keep own market impact in the ablation; only remove the
                # temporal forecast, not the cost of selling a large lot.
                inventory = obs.get("market", {}).get("inventory", {})
                if crop in inventory:
                    overrides = obs["market"].get("params") or game_config.get("marketParams")
                    revenue = sale_revenue(crop, inventory[crop], total, overrides)
                else:
                    revenue = total * float(prices.get(crop, 0.0))
            seed_cost = max(0, count * cycles - int(seeds.get(crop, 0))) * CROPS[crop].seed
            profit = revenue - seed_cost - count * actions * config.labor_cost_per_action
            if CROPS[crop].ongoing and profit > 0:
                profit *= 1.0 + config.ongoing_preference
            forecast_cache[key] = profit if math.isfinite(profit) else -math.inf
            return forecast_cache[key]

        scores = {crop: value(crop, 1) for crop in CROPS}
        # Retain the dominant planted crop in each group until its area becomes
        # free; changing a price quote never causes healthy plants to be dug up.
        planted_short = [c for c in SHORT_CROPS if existing[c]]
        allowed_long = LONG_CROPS if config.long_mode == "any" else RECURRENT_CROPS
        planted_long = [c for c in allowed_long if existing[c]]
        short_choices = ([max(planted_short, key=existing.get)] if planted_short
                         else [c for c in SHORT_CROPS if scores[c] > 0])
        long_choices = ([max(planted_long, key=existing.get)] if planted_long
                        else [c for c in allowed_long if scores[c] > 0])
        if not config.dual_cycle:
            long_choices = []
        short_choices = short_choices or [None]
        long_choices = long_choices or [None]
        best = None
        for short in short_choices:
            for long in long_choices:
                protected = sum(count for crop, count in existing.items() if crop not in {short, long})
                cap = max(0, capacity - protected)
                short_old = min(existing[short], cap) if short else 0
                long_old = min(existing[long], max(0, cap - short_old)) if long else 0
                minimum = short_old, long_old

                def objective(s: int, l: int) -> float:
                    if (short is None and s) or (long is None and l):
                        return -math.inf
                    add_s, add_l = s - short_old, l - long_old
                    for crop, addition in ((short, add_s), (long, add_l)):
                        if addition and (crop is None or scores[crop] <= 0):
                            return -math.inf
                    costs = sum(max(0, add - int(seeds.get(crop, 0))) * CROPS[crop].seed
                                for crop, add in ((short, add_s), (long, add_l)) if crop)
                    if costs > cash:
                        return -math.inf
                    # Existing plots are sunk investments. Value the marginal
                    # batch impact caused by additions, without selling existing
                    # plants to finance an unrelated replacement decision.
                    gain = 0.0
                    for crop, count, old in ((short, s, short_old), (long, l, long_old)):
                        if crop and count > old:
                            old_value = value(crop, old) if old else 0.0
                            marginal = value(crop, count) - old_value
                            if not math.isfinite(marginal) or marginal <= 0:
                                return -math.inf
                            gain += marginal
                    return gain

                pair = optimize_allocation(cap, objective, method, minimum, config.beam_width)
                # Require both cycles only if additions can fund a profitable
                # mixed allocation. A target ratio is soft, never an obligation
                # to plant at a loss or to clear the existing farm.
                mixed = None
                if config.dual_cycle and short and long and cap >= 2:
                    def mixed_score(s: int, l: int) -> float:
                        raw = objective(s, l)
                        if not s or not l or not math.isfinite(raw):
                            return -math.inf
                        if not (short_old or long_old):
                            tolerance = max(config.allocation_slack, 1 / (s + l))
                            if abs(s / (s + l) - config.fast_share) > tolerance:
                                return -math.inf
                        # Soft preference once legacy plantings constrain area.
                        deviation = abs(s / (s + l) - config.fast_share)
                        return raw * (1 - 0.15 * deviation) if raw > 0 else raw
                    candidate = optimize_allocation(cap, mixed_score, method, minimum, config.beam_width)
                    if mixed_score(*candidate) > 0:
                        mixed = candidate
                pair = mixed or pair
                score = objective(*pair)
                if best is None or score > best[0]:
                    best = score, short, long, pair

        if best is None:
            return CropPlan(None, None, 0, 0, scores, "Sem área disponível para plantio.")
        _, short, long, (s, l) = best
        descriptions = []
        if s and l:
            descriptions.append("Dois ciclos: cultura rápida para caixa e lenta com retorno previsto positivo.")
        elif s:
            descriptions.append("Ciclo rápido; novas culturas lentas sem horizonte ou margem suficiente.")
        elif l and l > existing[long]:
            descriptions.append("Ciclo longo viável; cultura rápida sem margem suficiente no mercado atual.")
        else:
            descriptions.append("Preservar plantas existentes e caixa; sem forçar novos investimentos.")
        if long and l:
            descriptions.append("Melão tem colheita única, não é recorrente." if long == "MELON" else
                                "Tomate/morango têm somente quatro pulsos sem fertilizante, depois decaem.")
        descriptions.append(f"Teto {capacity} parcelas; reserva {config.cash_reserve:g}; margem final {margin} turnos.")
        return CropPlan(short if s else None, long if l else None, s, l, scores, " ".join(descriptions))

    def _portfolio(self, obs: dict, game_config: dict, market_model) -> CropPlan:
        """Allocate crops with bounded tranche optimization and marginal wages.

        Sales follow executable harvest days, batching recurrent crop pulses.
        Forecast scenario quantiles are empirical risk proxies, not a fitted
        quantile regression. New-crop batches use synchronized cohorts; actual
        worker routing and existing plant ages remain execution approximations.
        """
        config = self.config
        farms = obs.get("farms", [])
        if not farms:
            return CropPlan(None, None, 0, 0, reason="Sem estado de fazenda.")
        player = int(obs.get("player", 0))
        farm = farms[player]
        tiles = [tile for row in farm.get("tiles", []) for tile in row]
        existing = Counter(tile["crop"] for tile in tiles if isinstance(tile, dict)
                           and tile.get("kind") == "PLANT" and tile.get("crop") in CROPS)
        cultivable = sum(tile is None or isinstance(tile, dict)
                         and tile.get("kind") in {"PLANT", "WEED"} for tile in tiles)
        tpd = max(1, int(game_config.get("turnsPerDay", 24)))
        step = int(obs.get("step", int(obs.get("day", 0)) * tpd + int(obs.get("hour", 0))))
        last_action = int(game_config.get("episodeSteps", 720)) - 2
        margin = max(config.logistics_margin, int(game_config.get("boardSize", 10)) + 2)
        last_day = (last_action - margin - 1) // tpd
        day = step // tpd
        capacity = min(config.max_plots, cultivable)
        free = max(0, capacity - sum(existing.values()) - int(game_config.get("reservedAnimalPlots", 0)))
        cash = max(0.0, float(farm.get("money", 0)) - config.cash_reserve
                   - float(game_config.get("reservedAnimalCash", 0)))
        seeds = obs.get("private", {}).get("seeds", {})
        animals = sum(isinstance(tile, dict) and "animal" in tile for tile in tiles)
        committed_labor = (sum(existing.values()) * config.labor_actions_per_plot_day
                           + animals * 6.0 + float(game_config.get("reservedAnimalActions", 0)))
        worker_actions = tpd * config.labor_utilization
        labor = max(0.0, config.target_workers * worker_actions - committed_labor)
        shed_capacity = max(0, int(game_config.get("shedCapacity", 100)))
        # Inventory can be sold before future harvests; reserve only explicitly
        # earmarked storage (for example feed), not transient unsold produce.
        storage = max(0, shed_capacity - int(game_config.get("reservedShedCapacity", 0)))
        profiles = {}
        for crop, spec in CROPS.items():
            batches, start, cycles, actions = [], day, 0, 0.0
            while start + spec.first_day <= last_day:
                age = min(full_cycle_day(crop), last_day - start)
                if age < minimum_cycle_day(crop, config):
                    break
                if spec.ongoing:
                    # Match the execution policy: collect two units at a time,
                    # then liquidate the final finite pulse by the deadline.
                    # Pricing a one-unit sale on every production date both
                    # understates batch pressure and invents extra journeys.
                    projected = projected_crop_harvests(
                        {"crop": crop, "planted_day": start, "yield_units": 0},
                        start, start + age)
                    for harvest_day, quantity in projected:
                        harvest_step = harvest_day * tpd + 1
                        batches.append((max(1, harvest_step - step), quantity))
                else:
                    harvest_step = (start + age) * tpd + 1
                    batches.append((max(1, harvest_step - step), yield_without_fertilizer(crop, age)))
                cycles += 1
                actions += (age + 1) * config.labor_actions_per_plot_day + 2
                start += age + 1
            profiles[crop] = (batches, cycles, actions)
        opposing = Counter(tile.get("crop") for index, other in enumerate(farms) if index != player
                           for row in other.get("tiles", []) for tile in row
                           if isinstance(tile, dict) and tile.get("kind") == "PLANT")
        specialist = config.strategy
        if specialist == "adaptive":
            specialist = ("liquidator" if last_action - step <= 4 * tpd else
                          "cashflow" if cash < 500 else
                          "contrarian" if sum(opposing.values()) >= 8 else "portfolio")
        prices = obs.get("market", {}).get("prices", {})
        inventory = obs.get("market", {}).get("inventory", {})
        overrides = obs.get("market", {}).get("params") or game_config.get("marketParams")
        revenue_cache, value_cache = {}, {}

        def revenue(crop: str, count: int) -> float:
            if not count:
                return 0.0
            key = crop, count
            if key in revenue_cache:
                return revenue_cache[key]
            batches = [(h, q * count) for h, q in profiles[crop][0]]
            if config.use_forecast:
                if hasattr(market_model, "forecast_batches"):
                    forecasts = [(sum(q for _, q in batches), market_model.forecast_batches(crop, batches))]
                else:
                    forecasts = [(q, market_model.forecast(crop, horizon_turns=h, sale_quantity=q))
                                 for h, q in batches]
                expected = sum(float(_field(f, "expected_revenue", q * _field(f, "expected_price")))
                               for q, f in forecasts)
                if hasattr(market_model, "revenue_quantile"):
                    downside = float(market_model.revenue_quantile(crop, batches, config.risk_quantile))
                else:
                    # Test/adaptor fallback interpolates reported scenario
                    # quantiles and never invents regression coefficients.
                    weight = max(0.0, min(1.0, (config.risk_quantile - .1) / .4))
                    downside = sum(q * (float(_field(f, "p10", 0)) * (1 - weight)
                                        + float(_field(f, "p50", _field(f, "expected_price"))) * weight)
                                   for q, f in forecasts)
                result = expected - config.risk_aversion * max(0.0, expected - downside)
            elif crop in inventory:
                # Spot-only ablation retains cumulative own slippage; no town
                # demand or opponent inflow is forecast between harvest days.
                total = sum(q for _, q in batches)
                result = sale_revenue(crop, inventory[crop], total, overrides)
            else:
                result = sum(q for _, q in batches) * float(prices.get(crop, 0.0))
            revenue_cache[key] = result if math.isfinite(result) else -math.inf
            return revenue_cache[key]

        def value(crop: str, addition: int) -> float:
            if not addition:
                return 0.0
            key = crop, addition
            if key in value_cache:
                return value_cache[key]
            _, cycles, actions = profiles[crop]
            if not cycles:
                return -math.inf
            gain = revenue(crop, existing[crop] + addition) - revenue(crop, existing[crop])
            seed_cost = max(0, addition * cycles - int(seeds.get(crop, 0))) * CROPS[crop].seed
            profit = gain - seed_cost - addition * actions * config.labor_cost_per_action
            if opposing[crop]:
                rate = 0.04 if specialist == "contrarian" else 0.025
                profit -= max(0.0, gain) * min(0.40, opposing[crop] * rate)
            if specialist == "cashflow" and crop in LONG_CROPS:
                profit *= .6
            if CROPS[crop].ongoing and profit > 0:
                profit *= 1.0 + config.ongoing_preference
            value_cache[key] = profit if math.isfinite(profit) else -math.inf
            return value_cache[key]

        scores = {crop: value(crop, 1) for crop in CROPS}
        eligible = [crop for crop in CROPS if scores[crop] > 0
                    and (specialist != "melon" or crop == "MELON")
                    and (specialist != "liquidator")]
        horizon_rows = sorted({h for crop in eligible for h, _ in profiles[crop][0]})
        daily = {crop: dict(profiles[crop][0]) for crop in eligible}
        # Keep the same conservative synchronized-cohort assumption for planted
        # land when reserving batch space. Otherwise replanning could repeatedly
        # add a feasible new batch atop a now-existing full batch.
        existing_batches = {crop: dict(profiles[crop][0]) for crop in existing}
        available_storage = {h: max(0, storage - sum(count * existing_batches[crop].get(h, 0)
                                                      for crop, count in existing.items()))
                             for h in horizon_rows}
        def cost(counts):
            return sum(max(0, count - int(seeds.get(crop, 0))) * CROPS[crop].seed
                       for crop, count in counts.items())

        def feasible(counts, cash_budget, labor_budget):
            return (sum(counts.values()) <= free and cost(counts) <= cash_budget + 1e-7
                    and sum(counts.values()) * config.labor_actions_per_plot_day <= labor_budget + 1e-7
                    and all(sum(counts[crop] * daily[crop].get(h, 0) for crop in counts)
                            <= available_storage[h] for h in horizon_rows))

        def fill(counts, cash_budget, labor_budget):
            counts = dict(counts)
            for _ in range(free):
                choices = []
                for crop in eligible:
                    candidate = dict(counts)
                    candidate[crop] += 1
                    if feasible(candidate, cash_budget, labor_budget):
                        gain = value(crop, candidate[crop]) - value(crop, counts[crop])
                        if gain > 0 and math.isfinite(gain):
                            choices.append((gain, crop))
                if not choices:
                    break
                counts[max(choices)[1]] += 1
            return counts

        tranches = []
        for crop in eligible:
            # Seed-stock boundaries make immediate liquidity linear within
            # tranches. Nonincreasing slopes approximate a concave objective.
            breaks = sorted({0, min(free, 5), min(free, 12), free,
                             min(free, max(0, int(seeds.get(crop, 0))))})
            previous_slope = math.inf
            for low, high in zip(breaks, breaks[1:]):
                slope = min(previous_slope, (value(crop, high) - value(crop, low)) / (high - low))
                previous_slope = slope
                if slope <= 0 or not math.isfinite(slope):
                    break
                tranches.append((crop, high - low, slope,
                                 0 if high <= int(seeds.get(crop, 0)) else CROPS[crop].seed))
        matrix = [[1.0] * len(tranches),
                  [float(item[3]) for item in tranches],
                  [config.labor_actions_per_plot_day] * len(tranches)]
        fixed_bounds = [float(free), cash, labor]
        for h in horizon_rows:
            matrix.append([float(daily[crop].get(h, 0)) for crop, _, _, _ in tranches])
            fixed_bounds.append(float(available_storage[h]))
        for index, (_, width, _, _) in enumerate(tranches):
            matrix.append([float(j == index) for j in range(len(tranches))])
            fixed_bounds.append(float(width))

        # Workforce is a small outer enumeration with actual Fibonacci wages.
        # Existing care is a commitment; extra workers must justify their cost
        # through the feasible marginal crop production they can support.
        present = 1 + len(farm.get("hands", []))
        hires = int(farm.get("hires_today", present - 1))
        multiplier = max(0, int(game_config.get("farmHandCostMult", 1)))
        hire_window = int(obs.get("hour", 0)) < min(6, tpd // 2)
        future_days = max(0, last_day - day)
        candidates = []
        unlocked_quads = len(farm.get("unlocked_quadrants", ["NW"]))
        effective_target = min(config.target_workers, 8 if unlocked_quads <= 2 else (9 if unlocked_quads == 3 else 10))
        useful_workers = min(effective_target, max(1, math.ceil(
            (committed_labor + free * config.labor_actions_per_plot_day) / worker_actions)))
        for workers in range(1, useful_workers + 1):
            immediate = (sum(fib_cost(hires + i, multiplier) for i in range(max(0, workers - present)))
                         if hire_window else 0)
            if immediate <= cash + 1e-7:
                daily_wage = sum(fib_cost(i, multiplier) for i in range(workers - 1))
                candidates.append((workers, immediate, immediate + future_days * daily_wage))
        sufficient = [item for item in candidates if item[0] * worker_actions >= committed_labor - 1e-7]
        # An underfunded inherited farm cannot always be made sustainable. Keep
        # plants, avoid extra work, and expose the deficit instead of claiming it
        # satisfies a workforce constraint that the cash cannot actually fund.
        if sufficient:
            candidates = sufficient
        elif candidates:
            candidates = [max(candidates, key=lambda item: item[0])]
        else:
            candidates = [(1, 0, 0)]
        diagnostics, best = [], None
        for index, (workers, immediate, wages) in enumerate(candidates):
            labor_budget = max(0., workers * worker_actions - committed_labor)
            cash_budget = max(0., cash - immediate)
            counts = fill(dict.fromkeys(eligible, 0), cash_budget, labor_budget)
            greedy_counts = dict(counts)
            result = None
            status = "empty"
            if tranches:
                bounds = list(fixed_bounds)
                bounds[1], bounds[2] = cash_budget, labor_budget
                seed_solution, remaining_counts = [], dict(counts)
                for crop, width, _, _ in tranches:
                    quantity = min(width, remaining_counts[crop])
                    seed_solution.append(float(quantity))
                    remaining_counts[crop] -= quantity
                if config.optimizer == "branch_bound":
                    # Share one deterministic LP-node budget across workforce
                    # choices; a zero-budget choice retains its feasible seed.
                    budget = config.integer_node_limit // len(candidates) + int(index < config.integer_node_limit % len(candidates))
                    result = branch_bound([item[2] for item in tranches], matrix, bounds,
                                          node_limit=budget, incumbent=seed_solution)
                else:
                    result = simplex([item[2] for item in tranches], matrix, bounds)
                status = result.status
                if result.variables:
                    counts = {crop: max(0, math.floor(sum(x for x, item in zip(result.variables, tranches)
                                                         if item[0] == crop) + 1e-7)) for crop in eligible}
                    while not feasible(counts, cash_budget, labor_budget) and any(counts.values()):
                        crop = min((c for c in eligible if counts[c]),
                                   key=lambda c: value(c, counts[c]) - value(c, counts[c] - 1))
                        counts[crop] -= 1
                    counts = fill(counts, cash_budget, labor_budget)
                    if sum(value(c, greedy_counts[c]) for c in eligible) > sum(value(c, counts[c]) for c in eligible):
                        counts = greedy_counts
            production_value = sum(value(c, counts[c]) for c in eligible)
            net_value = production_value - wages
            detail = {"workers": workers, "status": status, "hire_cost_now": immediate,
                      "wage_cost_horizon": wages, "crop_value": production_value,
                      "net_value": net_value, "labor_deficit": max(0., committed_labor - workers * worker_actions),
                      "nodes": getattr(result, "nodes", 0), "best_bound": getattr(result, "best_bound", None),
                      "gap": getattr(result, "gap", None)}
            diagnostics.append(detail)
            if best is None or (net_value, -workers) > (best[0], -best[1]):
                best = net_value, workers, counts, detail
        _, workforce_target, additions, selected = best
        status = selected["status"]
        allocation = {crop: existing[crop] + additions.get(crop, 0) for crop in CROPS
                      if existing[crop] + additions.get(crop, 0) > 0}
        short = max((c for c in SHORT_CROPS if allocation.get(c)), key=allocation.get, default=None)
        long = max((c for c in LONG_CROPS if allocation.get(c)), key=allocation.get, default=None)
        reason = (f"Portfólio {specialist}; {config.optimizer} {status}, alocação inteira viável; "
                  f"colheitas cronológicas; teto {capacity}, reserva {config.cash_reserve:g}. "
                  "Previsões e execução aproximadas, sem garantia de ótimo global.")
        return CropPlan(short, long, allocation.get(short, 0), allocation.get(long, 0),
                        scores, reason, allocation=allocation, specialist=specialist,
                        workforce_target=workforce_target, optimizer_diagnostics={
                            "method": config.optimizer, "selected": selected, "workforce_choices": diagnostics,
                            "nodes": sum(item["nodes"] for item in diagnostics),
                            "certificate_scope": "individual linear tranche model only; workforce and nonlinear forecasts are approximations"})
