"""Paired evaluations against the pinned, unmodified Kaggle game engine.

Run from the project root with ``python -m scripts.benchmark``.
The Kaggle framework is an optional development dependency, never submitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import copy
from dataclasses import asdict, is_dataclass
from functools import lru_cache
import hashlib
import importlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import statistics
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "vendor" / "kaggriculture"
ENVIRONMENT_NAME = "kaggriculture_pinned"
FRAMEWORK_IMPORT_MESSAGES = ""
BASELINE_ARCHIVE = ROOT / "dist" / "submission-0.1.0.tar.gz"
BASELINE_NAMESPACE = "_baseline_010"
_BASELINE_TEMPS = {}
VARIANTS = ("adaptive", "bandit", "scenarios", "no_mrlm", "no_fertilizer", "simplex", "baseline_012", "baseline_011", "baseline_010", "legacy_policy", "dual_forecast", "dual_spot", "fast_only", "hungarian", "beam", "long_any")
OPPONENTS = ("starter", "random", "pass", "selfplay", "baseline_012", "baseline_011", "baseline_010", "legacy_policy", "passive",
             "crop_flood", "livestock", "expansive")
PROFILE_DESCRIPTIONS = {
    "starter": "Unmodified pinned engine starter agent.",
    "random": "Unmodified engine random agent; its private RNG is not reproducible.",
    "pass": "Unmodified engine PASS agent.",
    "selfplay": "Independent instance of the current adaptive default policy.",
    "baseline_010": "Frozen original 0.1.0 submission archive, loaded in an isolated package namespace.",
    "baseline_011": "Frozen validated 0.1.1 submission archive, loaded in an isolated package namespace.",
    "baseline_012": "Frozen validated 0.1.2 submission archive, including its fitted quantile coefficients.",
    "legacy_policy": "Legacy dual-cycle parameters on the current runtime; not a frozen 0.1.0 binary.",
    "passive": "No crops, animals or land investment; one worker and spot sales.",
    "crop_flood": "Expanded short-cycle crop production with immediate spot sales, no livestock.",
    "livestock": "Animal-focused farm with at most eight crop plots, eight cows and four sheep.",
    "expansive": "Crop-only adaptive farm targeting sixty plots and all three land expansions.",
}


@lru_cache(maxsize=3)
def load_baseline(version="0.1.0"):
    """Load the local release snapshot without modifying its package or source.

    Only regular Python/JSON files below the exact package prefix are copied. No
    tar extraction API is used, so links, absolute paths and traversal members
    cannot write elsewhere. Imports use a dedicated alias and relative imports.
    """
    if version not in {"0.1.0", "0.1.1", "0.1.2"}:
        raise ValueError("Unknown frozen release")
    archive_path = ROOT / "dist" / f"submission-{version}.tar.gz"
    namespace = "_baseline_" + version.replace(".", "")
    if not archive_path.is_file():
        raise RuntimeError(f"Frozen baseline archive is missing: {archive_path}")
    _BASELINE_TEMPS[version] = tempfile.TemporaryDirectory(prefix=f".arena-baseline-{version}-", dir=ROOT)
    destination = Path(_BASELINE_TEMPS[version].name)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = member.name.split("/")
            if not parts or parts[0] != "kaggriculture_agent":
                continue
            if (not member.isfile() or any(part in ("", ".", "..") or "\\" in part or ":" in part for part in parts)
                    or not member.name.endswith((".py", ".json"))):
                raise RuntimeError(f"Unsafe frozen baseline package member: {member.name}")
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError("Frozen baseline member escapes extraction directory")
            content = archive.extractfile(member)
            if content is None:
                raise RuntimeError(f"Unreadable frozen baseline member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.read())
    package_dir = destination / "kaggriculture_agent"
    spec = importlib.util.spec_from_file_location(
        namespace, package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen baseline package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    spec.loader.exec_module(package)
    if package.__version__ != version:
        raise RuntimeError(f"Frozen baseline declares unexpected version: {package.__version__}")
    return (importlib.import_module(namespace + ".agent").FarmAgent,
            importlib.import_module(namespace + ".planner").PolicyConfig)


def load_baseline_010():
    return load_baseline("0.1.0")


def baseline_metadata(version="0.1.0"):
    path = ROOT / "dist" / f"submission-{version}.tar.gz"
    return {"available": path.is_file(), "version": version, "frozen": True,
            "archive": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


@lru_cache(maxsize=1)
def runtime_metadata():
    """Fingerprint runtime sources at first use in this benchmark process."""
    import kaggriculture_agent
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((ROOT / "kaggriculture_agent").rglob("*"))
              if path.is_file() and path.suffix in {".py", ".json"}}
    serialized = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"version": kaggriculture_agent.__version__, "source": "working_tree", "frozen": False,
            "runtime_sha256": hashes, "runtime_fingerprint_sha256": hashlib.sha256(serialized).hexdigest()}


@lru_cache(maxsize=1)
def load_engine():
    """Register our reviewed engine instead of whichever game ships on PyPI."""
    global FRAMEWORK_IMPORT_MESSAGES
    messages = StringIO()
    try:
        with redirect_stdout(messages):
            import kaggle_environments
    except ImportError as exc:
        raise RuntimeError(
            "Use .venv/Scripts/python.exe; see documentacao/tecnica/ENGINE_NOTES.md for the "
            "minimal Kaggle framework installation."
        ) from exc
    FRAMEWORK_IMPORT_MESSAGES = messages.getvalue().strip()
    path = ENGINE_DIR / "kaggriculture.py"
    spec = importlib.util.spec_from_file_location("_pinned_kaggriculture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned engine: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kaggle_environments.register(ENVIRONMENT_NAME, {
        "agents": module.agents,
        "interpreter": module.interpreter,
        "renderer": module.renderer,
        "html_renderer": module.html_renderer,
        "specification": module.specification,
    })
    return module


def engine_metadata():
    load_engine()
    import kaggle_environments
    return {
        "framework_version": kaggle_environments.__version__,
        "engine_commit": (ENGINE_DIR / "COMMIT.txt").read_text().strip(),
        "engine_sha256": hashlib.sha256(
            (ENGINE_DIR / "kaggriculture.py").read_bytes()
        ).hexdigest(),
        "framework_import_messages": FRAMEWORK_IMPORT_MESSAGES,
        "baseline_010": baseline_metadata(),
        "baseline_011": baseline_metadata("0.1.1"),
        "baseline_012": baseline_metadata("0.1.2"),
        "runtime": runtime_metadata(),
    }


def make_environment(seed=0, steps=720, configuration=None):
    if steps < 2:
        raise ValueError("episodeSteps must be at least 2")
    load_engine()
    from kaggle_environments import make
    cfg = {"episodeSteps": steps, "seed": seed}
    cfg.update(configuration or {})
    return make(ENVIRONMENT_NAME, configuration=cfg, debug=False)


def variant_config(name):
    from kaggriculture_agent.planner import PolicyConfig
    if name in {"baseline_010", "baseline_011", "baseline_012"}:
        _, frozen_config = load_baseline({"baseline_010": "0.1.0", "baseline_011": "0.1.1", "baseline_012": "0.1.2"}[name])
        return frozen_config()
    # Preserve the old parameter set explicitly as defaults evolve. Tactical
    # correctness fixes remain shared: this is a legacy policy, not old code.
    legacy = {"strategy": "dual", "optimizer": "enumerate", "max_plots": 20,
              "target_workers": 6, "enable_livestock": False, "enable_expansion": False}
    overrides = {
        "legacy_policy": {},
        "dual_forecast": {},
        "dual_spot": {"use_forecast": False},
        "fast_only": {"dual_cycle": False},
        "hungarian": {"scheduler": "hungarian"},
        "beam": {"optimizer": "beam"},
        "long_any": {"long_mode": "any"},
    }
    if name == "adaptive":
        return PolicyConfig()
    experimental = {"bandit": {"enable_contextual_bandit": True},
                    "no_mrlm": {"enable_mrlm": False},
                    "scenarios": {"market_method": "scenarios"},
                    "no_fertilizer": {"enable_fertilizer": False},
                    "simplex": {"optimizer": "simplex"}}
    if name in experimental:
        return PolicyConfig(**experimental[name])
    if name not in overrides:
        raise ValueError(f"Unknown variant: {name}")
    return PolicyConfig(**(legacy | overrides[name]))


def opponent_config(name):
    """Synthetic policies share our implementation, with distinct allocations.

    They exercise mechanisms and market pressure; they are not independent
    competitive implementations or downloaded third-party agents.
    """
    from kaggriculture_agent.planner import PolicyConfig
    if name in ("legacy_policy", "baseline_010", "baseline_011", "baseline_012"):
        return variant_config(name)
    overrides = {
        "selfplay": {},
        "passive": {"max_plots": 0, "target_workers": 1, "use_forecast": False,
                    "enable_livestock": False, "enable_expansion": False},
        "crop_flood": {"strategy": "dual", "dual_cycle": False, "optimizer": "enumerate",
                       "max_plots": 40, "target_workers": 10, "use_forecast": False,
                       "enable_livestock": False, "enable_expansion": True,
                       "max_expansions": 2, "cash_reserve": 200.0},
        "livestock": {"max_plots": 8, "target_workers": 12,
                      "enable_livestock": True, "enable_expansion": True,
                      "max_cows": 8, "max_sheep": 4, "max_geese": 0},
        "expansive": {"max_plots": 60, "target_workers": 10,
                      "enable_livestock": False, "enable_expansion": True,
                      "max_expansions": 3, "cash_reserve": 200.0},
    }
    if name not in overrides:
        raise ValueError(f"Opponent has no policy configuration: {name}")
    return PolicyConfig(**overrides[name])


class MeasuredAgent:
    """Time only decisions; preserve exceptions for framework error handling."""
    def __init__(self, agent):
        self.agent = agent
        self.durations = []
        self.errors = []
        self.operations = Counter()
        self.plant_requests = Counter()
        self.market_requests = Counter()
        self.market_units_requested = Counter()

    def __call__(self, obs, configuration):
        started = time.perf_counter()
        try:
            action = self.agent(obs, configuration)
            if isinstance(action, dict):
                actions = [action.get("farmer", ["PASS"])]
                actions.extend(action.get("hands", []))
                for operation in actions:
                    if isinstance(operation, list) and operation:
                        self.operations[operation[0]] += 1
                        if operation[0] == "PLANT" and len(operation) > 1:
                            self.plant_requests[operation[1]] += 1
                for order in action.get("market", []):
                    if isinstance(order, list) and order:
                        self.market_requests[order[0]] += 1
                        if len(order) >= 3 and isinstance(order[2], (int, float)):
                            self.market_units_requested[f"{order[0]}/{order[1]}"] += order[2]
            return action
        except Exception as exc:
            self.errors.append({"step": obs.get("step"), "error": repr(exc)})
            raise
        finally:
            self.durations.append(time.perf_counter() - started)

    def timing(self):
        values = sorted(value * 1000 for value in self.durations)
        if not values:
            return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
        return {
            "count": len(values), "mean_ms": statistics.fmean(values),
            "p95_ms": values[max(0, (95 * len(values) + 99) // 100 - 1)],
            "max_ms": values[-1],
        }


def _crop_statistics(env, side):
    planted = Counter()
    crop_turns = Counter()
    first_planted_day = {}
    peak_plots = Counter()
    previous = {}
    for state in env.steps:
        current = {}
        farm = state[0].observation.farms[side]
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    key = tile["crop"], tile["planted_day"]
                    current[x, y] = key
                    crop_turns[tile["crop"]] += 1
                    if previous.get((x, y)) != key:
                        planted[tile["crop"]] += 1
                        first_planted_day.setdefault(tile["crop"], int(tile["planted_day"]))
        active = Counter(crop for crop, _ in current.values())
        for crop, quantity in active.items():
            peak_plots[crop] = max(peak_plots[crop], quantity)
        previous = current
    recurrent_turns = sum(crop_turns[crop] for crop in ("TOMATO", "STRAWBERRY"))
    return {"successful_plantings": dict(planted), "occupied_tile_states": dict(crop_turns),
            "first_planted_day": first_planted_day, "peak_plots_by_crop": dict(peak_plots),
            "recurrent_occupied_share": recurrent_turns / sum(crop_turns.values()) if crop_turns else 0.0}


def episode_metrics(env, side, audit_unit_actions=True):
    """Summarize recorded states; optionally reconstruct physical unit actions.

    The pinned engine exposes no error counter for illegal physical actions:
    they silently do nothing. Replay each unit action on a private copy using
    that same engine function to count changed/no-op outcomes and collection.
    This runs after the episode, outside decision and episode wall timings.
    Market transactions are not reconstructed, so requests are never reported
    as completed sales. Cash inflows/outflows below are *net per turn*.
    """
    if not env.steps:
        return {"available": False}
    engine = load_engine()
    cfg = env.configuration
    tpd = max(1, int(cfg.get("turnsPerDay", 24)))
    board_size = int(cfg.get("boardSize", 10))
    shed_capacity = int(cfg.get("shedCapacity", 100))
    cash, land, workforce = [], [], []
    peak_animals, final_animals = Counter(), Counter()
    peak_crops = 0
    animal_tile_states = Counter()
    locked_worker_states = 0
    purchase_steps, productive_quadrants = [], set()
    quadrant_tile_states = Counter()
    for state in env.steps:
        farm = state[0].observation.farms[side]
        cash.append(float(farm["money"]))
        land.append(len(farm.get("unlocked_quadrants", [])))
        workforce.append(1 + len(farm.get("hands", [])))
        observation = state[0].observation
        observed_step = int(observation.get("step", int(observation.get("day", 0)) * tpd + int(observation.get("hour", 0))))
        if len(land) > 1 and land[-1] > land[-2]:
            purchase_steps.append(observed_step)
        for x, y in [farm["farmer"], *farm.get("hands", [])]:
            locked_worker_states += farm["tiles"][y][x] == "LOCKED"
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and (tile.get("kind") == "PLANT" or tile.get("animal")):
                    quadrant = ("N" if y < board_size // 2 else "S") + ("W" if x < board_size // 2 else "E")
                    productive_quadrants.add(quadrant)
                    quadrant_tile_states[quadrant] += 1
        animals = Counter(tile["animal"] for row in farm["tiles"] for tile in row
                          if isinstance(tile, dict) and "animal" in tile)
        final_animals = animals
        animal_tile_states.update(animals)
        for animal, quantity in animals.items():
            peak_animals[animal] = max(peak_animals[animal], quantity)
        peak_crops = max(peak_crops, sum(isinstance(tile, dict) and tile.get("kind") == "PLANT"
                                       for row in farm["tiles"] for tile in row))
    changes = [after - before for before, after in zip(cash, cash[1:])]
    final_private = env.steps[-1][side].observation.private
    metrics = {
        "available": True,
        "economy": {"initial_cash": cash[0], "final_cash": cash[-1],
                    "net_cash_change": cash[-1] - cash[0], "minimum_cash": min(cash),
                    "maximum_cash": max(cash),
                    "positive_net_turn_cash": sum(max(0.0, delta) for delta in changes),
                    "negative_net_turn_cash": -sum(min(0.0, delta) for delta in changes),
                    "cash_flow_note": "Net turn cash changes combine sales and spending; these are not gross revenues/costs.",
                    "final_shed": dict(final_private.get("shed", {})),
                    "final_seeds": dict(final_private.get("seeds", {})),
                    "final_carried_units": dict(sum((Counter(inv) for inv in final_private.get("inventories", [])), Counter()))},
        "land": {"initial_quadrants": land[0], "final_quadrants": land[-1],
                 "purchased_quadrants": max(0, land[-1] - land[0]),
                 "purchase_steps": purchase_steps,
                 "productive_quadrants": sorted(productive_quadrants),
                 "productive_tile_states_by_quadrant": dict(quadrant_tile_states),
                 "worker_states_on_locked_tiles": locked_worker_states,
                 "locked_tile_note": "Movement and shed access on locked tiles are legal; tile production needs ownership.",
                 "peak_crop_plots": peak_crops, "peak_workers": max(workforce)},
        "livestock": {"final_animals": dict(final_animals), "peak_animals": dict(peak_animals),
                      "occupied_tile_states": dict(animal_tile_states)},
        "unit_action_audit": {"available": False, "reason": "disabled"},
    }
    if not audit_unit_actions:
        return metrics
    changed, noops, blocked, harvested, placed = Counter(), Counter(), Counter(), Counter(), Counter()
    fertilizer = 0
    audit_errors = []
    for before, after in zip(env.steps, env.steps[1:]):
        # Framework replay records keep shared fields (especially step) on
        # player zero. The live agent receives a merged observation, but these
        # raw side-one records do not: use the shared clock/farms and only the
        # selected side's private inventory. Otherwise side one replays at day
        # zero and valid mature harvests are misclassified as silent no-ops.
        shared = before[0].observation
        farm, private = copy.deepcopy((shared.farms[side], before[side].observation.private))
        action = after[side].action if isinstance(after[side].action, dict) else {}
        hands = action.get("hands", [])
        operations = [action.get("farmer", ["PASS"])] + (hands if isinstance(hands, list) else [])
        plant_demand = Counter(op[1] for op in operations
                               if isinstance(op, list) and len(op) >= 2 and op[0] == "PLANT")
        overbooked = {crop for crop, quantity in plant_demand.items()
                      if quantity > private.get("seeds", {}).get(crop, 0)}
        step = int(shared.get("step", int(shared.get("day", 0)) * tpd + int(shared.get("hour", 0))))
        day = step // tpd
        for unit, operation in enumerate(operations):
            if not isinstance(operation, list) or not operation:
                noops["MALFORMED"] += 1
                continue
            op = operation[0]
            if op == "PASS":
                continue
            if op == "PLANT" and len(operation) > 1 and operation[1] in overbooked:
                blocked[operation[1]] += 1
                noops[op] += 1
                continue
            snapshot = copy.deepcopy((farm, private))
            inventories = private.get("inventories", [])
            previous_inventory = dict(inventories[unit]) if unit < len(inventories) else {}
            try:
                engine._apply_unit_action(farm, private, unit, operation, board_size,
                                          day, tpd, shed_capacity)
            except Exception as exc:
                audit_errors.append({"step": step, "unit": unit, "error": repr(exc)})
                continue
            if snapshot == (farm, private):
                noops[op] += 1
                continue
            changed[op] += 1
            inventory = private.get("inventories", [])
            current_inventory = inventory[unit] if unit < len(inventory) else {}
            if op == "HARVEST":
                harvested.update({item: quantity - previous_inventory.get(item, 0)
                                  for item, quantity in current_inventory.items()
                                  if quantity > previous_inventory.get(item, 0)})
            elif op == "COLLECT_FERTILIZER":
                fertilizer += current_inventory.get("FERTILIZER", 0) - previous_inventory.get("FERTILIZER", 0)
            elif op == "PLACE" and len(operation) > 1 and operation[1] in engine.ANIMALS:
                # Successful animal PLACE can also deposit into the shed; only
                # count the transition when a new animal actually occupies land.
                prior_count = sum(isinstance(tile, dict) and tile.get("animal") == operation[1]
                                  for row in snapshot[0]["tiles"] for tile in row)
                next_count = sum(isinstance(tile, dict) and tile.get("animal") == operation[1]
                                 for row in farm["tiles"] for tile in row)
                placed[operation[1]] += max(0, next_count - prior_count)
    metrics["unit_action_audit"] = {
        "available": not audit_errors, "method": "Copied pre-turn state + pinned engine physical-action replay",
        "changed_actions": dict(changed), "non_pass_noops": dict(noops),
        "non_pass_noop_count": sum(noops.values()), "atomic_blocked_plant_requests": dict(blocked),
        "audit_errors": audit_errors,
        "note": "No-ops are reconstructed ineffective actions, not framework errors. Market no-ops are not measured.",
    }
    metrics["production"] = {"harvested_units": dict(harvested),
                             "fertilizer_collected": fertilizer,
                             "successful_plant_actions": changed["PLANT"]}
    metrics["livestock"]["animals_placed"] = dict(placed)
    metrics["livestock"]["animals_lost"] = {animal: max(0, quantity - final_animals[animal])
                                             for animal, quantity in placed.items()}
    return metrics


def save_replay(path, payload):
    """Preserve old episodes when repeating a seed with a different policy."""
    path = Path(path)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        digest = hashlib.sha256(encoded).hexdigest()
        path = path.with_name(f"{path.stem}_{digest}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Replay hash collision or modified archive: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(encoded)
    return path


def run_match(config=None, seed=0, opponent="starter", side=0, steps=720,
              replay_path=None, audit_unit_actions=True):
    """Run one genuine episode. Selfplay opponent uses independent default policy.

    Official random_agent creates an unseeded Random() each call; seed controls
    the environment only, and that opponent is deliberately left unmodified.
    """
    from kaggriculture_agent.agent import FarmAgent
    from kaggriculture_agent.planner import PolicyConfig
    current_source = runtime_metadata()
    if side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    if opponent not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent}")
    config = config or PolicyConfig()
    if isinstance(config, dict):
        config = PolicyConfig(**config)
    env = make_environment(seed, steps)
    frozen_versions = {"_baseline_010": "0.1.0", "_baseline_011": "0.1.1", "_baseline_012": "0.1.2"}
    frozen_version = frozen_versions.get(type(config).__module__.split(".")[0])
    frozen_candidate = frozen_version is not None
    candidate_factory = load_baseline(frozen_version)[0] if frozen_candidate else FarmAgent
    measured = MeasuredAgent(candidate_factory(config))
    other_config = opponent_config(opponent) if opponent not in ("starter", "random", "pass") else None
    other_version = {"baseline_010": "0.1.0", "baseline_011": "0.1.1", "baseline_012": "0.1.2"}.get(opponent)
    opponent_factory = load_baseline(other_version)[0] if other_version else FarmAgent
    other = MeasuredAgent(opponent_factory(other_config)) if other_config is not None else opponent
    players = [other, other]
    players[side] = measured
    started = time.perf_counter()
    run_error = None
    try:
        env.run(players)
    except Exception as exc:
        run_error = repr(exc)
    elapsed = time.perf_counter() - started
    final = env.state
    cash = [float(farm["money"]) for farm in final[0].observation.farms]
    bad_states = sorted({
        player.status for state in env.steps for player in state
        if player.status not in ("ACTIVE", "INACTIVE", "DONE")
    })
    # Logs also retain errors on the final step, where the engine sets DONE.
    stderr = [
        {"state": index, "player": player, "stderr": log.get("stderr")}
        for index, logs in enumerate(env.logs)
        for player, log in enumerate(logs)
        if isinstance(log, dict) and log.get("stderr")
    ]
    other_errors = other.errors if isinstance(other, MeasuredAgent) else []
    valid = (run_error is None and not measured.errors and not other_errors and not bad_states
             and not stderr and all(player.status == "DONE" for player in final)
             and len(env.steps) == steps)
    margin = cash[side] - cash[1 - side]
    result = {
        "seed": seed, "side": side, "opponent": opponent,
        "opponent_reproducible": opponent != "random",
        "opponent_description": PROFILE_DESCRIPTIONS[opponent],
        "opponent_policy": asdict(other_config) if other_config is not None else None,
        "policy_source": baseline_metadata(frozen_version) if frozen_candidate else current_source,
        "opponent_source": (baseline_metadata(other_version) if other_version else current_source
                            if other_config is not None else {"frozen": True, "source": "pinned_engine"}),
        "policy": asdict(config) if is_dataclass(config) else dict(vars(config)),
        "valid": valid, "money": cash[side], "opponent_money": cash[1 - side],
        "margin": margin,
        "outcome": ("win" if margin > 0 else "loss" if margin < 0 else "draw") if valid else "invalid",
        "statuses": [player.status for player in final],
        "bad_statuses_seen": bad_states, "recorded_states": len(env.steps),
        "last_observation_step": final[0].observation.get("step"),
        "decisions": measured.timing(), "wall_seconds": elapsed,
        "operations_requested": dict(measured.operations),
        "market_orders_requested": dict(measured.market_requests),
        "market_units_requested": dict(measured.market_units_requested),
        "plant_requests": dict(measured.plant_requests),
        "crop_mix": _crop_statistics(env, side),
        "metrics": episode_metrics(env, side, audit_unit_actions),
        "opponent_metrics": episode_metrics(env, 1 - side, audit_unit_actions),
        "opponent_decisions": other.timing() if isinstance(other, MeasuredAgent) else None,
        "error_counts": {"agent_exceptions": len(measured.errors), "opponent_exceptions": len(other_errors),
                         "framework_stderr_entries": len(stderr), "bad_status_types": len(bad_states),
                         "run_exceptions": int(run_error is not None)},
        "errors": measured.errors, "framework_stderr": stderr,
        "opponent_errors": other_errors,
        "run_error": run_error,
    }
    if replay_path is not None:
        replay_path = save_replay(replay_path, env.toJSON())
        result["replay"] = str(replay_path)
    return result


def summarize(matches):
    valid = [match for match in matches if match["valid"]]
    outcomes = Counter(match["outcome"] for match in matches)
    return {
        "matches": len(matches), "valid_matches": len(valid),
        "outcomes": dict(outcomes),
        "win_rate": outcomes["win"] / len(valid) if valid else None,
        "score_rate": (outcomes["win"] + 0.5 * outcomes["draw"]) / len(valid) if valid else None,
        "mean_money": statistics.fmean(m["money"] for m in valid) if valid else None,
        "mean_margin": statistics.fmean(m["margin"] for m in valid) if valid else None,
        "median_money": statistics.median(m["money"] for m in valid) if valid else None,
        "min_money": min((m["money"] for m in valid), default=None),
        "max_decision_ms": max((m["decisions"]["max_ms"] or 0 for m in matches), default=0),
        "mean_wall_seconds": statistics.fmean(m["wall_seconds"] for m in matches) if matches else None,
        "action_audit_available_matches": sum(m.get("metrics", {}).get("unit_action_audit", {}).get("available", False)
                                              for m in matches),
        "non_pass_noop_count": sum(m.get("metrics", {}).get("unit_action_audit", {}).get("non_pass_noop_count", 0)
                                    for m in matches),
        "note": "Descriptive paired-seed sample; invalid episodes excluded from financial and outcome averages.",
    }


def main(argv=None):
    from scripts.arena import rate_benchmark
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["adaptive", "baseline_012"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS,
                        default=["starter", "passive", "crop_flood", "livestock", "expansive", "selfplay"])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "benchmark.json")
    parser.add_argument("--replays", type=Path, default=ROOT / "outputs" / "replays",
                        help="Save immutable full episodes for regression (default: outputs/replays)")
    parser.add_argument("--no-replays", action="store_true", help="Save summary history only, without training data")
    parser.add_argument("--history", type=Path, default=ROOT / "outputs" / "match_history.jsonl",
                        help="Append each completed result to a deduplicated persistent history")
    parser.add_argument("--skip-action-audit", action="store_true",
                        help="Skip post-episode physical action reconstruction; retain state and cash metrics")
    parser.add_argument("--rating-regularization", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.rating_regularization <= 0:
        parser.error("--rating-regularization must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Do not repeat seeds")
    results = []
    for variant in args.variants:
        for opponent in args.opponents:
            for seed in args.seeds:
                for side in (0, 1):
                    replay = args.replays / f"{variant}_{opponent}_{seed}_{side}.json" if not args.no_replays else None
                    result = run_match(variant_config(variant), seed, opponent, side, args.steps,
                                       replay, audit_unit_actions=not args.skip_action_audit)
                    result["variant"] = variant
                    results.append(result)
                    from scripts.match_history import append_match
                    append_match(args.history, result, engine_metadata())
                    print(f"{variant} vs {opponent} seed={seed} side={side}: "
                          f"{result['outcome']} money={result['money']:.0f} "
                          f"margin={result['margin']:.0f}", file=sys.stderr, flush=True)
    summaries = {
        f"{variant}/{opponent}": summarize([
            m for m in results if m["variant"] == variant and m["opponent"] == opponent
        ]) for variant in args.variants for opponent in args.opponents
    }
    report = {"engine": engine_metadata(), "episode_steps": args.steps, "history": str(args.history),
              "seeds": args.seeds, "paired_sides": True,
              "opponent_profiles": {name: PROFILE_DESCRIPTIONS[name] for name in args.opponents},
              "note": "Local baselines measure functionality and ablations; they do not estimate leaderboard strength.",
              "legacy_note": "Legacy variants use 0.1.0 policy parameters with current runtime corrections; they are not the frozen 0.1.0 artifact.",
              "ratings": rate_benchmark(results, args.rating_regularization),
              "summary": summaries, "matches": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0 if all(match["valid"] for match in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
