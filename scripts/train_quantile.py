"""Train regularized price quantiles on local, episode-disjoint replay splits.

No replay is used as an interactive opponent. Targets are future observed spot
prices; features contain only the earlier public observation. Hyperparameters
are fixed before validation; approval is an explicit exploratory deployment
gate, not an untouched final test or a claim of competitive improvement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics

from kaggriculture_agent.market import MarketModel, PRODUCTS, _params
from kaggriculture_agent.quantile import FEATURE_NAMES, SCHEMA_VERSION, configuration_signature, public_features, within_training_domain


def pinball(target, prediction, tau):
    error = target - prediction
    return error * (tau if error >= 0 else tau - 1.)


def _coordinate_minimum(features, residual, tau, penalty):
    """Exact scalar minimizer of mean pinball plus penalty * coefficient**2."""
    n = len(features)
    points = sorted((y / x, abs(x) / n) for x, y in zip(features, residual) if abs(x) > 1e-14)
    if not points:
        return 0.0
    slope = sum(-tau * x if x > 0 else (1. - tau) * x for x in features) / n
    previous, index = -math.inf, 0
    while index < len(points):
        threshold = points[index][0]
        if penalty > 0:
            root = -slope / (2. * penalty)
            if previous <= root <= threshold:
                return root
        jump = 0.0
        while index < len(points) and points[index][0] == threshold:
            jump += points[index][1]
            index += 1
        before = slope + 2. * penalty * threshold
        if before <= 1e-12 and before + jump >= -1e-12:
            return threshold
        slope += jump
        previous = threshold
    return -slope / (2. * penalty) if penalty else points[-1][0]


def fit_linear_quantile(rows, tau=.25, l2=.01, max_sweeps=80):
    """Cyclic exact coordinate minimization; bounded approximate global fit.

    Pinball loss is convex but nonsmooth. Convergence of this finite cyclic
    solver is reported descriptively, without claiming a global optimum.
    Validation is never passed to this function or used to standardize inputs.
    """
    if not rows or not 0 < tau < 1 or l2 <= 0 or max_sweeps < 1:
        raise ValueError("Training requires samples, tau in (0,1), positive L2 and sweeps")
    width, n = len(rows[0]["features"]), len(rows)
    means = [statistics.fmean(row["features"][j] for row in rows) for j in range(width)]
    scales = [max(1e-6, statistics.pstdev(row["features"][j] for row in rows)) for j in range(width)]
    design = [[1.] + [(value - mean) / scale for value, mean, scale in zip(row["features"], means, scales)]
              for row in rows]
    target = [float(row["target_ratio"]) for row in rows]
    coefficients = [sorted(target)[int((n - 1) * tau)]] + [0.] * width
    prediction = [coefficients[0]] * n
    converged = False
    for sweep in range(1, max_sweeps + 1):
        largest = 0.0
        for j in range(width + 1):
            column = [row[j] for row in design]
            residual = [y - p + coefficients[j] * x for y, p, x in zip(target, prediction, column)]
            new = _coordinate_minimum(column, residual, tau, l2 if j else 0.)
            delta = new - coefficients[j]
            coefficients[j] = new
            prediction = [p + delta * x for p, x in zip(prediction, column)]
            largest = max(largest, abs(delta))
        if largest < 1e-7:
            converged = True
            break
    return {"coefficients": coefficients, "feature_means": means, "feature_scales": scales,
            "feature_min": [min(row["features"][j] for row in rows) for j in range(width)],
            "feature_max": [max(row["features"][j] for row in rows) for j in range(width)],
            "training_rows": n, "sweeps": sweep, "coordinate_change_converged": converged,
            "training_objective": statistics.fmean(pinball(y, p, tau) for y, p in zip(target, prediction))
                                  + l2 * sum(beta * beta for beta in coefficients[1:])}


def predict_ratio(fit, features):
    return max(0., fit["coefficients"][0] + sum(beta * (x - mean) / scale for beta, x, mean, scale in
                                             zip(fit["coefficients"][1:], features,
                                                 fit["feature_means"], fit["feature_scales"])))


def replay_paths(patterns):
    found = set()
    for pattern in patterns:
        path = Path(pattern)
        candidates = path.glob("*.json") if path.is_dir() else map(Path, glob.glob(pattern))
        found.update(candidate.resolve() for candidate in candidates if candidate.is_file())
    if not found:
        raise ValueError("No local replay files matched")
    return sorted(found)


def replay_manifest(paths):
    manifest = []
    for path in paths:
        payload = path.read_bytes()
        replay = json.loads(payload)
        if replay.get("statuses") != ["DONE", "DONE"] or len(replay.get("steps", [])) < 2:
            raise ValueError(f"Replay is not a completed two-player episode: {path}")
        manifest.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
                         "seed": replay.get("info", {}).get("seed"),
                         "configuration": configuration_signature(replay.get("configuration"), replay["steps"][0][0]["observation"].get("market")),
                         "recorded_states": len(replay["steps"])})
    return manifest


def validate_splits(training, validation):
    if {row["sha256"] for row in training} & {row["sha256"] for row in validation}:
        raise ValueError("Train and validation replay episodes overlap")
    train_seeds = {row["seed"] for row in training if row["seed"] is not None}
    validation_seeds = {row["seed"] for row in validation if row["seed"] is not None}
    if train_seeds & validation_seeds:
        raise ValueError("Train and validation seeds overlap, including opposite player sides")


def collect_rows(paths, horizon=24, stride=12, tau=.25, baseline_method="quadrature"):
    """Read each replay causally; the future frame supplies only the target."""
    rows = defaultdict(list)
    for path in paths:
        replay = json.loads(path.read_bytes())
        frames, cfg = replay["steps"], replay["configuration"]
        config = dict(cfg, forecastingMethod=baseline_method, quantileModel=None)
        models = [MarketModel(config), MarketModel(config)]
        for index, frame in enumerate(frames):
            shared = frame[0]["observation"]
            step = int(shared.get("step", index))
            for side, model in enumerate(models):
                obs = dict(shared, player=side, private=frame[side]["observation"].get("private", {}))
                previous_action = frame[side].get("action") or {}
                previous_orders = previous_action.get("market", []) if isinstance(previous_action, dict) else []
                model.update(obs, previous_orders)
                if step % stride or index + horizon >= len(frames) - 1:
                    continue
                future = frames[index + horizon][0]["observation"]
                if int(future.get("step", index + horizon)) != step + horizon:
                    raise ValueError("Replay has missing/misaligned time steps")
                for product in PRODUCTS:
                    params = _params(product, shared["market"].get("params") or cfg.get("marketParams"))
                    base = max(1., params["base"])
                    rows[product].append({
                        "features": public_features(obs, product, horizon, cfg),
                        "target_ratio": future["market"]["prices"][product] / base,
                        "baseline_ratio": model.price_quantile(product, horizon, tau) / base,
                        "base": base, "episode": path.name, "side": side, "step": step,
                    })
    return rows


def evaluation(rows, fit, tau):
    predictions = [max(1. / row["base"], predict_ratio(fit, row["features"])) for row in rows]
    eligible = [within_training_domain(fit, row["features"]) for row in rows]
    deployed = [p if allowed else row["baseline_ratio"] for row, p, allowed in zip(rows, predictions, eligible)]
    learned = [(row, p) for row, p, allowed in zip(rows, predictions, eligible) if allowed]
    return {"rows": len(rows), "episodes": len({row["episode"] for row in rows}),
            "eligible_rows": len(learned),
            "deployed_pinball": statistics.fmean(pinball(row["target_ratio"], p, tau) for row, p in zip(rows, deployed)),
            "eligible_coverage_le": statistics.fmean(row["target_ratio"] <= p + 1e-12 for row, p in learned) if learned else None,
            "eligible_coverage_lt": statistics.fmean(row["target_ratio"] < p - 1e-12 for row, p in learned) if learned else None,
            "pinball": statistics.fmean(pinball(row["target_ratio"], p, tau) for row, p in zip(rows, predictions)),
            "empirical_pinball": statistics.fmean(pinball(row["target_ratio"], row["baseline_ratio"], tau) for row in rows),
            "coverage_le": statistics.fmean(row["target_ratio"] <= p + 1e-12 for row, p in zip(rows, predictions)),
            "coverage_lt": statistics.fmean(row["target_ratio"] < p - 1e-12 for row, p in zip(rows, predictions))}


def train_and_validate(training_paths, validation_paths, horizon=24, stride=12, tau=.25,
                       l2=.01, max_sweeps=80, baseline_method="quadrature"):
    training, validation = replay_manifest(training_paths), replay_manifest(validation_paths)
    validate_splits(training, validation)
    signature = training[0]["configuration"]
    if any(episode["configuration"] != signature for episode in [*training, *validation]):
        raise ValueError("All training/validation episodes must share the target price curves and calendar")
    train_rows = collect_rows(training_paths, horizon, stride, tau, baseline_method)
    validation_rows = collect_rows(validation_paths, horizon, stride, tau, baseline_method)
    products, fits = {}, {}
    enough_episodes = len(training) >= 2 and len(validation) >= 2
    for product in PRODUCTS:
        fit = fit_linear_quantile(train_rows[product], tau, l2, max_sweeps)
        measured = evaluation(validation_rows[product], fit, tau)
        # Discrete prices have point masses: tau may lie between P(Y<q) and
        # P(Y<=q). Tolerance is fixed here before observing validation results.
        covered = (measured["eligible_rows"] > 0 and measured["eligible_coverage_lt"] <= tau + .15
                   and measured["eligible_coverage_le"] >= tau - .15)
        improved = measured["deployed_pinball"] < .98 * measured["empirical_pinball"]
        fit["approved"] = bool(enough_episodes and measured["eligible_rows"] >= 30 and covered and improved)
        products[product] = {"approved": fit["approved"], "validation": measured,
                             "training": {key: fit[key] for key in ("training_rows", "sweeps", "coordinate_change_converged", "training_objective")},
                             "gate": {"sufficient_episodes": enough_episodes, "pinball_improved_2pct": improved,
                                      "coverage_within_fixed_tolerance": covered}}
        # Unapproved fits are not bundled and cannot accidentally become active.
        if fit["approved"]:
            fits[product] = fit
    approved = bool(fits)
    model = {"schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
             "approved": approved, "quantile_tau": tau, "horizon_turns": horizon, "models": fits,
             "training_configuration": signature,
             "provenance": {"training_replay_sha256": [row["sha256"] for row in training],
                            "validation_replay_sha256": [row["sha256"] for row in validation]}}
    report = {"approved": approved, "approved_products": sorted(fits), "products": products,
              "training_replays": training, "validation_replays": validation,
              "settings": {"horizon_turns": horizon, "stride": stride, "tau": tau, "l2": l2,
                           "max_sweeps": max_sweeps, "empirical_baseline_method": baseline_method},
              "fit": "Linear pinball + L2 with training-only standardization and bounded coordinate minimization",
              "gate": "At least 2 episodes per split, >=30 eligible validation rows per product, >2% deployed pinball improvement over empirical baseline (out-of-domain uses empirical fallback), eligible learned coverage interval within fixed +/-0.15 of tau",
              "limitations": "Small local synthetic-policy sample; paired sides and overlapping horizons are correlated. Validation is a deployment filter, not an untouched final test. New opponents/strategies can shift price distributions. Calibration applies only to single future spot prices at the trained horizon/tau; multi-lot revenue remains empirical.",
              "model": model}
    return report, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-replays", nargs="+", required=True)
    parser.add_argument("--validation-replays", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--tau", type=float, default=.25)
    parser.add_argument("--l2", type=float, default=.01)
    parser.add_argument("--max-sweeps", type=int, default=80)
    parser.add_argument("--baseline-method", choices=["scenarios", "quadrature"], default="quadrature")
    args = parser.parse_args(argv)
    if args.horizon < 1 or args.stride < 1 or not 0 < args.tau < 1 or args.l2 <= 0 or args.max_sweeps < 1:
        parser.error("Invalid training settings")
    report, model = train_and_validate(replay_paths(args.train_replays), replay_paths(args.validation_replays),
                                       args.horizon, args.stride, args.tau, args.l2,
                                       args.max_sweeps, args.baseline_method)
    for path, payload in ((args.output, report), (args.model_output, model)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"approved": report["approved"], "products": report["products"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
