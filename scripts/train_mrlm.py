"""Fit MRLM prices, calibrate residuals, and validate direction probabilities.

Three seed-disjoint episode sets separate coefficient fitting, residual
calibration and the exploratory deployment gate. Only decision-time public
features enter the regression. Final scores and future orders never do.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from kaggriculture_agent.market import MarketModel, PRODUCTS, _params
from kaggriculture_agent.mrlm import CLASS_NAMES, MODEL_KIND, SCHEMA_VERSION, predict_ratio, residual_probabilities
from kaggriculture_agent.quantile import FEATURE_NAMES, configuration_signature, public_features, within_training_domain
from scripts.match_history import replay_paths_from_history
from scripts.train_quantile import replay_manifest, replay_paths, validate_splits


def _solve(matrix, target):
    """Small dense linear system with partial pivoting, using only stdlib."""
    augmented = [list(row) + [value] for row, value in zip(matrix, target)]
    n = len(target)
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular regression system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            weight = augmented[row][column]
            for j in range(column, n + 1):
                augmented[row][j] -= weight * augmented[column][j]
    return [row[-1] for row in augmented]


def fit_linear_regression(rows, l2=.01):
    """Minimize mean squared error + L2 on standardized non-intercept terms."""
    if not rows or not math.isfinite(l2) or l2 <= 0:
        raise ValueError("MRLM requires training rows and a positive finite L2 penalty")
    width, count = len(rows[0]["features"]), len(rows)
    if (not width or any(len(row["features"]) != width for row in rows)
            or any(not math.isfinite(value) for row in rows
                   for value in [*row["features"], row["target_ratio"]])):
        raise ValueError("Regression rows must have consistent, finite features and targets")
    means = [statistics.fmean(row["features"][j] for row in rows) for j in range(width)]
    scales = [max(1e-6, statistics.pstdev(row["features"][j] for row in rows)) for j in range(width)]
    matrix = [[0.] * (width + 1) for _ in range(width + 1)]
    target = [0.] * (width + 1)
    for row in rows:
        values = [1.] + [(x - mean) / scale for x, mean, scale in zip(row["features"], means, scales)]
        for j, value in enumerate(values):
            target[j] += value * row["target_ratio"] / count
            for k, other in enumerate(values):
                matrix[j][k] += value * other / count
    for j in range(1, width + 1):
        matrix[j][j] += l2
    fit = {"coefficients": _solve(matrix, target), "feature_means": means, "feature_scales": scales,
           "feature_min": [min(row["features"][j] for row in rows) for j in range(width)],
           "feature_max": [max(row["features"][j] for row in rows) for j in range(width)],
           "training_rows": count}
    fit["training_rmse_ratio"] = math.sqrt(statistics.fmean(
        (row["target_ratio"] - predict_ratio(fit, row["features"])) ** 2 for row in rows))
    return fit


def calibrate_residuals(fit, rows):
    """Keep only held-out, in-domain residuals; never refit the regression."""
    fit = dict(fit)
    fit["calibration_residuals"] = sorted(
        row["target_ratio"] - predict_ratio(fit, row["features"])
        for row in rows if within_training_domain(fit, row["features"]))
    fit["calibration_rows"] = len(fit["calibration_residuals"])
    return fit


def collect_rows(paths, horizon=24, stride=12, baseline_method="quadrature"):
    """Causal replay extraction; the later frame supplies only the price target."""
    rows = defaultdict(list)
    for path in paths:
        replay = json.loads(path.read_bytes())
        frames, cfg = replay["steps"], replay["configuration"]
        models = [MarketModel(dict(cfg, forecastingMethod=baseline_method, quantileModel=None, mrlmModel=None))
                  for _ in (0, 1)]
        for index, frame in enumerate(frames):
            shared = frame[0]["observation"]
            step = int(shared.get("step", index))
            for side, model in enumerate(models):
                obs = dict(shared, player=side, private=frame[side]["observation"].get("private", {}))
                action = frame[side].get("action") or {}
                model.update(obs, action.get("market", []) if isinstance(action, dict) else [])
                if step % stride or index + horizon >= len(frames) - 1:
                    continue
                future = frames[index + horizon][0]["observation"]
                if int(future.get("step", index + horizon)) != step + horizon:
                    raise ValueError("Replay has missing/misaligned time steps")
                for product in PRODUCTS:
                    params = _params(product, shared["market"].get("params") or cfg.get("marketParams"))
                    base = max(1., params["base"])
                    current, target = shared["market"]["prices"][product], future["market"]["prices"][product]
                    forecast = model.forecast(product, horizon)
                    rows[product].append({
                        "features": public_features(obs, product, horizon, cfg),
                        "target_ratio": target / base, "current_price": current, "base": base,
                        "outcome": 0 if target > current else 1 if target == current else 2,
                        "baseline_probabilities": [forecast.prob_up, forecast.prob_flat, forecast.prob_down],
                        "episode": str(path.resolve()), "side": side, "step": step,
                    })
    return rows


def probability_metrics(rows, predictions):
    if not rows:
        return {"rows": 0, "brier": None, "ece": None, "class_calibration": {}}
    class_calibration = {}
    for k, name in enumerate(CLASS_NAMES):
        bins = []
        for bin_index in range(10):
            members = [(probability[k], float(row["outcome"] == k))
                       for row, probability in zip(rows, predictions)
                       if min(9, int(probability[k] * 10)) == bin_index]
            if members:
                bins.append({"bin": bin_index, "rows": len(members),
                             "mean_probability": statistics.fmean(p for p, _ in members),
                             "observed_frequency": statistics.fmean(y for _, y in members)})
        class_calibration[name] = {
            "mean_probability": statistics.fmean(p[k] for p in predictions),
            "observed_frequency": statistics.fmean(row["outcome"] == k for row in rows),
            "ece": sum(b["rows"] * abs(b["mean_probability"] - b["observed_frequency"])
                       for b in bins) / len(rows), "bins": bins}
    return {"rows": len(rows), "brier": statistics.fmean(
        sum((p[k] - float(row["outcome"] == k)) ** 2 for k in range(3))
        for row, p in zip(rows, predictions)),
        "ece": statistics.fmean(item["ece"] for item in class_calibration.values()),
        "class_calibration": class_calibration}


def evaluation(rows, fit):
    eligible = [within_training_domain(fit, row["features"]) and fit["calibration_rows"] >= 30 for row in rows]
    candidate = [residual_probabilities(fit, row["features"], row["current_price"], row["base"])
                 if allowed else row["baseline_probabilities"] for row, allowed in zip(rows, eligible)]
    selected_rows = [row for row, allowed in zip(rows, eligible) if allowed]
    selected_predictions = [probability for probability, allowed in zip(candidate, eligible) if allowed]
    return {"episodes": len({row["episode"] for row in rows}), "eligible_rows": len(selected_rows),
            "deployed": probability_metrics(rows, candidate),
            "eligible": probability_metrics(selected_rows, selected_predictions),
            "baseline": probability_metrics(rows, [row["baseline_probabilities"] for row in rows])}


def train_and_validate(training_paths, calibration_paths, validation_paths, horizon=24, stride=12,
                       l2=.01, baseline_method="quadrature"):
    if horizon < 1 or stride < 1 or not math.isfinite(l2) or l2 <= 0:
        raise ValueError("Invalid regression settings")
    manifests = [replay_manifest(paths) for paths in (training_paths, calibration_paths, validation_paths)]
    if any(not manifest for manifest in manifests):
        raise ValueError("All three replay splits must be nonempty")
    for i, first in enumerate(manifests):
        if any(row["seed"] is None for row in first):
            raise ValueError("Replay seeds are required for leakage checks")
        if len({row["sha256"] for row in first}) != len(first):
            raise ValueError("Duplicate episode within a replay split")
        for second in manifests[i + 1:]:
            validate_splits(first, second)
    signature = manifests[0][0]["configuration"]
    if any(row["configuration"] != signature for manifest in manifests for row in manifest):
        raise ValueError("All replay splits must share price curves and calendar")
    enough_seeds = all(len({row["seed"] for row in manifest}) >= 2 for manifest in manifests)
    training, calibration, validation = [collect_rows(paths, horizon, stride, baseline_method)
                                         for paths in (training_paths, calibration_paths, validation_paths)]
    products, fits = {}, {}
    for product in PRODUCTS:
        if not training[product] or not calibration[product] or not validation[product]:
            raise ValueError(f"Insufficient replay duration for horizon/stride: {product}")
        fit = calibrate_residuals(fit_linear_regression(training[product], l2), calibration[product])
        measured = evaluation(validation[product], fit)
        improved = measured["deployed"]["brier"] < .98 * measured["baseline"]["brier"]
        calibrated = measured["eligible"]["ece"] is not None and measured["eligible"]["ece"] <= .15
        enough_rows = fit["calibration_rows"] >= 30 and measured["eligible_rows"] >= 30
        gate = {"two_distinct_seeds_per_split": enough_seeds, "sufficient_eligible_rows": enough_rows,
                "brier_improved_2pct": improved, "eligible_ece_at_most_015": calibrated}
        fit["approved"] = all(gate.values())
        products[product] = {"approved": fit["approved"], "validation": measured, "gate": gate,
                             "training_rows": fit["training_rows"], "training_rmse_ratio": fit["training_rmse_ratio"],
                             "calibration_rows": fit["calibration_rows"]}
        if fit["approved"]:
            fits[product] = fit
    provenance = {name + "_replay_sha256": [row["sha256"] for row in manifest]
                  for name, manifest in zip(("training", "calibration", "validation"), manifests)}
    model = {"schema_version": SCHEMA_VERSION, "model_kind": MODEL_KIND, "feature_names": list(FEATURE_NAMES),
             "class_names": list(CLASS_NAMES), "horizon_turns": horizon, "training_configuration": signature,
             "approved": bool(fits), "models": fits, "provenance": provenance}
    report = {"approved": bool(fits), "approved_products": sorted(fits), "products": products,
              "training_replays": manifests[0], "calibration_replays": manifests[1], "validation_replays": manifests[2],
              "settings": {"horizon_turns": horizon, "stride": stride, "l2": l2, "baseline_method": baseline_method},
              "method": "Multiple linear price regression with L2, training-only standardization, and held-out empirical residual CDF with one pseudo-count per possible price-direction event",
              "gate": "At least two distinct seeds per split, at least 30 in-domain calibration and validation rows, >2% deployed multiclass Brier improvement, and in-domain mean per-class 10-bin ECE <=0.15",
              "limitations": "Local exploratory deployment filter, not an untouched final test or Kaggle score prediction. Overlapping horizons and both player sides are correlated. Pooled residuals assume sufficiently stable error distributions; new opponents or policies can invalidate calibration. Probabilities describe future spot direction only; batch revenue and impact remain empirical. Retrain and validate on new-policy episodes before relying on old-policy calibration.",
              "model": model}
    return report, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "calibration", "validation"):
        parser.add_argument(f"--{name}-replays", nargs="+")
        parser.add_argument(f"--{name}-seeds", nargs="+", type=int)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--overwrite-model", action="store_true")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--l2", type=float, default=.01)
    parser.add_argument("--baseline-method", choices=["quadrature", "scenarios"], default="quadrature")
    args = parser.parse_args(argv)
    if args.model_output.name == "quantile_coefficients.json":
        parser.error("MRLM must use a separate model artifact; the deployed quantile file is protected")
    if args.output.resolve() == args.model_output.resolve():
        parser.error("Report and model output must have distinct paths")
    if args.model_output.exists() and not args.overwrite_model:
        parser.error("Model output already exists; choose a new path or explicitly use --overwrite-model")
    splits = []
    try:
        for name in ("train", "calibration", "validation"):
            patterns, seeds = getattr(args, name + "_replays"), getattr(args, name + "_seeds")
            if args.history:
                if patterns or not seeds:
                    parser.error("With --history, supply seed lists for all three splits and no replay patterns")
                splits.append(replay_paths_from_history(args.history, seeds))
            else:
                if seeds or not patterns:
                    parser.error("Supply all three replay splits, or use --history with all three seed lists")
                splits.append(replay_paths(patterns))
        report, model = train_and_validate(*splits, args.horizon, args.stride, args.l2, args.baseline_method)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path, payload in ((args.output, report), (args.model_output, model)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"approved": report["approved"], "approved_products": report["approved_products"],
                      "validation_brier": {product: result["validation"]["deployed"]["brier"]
                                           for product, result in report["products"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
