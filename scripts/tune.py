"""Compare random search and coordinate hill climbing with equal match budgets.

Training seeds select parameters; validation seeds are evaluated only after
both methods finish. The default policy is also evaluated on validation seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import statistics
import sys

from scripts.benchmark import ROOT, engine_metadata, run_match, summarize

SEARCH_SPACE = {
    "cash_reserve": (150., 300., 500., 800.),
    "target_workers": (6, 8, 10, 12),
    "risk_aversion": (0.0, 0.2, 0.35, 0.6, 1.0),
    "max_plots": (16, 24, 32, 40),
    "risk_quantile": (.1, .15, .25, .4),
    "max_cows": (4, 6, 8, 10),
}


def evaluate_policy(config, seeds, opponent="starter", steps=720):
    matches = [run_match(config, seed, opponent, side, steps)
               for seed in seeds for side in (0, 1)]
    summary = summarize(matches)
    # Any execution failure makes the candidate inadmissible, regardless of money.
    score = summary["mean_margin"] if all(m["valid"] for m in matches) else None
    return {"config": asdict(config), "score": score,
            "summary": summary, "matches": matches}


def _rank(result):
    score = result["score"]
    return float("-inf") if score is None else score


def search(method, baseline, trials, seeds, opponent, steps, rng_seed):
    rng = random.Random(rng_seed)
    trace = []
    incumbent = baseline
    for trial in range(trials):
        if trial == 0:
            candidate = baseline
        elif method == "random":
            candidate = replace(baseline, **{
                name: rng.choice(values) for name, values in SEARCH_SPACE.items()
            })
        else:
            neighbors = []
            for name, values in SEARCH_SPACE.items():
                index = min(range(len(values)), key=lambda i: abs(values[i] - getattr(incumbent, name)))
                for next_index in (index - 1, index + 1):
                    if 0 <= next_index < len(values):
                        neighbors.append(replace(incumbent, **{name: values[next_index]}))
            # Stochastic coordinate hill climbing: one neighboring candidate per
            # trial and strict improvement acceptance. Same evaluation budget.
            candidate = rng.choice(neighbors)
        result = evaluate_policy(candidate, seeds, opponent, steps)
        result["trial"] = trial
        if not trace or _rank(result) > max(_rank(item) for item in trace):
            incumbent = candidate
        trace.append(result)
        print(f"{method} trial={trial + 1}/{trials}: score={result['score']} "
              f"config={asdict(candidate)}", file=sys.stderr, flush=True)
    best = max(trace, key=_rank)
    return {"selected_config": best["config"], "training_score": best["score"],
            "training_trace": trace,
            "training_match_budget": trials * len(seeds) * 2}


def main(argv=None):
    from kaggriculture_agent.planner import PolicyConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8,
                        help="Candidates per method, including baseline")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[101, 103, 107])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[211, 223, 227])
    parser.add_argument("--opponent", choices=["starter", "selfplay"], default="starter")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--search-seed", type=int, default=20260919)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/tuning.json")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive")
    if set(args.train_seeds) & set(args.validation_seeds):
        parser.error("Training and validation seeds must be disjoint")
    if len(set(args.train_seeds)) != len(args.train_seeds) or len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error("Do not repeat seeds inside a split")
    baseline = PolicyConfig()
    methods = {method: search(method, baseline, args.trials, args.train_seeds,
                              args.opponent, args.steps, args.search_seed)
               for method in ("random", "hill_climbing")}
    # No selection or parameter changes after validation has been observed.
    for result in methods.values():
        result["validation"] = evaluate_policy(
            PolicyConfig(**result["selected_config"]), args.validation_seeds,
            args.opponent, args.steps)
    report = {
        "engine": engine_metadata(), "steps": args.steps,
        "train_seeds": args.train_seeds, "validation_seeds": args.validation_seeds,
        "search_seed": args.search_seed, "search_space": SEARCH_SPACE,
        "objective": "Mean bank-money margin against fixed opponent, both sides; any error rejects candidate.",
        "validation_baseline": evaluate_policy(baseline, args.validation_seeds, args.opponent, args.steps),
        "methods": methods,
        "note": "Small local experiment; validation is descriptive, not proof of competitive strength. No policy file is changed automatically.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({method: {"training_score": value["training_score"],
                               "validation": value["validation"]["summary"],
                               "selected_config": value["selected_config"]}
                      for method, value in methods.items()}, indent=2))
    evaluations = [report["validation_baseline"]]
    for value in methods.values():
        evaluations.extend(value["training_trace"])
        evaluations.append(value["validation"])
    return 0 if all(item["score"] is not None for item in evaluations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
