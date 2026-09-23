"""Regularized Bradley--Terry estimates for a local, actually played arena.

This is a descriptive local model, not a reproduction of Kaggle's ratings.
Draws contribute half a win to each player. A Gaussian prior keeps ratings
finite for undefeated players and disconnected comparison graphs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math


def _solve(matrix, vector):
    """Solve the small positive-definite Newton system without dependencies."""
    n = len(vector)
    rows = [list(row) + [float(value)] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-14:
            raise ValueError("Singular rating system")
        for j in range(col, n + 1):
            rows[col][j] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = rows[row][col]
            for j in range(col, n + 1):
                rows[row][j] -= factor * rows[col][j]
    return [row[-1] for row in rows]


def bradley_terry(matches, regularization=1.0):
    """Fit valid outcomes with ``player_a``, ``player_b`` and ``score_a``.

    score_a must be 0, .5 or 1. Invalid episodes are excluded, never silently
    assigned losses. Rating differences use the conventional 400/log(10)
    scale; the location 1500 is arbitrary. Uncertainty is a prior-conditioned
    Laplace standard deviation, not a frequentist confidence interval.
    """
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    played = []
    excluded = 0
    recorded, valid_count = 0, 0
    excluded_by_reason = Counter()
    for match in matches:
        recorded += 1
        if not match.get("valid", True):
            excluded += 1
            excluded_by_reason["invalid_match"] += 1
            continue
        a, b, score = match["player_a"], match["player_b"], match["score_a"]
        if score not in (0, 0.5, 1):
            raise ValueError("score_a must be 0, .5 or 1")
        valid_count += 1
        if a == b:
            excluded += 1
            excluded_by_reason["same_player"] += 1
            continue
        played.append((a, b, float(score)))
    names = sorted({name for a, b, _ in played for name in (a, b)})
    index = {name: i for i, name in enumerate(names)}
    pairs = [(index[a], index[b], score) for a, b, score in played]
    n = len(names)
    abilities = [0.0] * n

    def derivatives(values):
        gradient = [-regularization * value for value in values]
        information = [[regularization if i == j else 0.0 for j in range(n)]
                       for i in range(n)]
        for a, b, score in pairs:
            delta = max(-700.0, min(700.0, values[a] - values[b]))
            probability = 1.0 / (1.0 + math.exp(-delta))
            residual = score - probability
            gradient[a] += residual
            gradient[b] -= residual
            weight = probability * (1.0 - probability)
            information[a][a] += weight
            information[b][b] += weight
            information[a][b] -= weight
            information[b][a] -= weight
        return gradient, information

    def objective(values):
        value = -0.5 * regularization * sum(x * x for x in values)
        for a, b, score in pairs:
            delta = values[a] - values[b]
            log_partition = max(0.0, delta) + math.log1p(math.exp(-abs(delta)))
            value += score * delta - log_partition
        return value

    converged = not n
    iterations = 0
    for iterations in range(1, 101) if n else ():
        gradient, information = derivatives(abilities)
        step = _solve(information, gradient)
        if max(map(abs, step), default=0.0) < 1e-9:
            converged = True
            break
        scale, current = 1.0, objective(abilities)
        proposal = [value + change for value, change in zip(abilities, step)]
        while objective(proposal) < current - 1e-12 and scale > 1e-8:
            scale *= 0.5
            proposal = [value + scale * change for value, change in zip(abilities, step)]
        abilities = proposal

    counts = Counter()
    points = Counter()
    ties = Counter()
    graph = defaultdict(set)
    for a, b, score in played:
        counts[a] += 1
        counts[b] += 1
        points[a] += score
        points[b] += 1.0 - score
        ties[a] += score == 0.5
        ties[b] += score == 0.5
        graph[a].add(b)
        graph[b].add(a)
    components, unseen = [], set(names)
    while unseen:
        pending, component = [min(unseen)], []
        while pending:
            name = pending.pop()
            if name not in unseen:
                continue
            unseen.remove(name)
            component.append(name)
            pending.extend(graph[name] & unseen)
        components.append(sorted(component))
    _, information = derivatives(abilities)
    conversion = 400.0 / math.log(10.0)
    ratings = []
    for i, name in enumerate(names):
        column = _solve(information, [float(j == i) for j in range(n)])
        ratings.append({
            "player": name, "rating": 1500.0 + conversion * abilities[i],
            "log_ability": abilities[i],
            "approx_prior_conditioned_sd": conversion * math.sqrt(max(0.0, column[i])),
            "matches": counts[name], "points": points[name], "draws": ties[name],
            "distinct_opponents": len(graph[name]),
        })
    ratings.sort(key=lambda row: (-row["rating"], row["player"]))
    return {
        "method": "Bradley-Terry with Gaussian L2 prior; draws = half a win",
        "regularization": regularization, "played_matches": len(played),
        "recorded_matches": recorded, "valid_matches": valid_count,
        "excluded_by_reason": dict(excluded_by_reason),
        "excluded_matches": excluded, "converged": converged, "iterations": iterations,
        "connected_components": components, "ratings": ratings,
        "caveat": (
            "Descriptive local ratings from these opponents and seeds only. Small samples, "
            "shared seeds and reused policy families limit independence and generalization. "
            "Uncertainty is a prior-conditioned Laplace approximation, not a confidence "
            "interval. Disconnected components cannot be ranked against each other from "
            "game evidence. These are not official Kaggle ratings or leaderboard estimates."
        ),
    }


def rate_benchmark(matches, regularization=1.0):
    """Canonicalize actual code+configuration before fitting played outcomes.

    Candidate/opponent roles are not player identities: an adaptive candidate
    and its independent selfplay opponent are the same policy. Frozen release
    opponents likewise share identity with that release's candidate variant.
    Fingerprinted code and serialized configuration take precedence over names;
    older reports without fingerprints use disclosed named-policy fallbacks.
    """
    groups, games = {}, []

    def identity(match, opponent):
        name = match["opponent"] if opponent else match.get("variant", "candidate")
        alias = ("opponent:" if opponent else "variant:") + name
        normalized = "adaptive" if name == "selfplay" else name
        policy = match.get("opponent_policy" if opponent else "policy")
        source = match.get("opponent_source" if opponent else "policy_source") or {}
        code_hash = source.get("runtime_fingerprint_sha256") or source.get("sha256")
        if name in {"starter", "random", "pass"} and policy is None:
            implementation = "pinned_engine:" + name
            label = "engine:" + name
        else:
            implementation = ("sha256:" + code_hash if code_hash else
                              "unfingerprinted:" + str(source.get("version", normalized)))
            label = "agent:" + normalized
        description = {"implementation": implementation, "policy": policy}
        # Lacking both configuration and source evidence, only known same-name
        # aliases are merged. Do not infer equivalence of arbitrary old variants.
        if policy is None and code_hash is None:
            description["named_policy"] = normalized
        encoded = json.dumps(description, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        group = groups.setdefault(fingerprint, {
            "identity_sha256": fingerprint, "implementation": implementation,
            "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if policy is not None else None,
            "fingerprinted": code_hash is not None, "aliases": set(), "labels": set(),
            "same_policy_matches": 0,
        })
        group["aliases"].add(alias)
        group["labels"].add(label)
        return fingerprint

    for match in matches:
        games.append({"player_a": identity(match, False), "player_b": identity(match, True),
                      "score_a": {"win": 1., "draw": .5, "loss": 0.}.get(match["outcome"], .5),
                      "valid": match["valid"]})
    # Prefer the default policy name when several aliases have identical code
    # and parameters; append a digest only when one name denotes different code
    # or configurations within the supplied evidence.
    labels = {}
    for key, group in groups.items():
        labels[key] = min(group["labels"], key=lambda name: (name != "agent:adaptive", name))
    duplicates = Counter(labels.values())
    labels = {key: name + (":" + key[:12] if duplicates[name] > 1 else "")
              for key, name in labels.items()}
    for game in games:
        if game["valid"] and game["player_a"] == game["player_b"]:
            groups[game["player_a"]]["same_policy_matches"] += 1
        game["player_a"], game["player_b"] = labels[game["player_a"]], labels[game["player_b"]]
    result = bradley_terry(games, regularization)
    result["excluded_same_policy_matches"] = result["excluded_by_reason"].get("same_player", 0)
    result["identity_method"] = (
        "Identical source fingerprint + serialized policy share one identity across roles. "
        "Selfplay is the default adaptive policy. Older unfingerprinted records use named "
        "aliases only. Same-policy games remain disclosed but add no skill evidence; "
        "played_matches counts only games included in the fit."
    )
    result["policy_identities"] = [
        {**{name: value for name, value in group.items() if name not in {"aliases", "labels"}},
         "player": labels[key], "aliases": sorted(group["aliases"])}
        for key, group in sorted(groups.items(), key=lambda item: labels[item[0]])
    ]
    return result
