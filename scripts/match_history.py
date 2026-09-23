"""Persistent local match history for audits and episode-disjoint regression.

The JSONL ledger stores results, policy/engine provenance and immutable replay
hashes. It is a single-writer offline tool; the Kaggle runtime never writes it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1


def read_history(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid history JSON at {path}:{line_number}") from exc
        if (not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION
                or not isinstance(record.get("match_id"), str) or not isinstance(record.get("result"), dict)):
            raise ValueError(f"Unsupported history record at {path}:{line_number}")
        records.append(record)
    return records


def append_match(history_path, result, engine=None):
    """Append one result, returning False when this exact match is already stored.

    Pass a benchmark result after setting its variant. A missing replay does
    not lose the summary, but that entry cannot train a regression model.
    """
    replay = None
    if result.get("replay"):
        path = Path(result["replay"]).resolve()
        replay = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                  if path.is_file() else None}
    identity = {key: result.get(key) for key in (
        "seed", "side", "opponent", "variant", "policy", "policy_source", "opponent_policy",
        "opponent_source", "recorded_states", "valid", "money", "opponent_money")}
    # Runtime/archive fingerprints already belong to the relevant policy
    # sources. Unrelated installed baselines and import logs are not identity.
    identity["engine"] = {key: engine.get(key) for key in (
        "framework_version", "engine_commit", "engine_sha256")} if engine else None
    identity["replay_sha256"] = replay["sha256"] if replay else None
    match_id = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
    history_path = Path(history_path)
    if any(record["match_id"] == match_id for record in read_history(history_path)):
        return False
    record = {"schema_version": SCHEMA_VERSION, "match_id": match_id,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "engine": engine,
              "replay": replay, "result": result}
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if history_path.exists() and history_path.stat().st_size:
        with history_path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                prefix = "\n"
    with history_path.open("a", encoding="utf-8", newline="\n") as stream:
        # Reject an interrupted final record in read_history instead of
        # silently replacing or accepting part of a previous experiment.
        stream.write(prefix + encoded + "\n")
    return True


def replay_paths_from_history(history_path, seeds):
    """Resolve valid archived episodes and check that replay bytes are intact."""
    requested, paths, found = set(seeds), set(), set()
    for record in read_history(history_path):
        result, replay = record["result"], record.get("replay")
        if result.get("seed") not in requested or result.get("valid") is not True or not replay:
            continue
        path = Path(replay["path"])
        if not path.is_file() or not replay.get("sha256"):
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != replay["sha256"]:
            raise ValueError(f"Archived replay changed since collection: {path}")
        paths.add(path.resolve())
        found.add(result["seed"])
    if requested - found:
        raise ValueError(f"No valid, intact replay in history for seeds: {sorted(requested - found)}")
    if not paths:
        raise ValueError("No archived replays selected")
    return sorted(paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("outputs/match_history.jsonl"))
    parser.add_argument("--import-report", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    added = 0
    for path in args.import_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        for result in report.get("matches", []):
            added += append_match(args.history, result, report.get("engine"))
    records = read_history(args.history)
    print(json.dumps({"history": str(args.history.resolve()), "added": added, "matches": len(records),
                      "valid_matches": sum(record["result"].get("valid") is True for record in records),
                      "with_replay_hash": sum(bool(record.get("replay", {}).get("sha256"))
                                              for record in records if record.get("replay")),
                      "seeds": sorted({record["result"]["seed"] for record in records
                                       if record["result"].get("seed") is not None})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
