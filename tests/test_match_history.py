"""Auditable single-writer match history and immutable archived episodes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.match_history import append_match, read_history, replay_paths_from_history


class MatchHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.history = self.root / "history.jsonl"
        self.replay = self.root / "episode.json"
        self.replay.write_text(json.dumps({"info": {"seed": 101}, "steps": []}), encoding="utf-8")
        self.result = {"seed": 101, "side": 0, "opponent": "starter", "variant": "adaptive",
                       "valid": True, "money": 10000, "opponent_money": 9000,
                       "policy_source": {"sha256": "policy-a"}, "replay": str(self.replay)}
        self.engine = {"framework_version": "1.32.7", "engine_commit": "commit", "engine_sha256": "engine",
                       "framework_import_messages": "optional imports", "runtime": {"sha256": "unrelated"}}

    def test_identity_ignores_unrelated_runtime_metadata_but_retains_provenance(self):
        self.assertTrue(append_match(self.history, self.result, self.engine))
        changed_engine = copy.deepcopy(self.engine)
        changed_engine["framework_import_messages"] = "other imports"
        changed_engine["runtime"] = {"sha256": "different"}
        changed_engine["baseline_999"] = {"sha256": "different"}
        self.assertFalse(append_match(self.history, self.result, changed_engine))
        records = read_history(self.history)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["engine"], self.engine)
        self.assertEqual(records[0]["replay"]["sha256"], hashlib.sha256(self.replay.read_bytes()).hexdigest())
        self.assertTrue(records[0]["recorded_at_utc"])

    def test_policy_engine_side_and_episode_hash_distinguish_matches(self):
        append_match(self.history, self.result, self.engine)
        for patch in ({"side": 1}, {"policy_source": {"sha256": "policy-b"}}):
            changed = dict(self.result, **patch)
            self.assertTrue(append_match(self.history, changed, self.engine))
        self.assertTrue(append_match(self.history, self.result, dict(self.engine, engine_sha256="other")))
        self.replay.write_text(json.dumps({"info": {"seed": 101}, "steps": [1]}), encoding="utf-8")
        self.assertTrue(append_match(self.history, self.result, self.engine))
        self.assertEqual(len(read_history(self.history)), 5)

    def test_replay_paths_require_valid_results_requested_seeds_and_intact_bytes(self):
        append_match(self.history, self.result, self.engine)
        self.assertEqual(replay_paths_from_history(self.history, [101]), [self.replay.resolve()])
        with self.assertRaisesRegex(ValueError, "seeds"):
            replay_paths_from_history(self.history, [101, 103])
        self.replay.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since collection"):
            replay_paths_from_history(self.history, [101])

    def test_missing_replay_or_invalid_match_can_be_audited_but_not_trained(self):
        missing = dict(self.result, replay=str(self.root / "missing.json"))
        self.assertTrue(append_match(self.history, missing, self.engine))
        self.assertTrue(append_match(self.history, dict(self.result, valid=False), self.engine))
        self.assertEqual(len(read_history(self.history)), 2)
        with self.assertRaisesRegex(ValueError, "No valid, intact replay"):
            replay_paths_from_history(self.history, [101])

    def test_reimport_from_a_different_path_keeps_episode_identity(self):
        append_match(self.history, self.result, self.engine)
        moved = self.root / "copy.json"
        moved.write_bytes(self.replay.read_bytes())
        self.assertFalse(append_match(self.history, dict(self.result, replay=str(moved)), self.engine))

    def test_corrupted_history_is_rejected_without_appending(self):
        append_match(self.history, self.result, self.engine)
        self.history.write_text(self.history.read_text(encoding="utf-8") + '{"schema_version":', encoding="utf-8")
        before = self.history.read_bytes()
        with self.assertRaisesRegex(ValueError, "Invalid history JSON"):
            append_match(self.history, dict(self.result, side=1), self.engine)
        self.assertEqual(self.history.read_bytes(), before)

    def test_valid_record_without_final_newline_still_appends_separately(self):
        append_match(self.history, self.result, self.engine)
        self.history.write_text(self.history.read_text(encoding="utf-8").rstrip(), encoding="utf-8")
        append_match(self.history, dict(self.result, side=1), self.engine)
        self.assertEqual(len(read_history(self.history)), 2)


if __name__ == "__main__":
    unittest.main()
