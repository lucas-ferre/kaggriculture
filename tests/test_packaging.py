"""Package checks use a clean Python process, without importing source modules."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from scripts.package_submission import package, ROOT


class PackagingTests(unittest.TestCase):
    def test_deterministic_archive_and_isolated_stdlib_entrypoint(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix="package-test-") as name:
            directory = Path(name)
            first = package(directory / "first.tar.gz")
            second = package(directory / "second.tar.gz")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertIn("main.py", first["members"])
            self.assertIn("THIRD_PARTY_NOTICE.txt", first["members"])
            self.assertIn("LICENSE-APACHE-2.0.txt", first["members"])
            self.assertFalse(any(p.startswith(("vendor/", ".venv/", "tests/", "scripts/"))
                                 for p in first["members"]))
            extracted = directory / "extracted"
            extracted.mkdir()
            with tarfile.open(first["path"]) as archive:
                archive.extractall(extracted, filter="data")
            # -I excludes cwd/source checkout; -S excludes site-packages. The
            # submission must work with only stdlib and its extracted modules.
            code = """
import importlib.util, json, sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("submitted_main", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from kaggriculture_agent.market import MARKET_PARAMS
farm = {"tiles": [[None if x<5 and y<5 else "LOCKED" for x in range(10)] for y in range(10)],
        "farmer": [4,4], "hands": [], "money": 3000, "hires_today": 0, "unlocked_quadrants": ["NW"]}
obs = {"step":0,"day":0,"hour":0,"player":0,"farms":[farm,farm],
       "private":{"shed":{},"seeds":{},"inventories":[{}]},
       "market":{"inventory":{k:10000 for k in MARKET_PARAMS},
                 "prices":{k:v["base"] for k,v in MARKET_PARAMS.items()}},
       "town":{"unlocked_shops":[]}}
action = module.agent(obs, {"episodeSteps":720})
assert set(action) == {"farmer", "hands", "market"}, action
assert all(isinstance(order, list) for order in action["market"])
assert len(action["market"]) <= 10
print(json.dumps(action))
"""
            result = subprocess.run([sys.executable, "-I", "-S", "-c", code,
                                     str(extracted / "main.py")],
                                    cwd=extracted, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["market"])


if __name__ == "__main__":
    unittest.main()
