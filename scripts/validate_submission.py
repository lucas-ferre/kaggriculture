"""Run the packaged main.py in a clean interpreter against the pinned engine."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/submission.tar.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/entrypoint_validation.json")
    args = parser.parse_args()
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix="validate-submission-") as temporary:
        target = Path(temporary)
        with tarfile.open(args.archive) as archive:
            archive.extractall(target, filter="data")
        code = """
import contextlib, hashlib, importlib.util, io, json, pathlib, sys
with contextlib.redirect_stdout(io.StringIO()):
    import kaggle_environments as ke
engine_path = pathlib.Path(sys.argv[1])
main_path = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("pinned_engine_validation", engine_path)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
ke.register("package_validation", dict(
    agents=engine.agents, interpreter=engine.interpreter, renderer=engine.renderer,
    html_renderer=engine.html_renderer, specification=engine.specification))
env = ke.make("package_validation", configuration={"seed":73,"episodeSteps":720}, debug=False)
env.run([str(main_path), str(main_path)])
import kaggriculture_agent
module_path = pathlib.Path(kaggriculture_agent.__file__).resolve()
assert module_path.is_relative_to(main_path.parent.resolve()), module_path
errors = [log["stderr"] for logs in env.logs for log in logs if log.get("stderr")]
bad_status = sorted({s.status for states in env.steps for s in states
                     if s.status not in {"ACTIVE","INACTIVE","DONE"}})
result = dict(
    source="extracted submission archive, clean Python -I process",
    recorded_states=len(env.steps), statuses=[s.status for s in env.state],
    money=[s.reward for s in env.state], errors=errors, bad_statuses=bad_status,
    engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest())
print(json.dumps(result))
assert len(env.steps)==720 and all(s.status=="DONE" for s in env.state)
assert not errors and not bad_status
"""
        run = subprocess.run([sys.executable, "-I", "-c", code,
                              str(ROOT / "vendor/kaggriculture/kaggriculture.py"),
                              str(target / "main.py")], cwd=target,
                             capture_output=True, text=True, timeout=60)
        if run.returncode:
            raise RuntimeError(run.stderr + run.stdout)
        report = json.loads(run.stdout)
        import hashlib
        report["archive_sha256"] = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
