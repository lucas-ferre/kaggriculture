"""Build a deterministic tar.gz containing only runtime source and its license."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def package(output: Path):
    files = [(ROOT / "main.py", "main.py")]
    package_dir = ROOT / "kaggriculture_agent"
    files.extend((path, path.relative_to(ROOT).as_posix())
                 for path in sorted(package_dir.rglob("*"))
                 if path.is_file() and path.suffix in {".py", ".json", ".toml"}
                 and "__pycache__" not in path.parts)
    files.extend((ROOT / name, name) for name in
                 ("LICENSE-APACHE-2.0.txt", "THIRD_PARTY_NOTICE.txt"))
    if len(files) < 3:
        raise RuntimeError("Runtime package is missing")
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path, name in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        if "main.py" not in names or any(name.startswith(("vendor/", ".venv/", "reports/", "replays/")) for name in names):
            raise RuntimeError("Invalid submission archive")
    return {"path": str(output.resolve()), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "members": names}


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/submission.tar.gz")
    args = parser.parse_args(argv)
    print(json.dumps(package(args.output), indent=2))


if __name__ == "__main__":
    main()
