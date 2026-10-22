"""Build and verify a portable code+MNIST ZIP without training or downloading."""

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from data_pipeline import load_saved_split, verify_raw_integrity
from experiments import canonical_hash, file_hash, make_jobs, variants


def bundle_members(root):
    files = []
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.yaml"), ("docs", "*.md")):
        files.extend(path for path in (root / folder).rglob(pattern) if "__pycache__" not in path.parts)
    files.extend(root / name for name in ("README.md", "requirements-runtime.txt"))
    files.extend((root / "data/MNIST/raw").glob("*-ubyte"))
    files.extend(root / name for name in ("data/processed/split_v1.npz", "outputs/data_check/data_report.json"))
    for name in ("environment_local.json", "pip_freeze_local_cpu.txt", "noise_examples.png"):
        candidate = root / "outputs/data_check" / name
        if candidate.exists():
            files.append(candidate)
    files = sorted(set(files))
    if any(not path.is_file() or not path.resolve().is_relative_to(root) for path in files):
        raise ValueError("Bundle input is missing or outside the project")
    required = {"train.py", "run_experiments.py", "evaluate.py", "summarize.py", "collect_results.py"}
    if not required.issubset({path.name for path in files}):
        raise ValueError("Formal workflow entry points are missing")
    return files


def matrix_csv():
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["variant", "stage", "model", "noise_mode", "training_sigma", "gradient_loss"])
    writer.writeheader()
    for item in variants():
        config = item["config"]
        writer.writerow({"variant": item["variant"], "stage": item["stage"], "model": config["model"]["type"],
                         "noise_mode": config["training"]["noise_mode"],
                         "training_sigma": config["training"].get("sigma", "uniform[0.1,0.5)"),
                         "gradient_loss": config["loss"]["type"] == "mse_gradient"})
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    load_saved_split(root)
    verify_raw_integrity(root)
    files = bundle_members(root)
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in files}
    generated = {"configs/task_matrix.json": json.dumps(make_jobs(), ensure_ascii=False, indent=2) + "\n",
                 "docs/experiment_matrix.csv": matrix_csv()}
    version = canonical_hash({"files": hashes, "generated": generated})[:12]
    directory = root / "dist"
    directory.mkdir(exist_ok=True)
    destination = directory / f"mnist_project_{version}.zip"
    temporary = directory / f"mnist_project_{version}.zip.tmp"
    manifest = {"version": version, "created_utc": datetime.now(timezone.utc).isoformat(),
                "files_sha256": hashes, "formal_configurations": 19, "formal_training_runs": 57,
                "validation_pilot_runs": 3, "note": "Includes MNIST raw files and fixed split. Excludes local environments and training outputs."}
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, "mnist_project/" + path.relative_to(root).as_posix())
        for name, content in generated.items():
            archive.writestr("mnist_project/" + name, content.encode("utf-8"))
        archive.writestr("mnist_project/bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        import hashlib
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read("mnist_project/" + name)).hexdigest() != expected:
                raise RuntimeError(f"ZIP member hash differs: {name}")
    temporary.replace(destination)
    print("BUNDLE READY")
    print(f"ZIP: {destination}")
    print(f"Size: {destination.stat().st_size / 1024**2:.2f} MiB")
    print(f"SHA256: {file_hash(destination)}")
    print("19 configurations x 3 seeds + 3 validation pilots = 60 training jobs.")
    print("No training or downloading was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
