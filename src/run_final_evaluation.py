"""Verify the consolidated package, evaluate all saved models, summarize and back up."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from experiments import file_hash, resolve_config, verified_selection
from evaluate import discover_jobs
from run_experiments import completed_run_matches
from train import source_hashes
from training import atomic_json


def verify_package(root):
    manifest = json.loads((root / "final_evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "mnist_final_evaluation" or manifest["formal_runs"] != 57:
        raise ValueError("Unexpected package manifest")
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Package member missing or changed: {name}")
    work = root / "outputs/formal"
    sources = source_hashes(root)
    split_hash = file_hash(root / "data/processed/split_v1.npz")
    selection = verified_selection(work, sources, split_hash)
    if selection is None or selection["weight"] != 0.1:
        raise ValueError("Frozen gradient selection is missing or inconsistent")
    jobs = discover_jobs(work, "test")
    for job in jobs:
        expected = resolve_config(job, "cuda", selection, file_hash(work / "gradient_selection.json"))
        if not completed_run_matches(work / "runs" / job["job_id"], expected, sources, split_hash):
            raise ValueError(f"Incomplete formal model: {job['job_id']}")
    print(f"PACKAGE VERIFIED | {len(jobs)} formal models | frozen gradient weight=0.1", flush=True)
    return manifest


def check_environment(root, manifest):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before running final evaluation.")
    expected = manifest["training_environment"]["packages"]
    packages = {name: importlib.metadata.version(name) for name in expected}
    sensitive = ("torch", "torchvision", "numpy", "scikit-image", "scipy")
    changed = {name: {"expected": expected[name], "actual": packages[name]}
               for name in sensitive if packages[name] != expected[name]}
    if changed:
        raise RuntimeError("Evaluation libraries differ; send this message for review before continuing: "
                           + json.dumps(changed))
    environment = {"python": platform.python_version(), "packages": packages,
                   "device": "cuda", "gpu": torch.cuda.get_device_name(0),
                   "cuda_build": torch.version.cuda, "package_version": manifest["version"]}
    path = root / "outputs/formal/final_evaluation_environment.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != environment:
        raise ValueError("Evaluation environment changed since the previous invocation; preserve results and inspect.")
    atomic_json(path, environment)
    print(f"GPU: {environment['gpu']} | torch: {packages['torch']}", flush=True)


def run(root, filename, *arguments):
    subprocess.run([sys.executable, "-u", str(root / "src" / filename),
                    "--root", str(root), *arguments], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true",
                        help="Check package and training records only; no image evaluation or GPU requirement")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify_package(root)
    if args.verify_only:
        print("No evaluation or training was performed.")
        return 0
    # Kaggle uses Linux. An OS lock prevents two notebook cells writing the same evaluation.
    import fcntl
    with (root / "outputs/formal/final_evaluation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another final evaluation is running; keep only one active cell.") from error
        check_environment(root, manifest)
        try:
            print("[1/3] Evaluating 57 saved models and 2 references at 7 noise levels...", flush=True)
            run(root, "evaluate.py", "--split", "test", "--device", "cuda")
            print("[2/3] Writing tables and comparison figures...", flush=True)
            run(root, "summarize.py", "--split", "test", "--device", "cuda")
        except subprocess.CalledProcessError:
            print("Evaluation/report step failed; preserving current results in a backup ZIP.", flush=True)
            run(root, "collect_results.py")
            raise
        print("[3/3] Backing up all training records and final evaluation outputs...", flush=True)
        run(root, "collect_results.py")
        print("FINAL EVALUATION PACKAGE READY | download the ZIP printed above", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
