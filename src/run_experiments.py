"""Serial, resumable formal training queue. No test-set evaluation here."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from data_pipeline import load_saved_split, verify_raw_integrity
from experiments import (SEEDS, STAGES, file_hash, make_jobs, resolve_config,
                         select_gradient_weight, verified_selection)
from train import source_hashes
from training import atomic_json, write_csv


def completed_run_matches(run, config, sources, split_hash):
    if not (run / "status.json").exists():
        return False
    if json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] != "completed":
        return False
    actual = json.loads((run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if (actual != config or manifest["source_sha256"] != sources or manifest["split_sha256"] != split_hash
            or summary["completed_epochs"] != config["training"]["epochs"]
            or summary["purpose"] != config["purpose"] or summary["train_size"] != 55000
            or summary["validation_size"] != 5000):
        raise ValueError(f"Completed run does not match the current protocol: {run}")
    if not all((run / name).exists() for name in ("last.pt", "metrics.csv", "validation_per_image.csv")):
        raise ValueError(f"Completed run has missing artifacts: {run}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--stage", choices=("all",) + STAGES, default="baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-jobs", type=int, help="Stop after this many newly run jobs; completed jobs are skipped")
    parser.add_argument("--list", action="store_true", help="Print the queue without running or writing anything")
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    work = work.resolve()
    jobs = [job for job in make_jobs(args.seeds) if args.stage == "all" or job["stage"] == args.stage]
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if args.list:
        for job in jobs:
            print(f"{job['stage']:12} {job['config']['purpose']:7} {job['job_id']}")
        print(f"Selected: {len(jobs)} runs | Full plan: 57 formal + 3 validation pilots")
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This queue requests CUDA. Enable a GPU, or explicitly choose --device cpu.")
    load_saved_split(root)
    verify_raw_integrity(root)
    sources = source_hashes(root)
    split_hash = file_hash(root / "data/processed/split_v1.npz")
    work.mkdir(parents=True, exist_ok=True)
    (work / "resolved_configs").mkdir(exist_ok=True)
    atomic_json(work / "task_matrix.json", make_jobs())
    index, executed = [], 0

    def record(job, status, run):
        index.append({"job_id": job["job_id"], "stage": job["stage"], "seed": job["seed"],
                      "purpose": job["config"]["purpose"], "status": status, "run_dir": str(run)})
        write_csv(work / "last_queue.csv", list(index[0]), index)

    for number, job in enumerate(jobs, 1):
        selection = select_gradient_weight(work, sources, split_hash)
        config = resolve_config(job, args.device, selection,
                                file_hash(work / "gradient_selection.json") if selection else None)
        run = work / "runs" / job["job_id"]
        if completed_run_matches(run, config, sources, split_hash):
            print(f"[{number}/{len(jobs)}] Already complete: {job['job_id']}", flush=True)
            record(job, "completed_previously", run)
            continue
        if args.max_jobs is not None and executed >= args.max_jobs:
            print("Batch limit reached; rerun the same command to continue.", flush=True)
            break
        resolved = work / "resolved_configs" / (job["job_id"] + ".yaml")
        if (run / "last.pt").exists():
            saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if saved != config:
                raise ValueError(f"Interrupted run has different settings: {run}")
            command = [sys.executable, "-u", str(root / "src/train.py"), "--root", str(root), "--resume", str(run)]
        else:
            if run.exists():
                raise RuntimeError(f"Incomplete initialization without a checkpoint: {run}. Preserve it and inspect before retrying.")
            resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "-u", str(root / "src/train.py"), "--root", str(root),
                       "--config", str(resolved), "--run-dir", str(run)]
        print(f"\n[{number}/{len(jobs)}] Running: {job['job_id']}", flush=True)
        try:
            result = subprocess.run(command, cwd=root)
        except KeyboardInterrupt:
            record(job, "interrupted", run)
            print("Queue interrupted; the next invocation resumes the saved epoch.", flush=True)
            return 130
        if result.returncode:
            record(job, f"failed_exit_{result.returncode}", run)
            return result.returncode
        if not completed_run_matches(run, config, sources, split_hash):
            raise RuntimeError("Training exited successfully but its completion record is missing")
        executed += 1
        record(job, "completed_now", run)
    select_gradient_weight(work, sources, split_hash)
    atomic_json(work / "queue_summary.json", {"selected_jobs": len(jobs), "executed_now": executed,
        "completed_in_this_selection": len(index), "stage": args.stage, "seeds": args.seeds,
        "note": "Read per-run status for the full plan; last_queue.csv records this invocation only."})
    print(f"\nQUEUE BATCH FINISHED | newly run: {executed} | records: {work}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
