"""Final-epoch evaluation on fixed noise; final test waits for all 57 formal runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import load_test_partition, load_training_partitions, verify_raw_integrity
from experiments import SIGMAS, file_hash, make_jobs, resolve_config, verified_selection
from metrics import per_image_gradient_error, per_image_metrics, summarize_metrics
from models import build_model, predict_clean
from noise import fixed_evaluation_noise
from references import gaussian_denoise, tune_gaussian
from run_experiments import completed_run_matches
from train import source_hashes
from training import atomic_json, synchronize, write_csv


METRICS = ("MSE", "PSNR", "SSIM", "GradientMAE")
PER_IMAGE_FIELDS = ["sigma", "image_id", "label", *METRICS]


def discover_jobs(work, split):
    formal = [job for job in make_jobs() if job["config"]["purpose"] == "formal"]
    completed, missing = [], []
    for job in formal:
        path = work / "runs" / job["job_id"] / "status.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["status"] == "completed":
            completed.append(job)
        else:
            missing.append(job["job_id"])
    if split == "test" and missing:
        raise ValueError(f"Final test requires all 57 formal runs. Missing {len(missing)}; first: {missing[:4]}")
    if not completed:
        raise ValueError("No completed formal training runs found")
    return completed


@torch.no_grad()
def score_method(loader, sigma, noise_seed, predictor):
    started = time.perf_counter()
    collected = {key: [] for key in METRICS}
    rows = []
    for batch in loader:
        clean = batch["clean"]
        noisy = fixed_evaluation_noise(clean, batch["image_id"], sigma, seed=noise_seed)
        prediction = predictor(noisy, sigma)
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().float()
        values = per_image_metrics(clean, prediction)
        values["GradientMAE"] = per_image_gradient_error(clean, prediction)
        for key in METRICS:
            collected[key].append(values[key])
        for index, (image_id, label) in enumerate(zip(batch["image_id"], batch["label"])):
            rows.append({"sigma": sigma, "image_id": int(image_id), "label": int(label),
                         **{key: float(values[key][index]) for key in METRICS}})
    if not rows:
        raise ValueError("Empty evaluation dataset")
    summary = summarize_metrics({key: np.concatenate(parts) for key, parts in collected.items()})
    classes = []
    for label in range(10):
        selected = [row for row in rows if row["label"] == label]
        if selected:
            classes.append({"sigma": sigma, "label": label, "n_images": len(selected),
                            **{key: float(np.mean([row[key] for row in selected])) for key in METRICS}})
    return summary, rows, classes, time.perf_counter() - started


@torch.no_grad()
def benchmark_model(model, loader, device):
    batch = next(iter(loader))["clean"].to(device)
    sigma = 0.3
    model.eval()
    for _ in range(10):
        predict_clean(model, batch, sigma)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(50):
        predict_clean(model, batch, sigma)
    synchronize(device)
    return 1000 * (time.perf_counter() - started) / (50 * len(batch)), len(batch)


def evaluation_protocol(root, split, device):
    names = ["evaluate.py", "references.py", "metrics.py", "noise.py", "models.py", "losses.py", "data_pipeline.py"]
    return {"split": split, "sigmas": list(SIGMAS), "noise_seed": 19300 if split == "test" else 9300,
            "split_sha256": file_hash(root / "data/processed/split_v1.npz"),
            "source_sha256": {name: file_hash(root / "src" / name) for name in names},
            "device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
            "ssim": {"data_range": 1, "win_size": 7, "gaussian_weights": False},
            "psnr_reduction": "per_image_then_average"}


def save_method(output, identity, protocol, loader, predictor):
    expected = {"identity": identity, "protocol": protocol}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["request"] != expected:
            raise ValueError(f"Evaluation provenance changed: use a new --output-dir: {output}")
        for name, digest in manifest["artifact_sha256"].items():
            if not (output / name).exists() or file_hash(output / name) != digest:
                raise ValueError(f"Evaluation artifact missing or modified: {output / name}")
        print(f"  Evaluation already complete: {identity['variant']}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    summaries, all_classes = [], []
    partial = output / "per_image.csv.gz.tmp"
    with gzip.open(partial, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        for sigma in SIGMAS:
            values, rows, classes, seconds = score_method(loader, sigma, protocol["noise_seed"], predictor)
            writer.writerows(rows)
            summaries.append({**identity, "split": protocol["split"], "sigma": sigma,
                              "n_images": len(rows), **values, "evaluation_seconds": seconds})
            all_classes.extend(classes)
            print(f"  sigma={sigma:.1f} | PSNR={values['PSNR']:.3f} | SSIM={values['SSIM']:.4f}", flush=True)
    partial.replace(output / "per_image.csv.gz")
    write_csv(output / "summary.csv", list(summaries[0]), summaries)
    write_csv(output / "class_metrics.csv", ["sigma", "label", "n_images", *METRICS], all_classes)
    atomic_json(manifest_path, {"request": expected, "artifact_sha256": {
        name: file_hash(output / name) for name in ("summary.csv", "class_metrics.csv", "per_image.csv.gz")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    jobs = discover_jobs(work, args.split)
    sources = source_hashes(root)
    split_hash = file_hash(root / "data/processed/split_v1.npz")
    selection = verified_selection(work, sources, split_hash)
    for job in jobs:
        run = work / "runs" / job["job_id"]
        saved_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        expected = resolve_config(job, saved_config["device"], selection,
                                  file_hash(work / "gradient_selection.json") if selection else None)
        if not completed_run_matches(run, expected, sources, split_hash):
            raise ValueError(f"Unfinished or inconsistent training run: {run}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    verify_raw_integrity(root)
    _, validation = load_training_partitions(root)
    val_loader = DataLoader(validation, batch_size=128, shuffle=False, num_workers=0)
    reference_selection = tune_gaussian(root, work, val_loader)
    dataset = load_test_partition(root) if args.split == "test" else validation
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    output = args.output_dir or work / "evaluation" / args.split
    if not output.is_absolute():
        output = root / output
    protocol = evaluation_protocol(root, args.split, device)
    for name in ("noisy_input", "gaussian_filter"):
        predict = ((lambda noisy, sigma: noisy) if name == "noisy_input" else
                   (lambda noisy, sigma: gaussian_denoise(noisy, reference_selection["width_by_sigma"][str(sigma)])))
        identity = {"variant": name, "job_id": name, "stage": "reference", "seed": "",
                    "parameters": 0, "checkpoint_sha256": "", "network_ms_per_image": "",
                    "benchmark_batch_size": "", "filter_selection_sha256": file_hash(work / "gaussian_selection.json")}
        save_method(output / name, identity, protocol, loader, predict)
    for number, job in enumerate(jobs, 1):
        run = work / "runs" / job["job_id"]
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if (state["epoch"] != 20 or state["config"]["purpose"] != "formal"
                or state["manifest"]["source_sha256"] != sources
                or state["manifest"]["split_sha256"] != protocol["split_sha256"]):
            raise ValueError(f"Run differs from frozen final-epoch protocol: {run}")
        model = build_model(state["config"]["model"]).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        milliseconds, batch_size = benchmark_model(model, loader, device)
        identity = {"variant": job["variant"], "job_id": job["job_id"], "stage": job["stage"],
                    "seed": job["seed"], "parameters": sum(p.numel() for p in model.parameters()),
                    "checkpoint_sha256": file_hash(run / "last.pt"),
                    "network_ms_per_image": milliseconds, "benchmark_batch_size": batch_size,
                    "filter_selection_sha256": "",
                    "gradient_weight": state["config"].get("loss", {}).get("gradient_weight", 0.0),
                    "residual_prediction": state["config"]["model"].get("residual", False),
                    "known_sigma_input": state["config"]["model"]["type"] == "cae_conditioned",
                    "training_noise": state["config"]["training"]["noise_mode"]}
        # Timing is measured once and retained on a repeated evaluation request.
        existing = output / job["job_id"] / "manifest.json"
        if existing.exists():
            old_identity = json.loads(existing.read_text(encoding="utf-8"))["request"]["identity"]
            identity["network_ms_per_image"] = old_identity["network_ms_per_image"]
        print(f"\nEvaluating [{number}/{len(jobs)}] {job['job_id']}", flush=True)
        save_method(output / job["job_id"], identity, protocol, loader,
                    lambda noisy, sigma, model=model: predict_clean(model, noisy.to(device), sigma).clamp(0, 1))
    atomic_json(output / "completion.json", {"split": args.split, "formal_runs_evaluated": len(jobs),
        "expected_formal_runs": 57, "reference_methods": 2, "protocol": protocol,
        "note": "Validation is for diagnostics/selection; paper test results require all 57 runs."})
    print(f"\nEVALUATION COMPLETED: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
