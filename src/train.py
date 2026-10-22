"""Train a baseline from YAML; create unique runs or resume a saved epoch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data_pipeline import load_training_partitions
from models import build_model, model_description
from run_figures import plot_history, plot_validation_preview
from training import (LOG_FIELDS, atomic_checkpoint, atomic_json, cpu_model_state,
                      make_checkpoint, restore_checkpoint, train_epoch,
                      validate_config, validate_epoch, write_csv)


SOURCES = ["data_pipeline.py", "noise.py", "metrics.py", "models.py", "training.py",
           "losses.py", "run_figures.py", "train.py"]


def source_hashes(root):
    return {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() for name in SOURCES}


def runtime_info(device):
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "scikit-image", "scipy", "matplotlib", "PyYAML")}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "device": str(device), "deterministic_algorithms": True,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def selected_indices(partition, limit):
    # Frozen split order is already randomized. Limits select a fixed debug prefix.
    return partition.indices if limit is None else partition.indices[:limit]


def export_records(run_dir, state):
    # last.pt is the epoch commit. Rebuild exports from it after an interrupted write.
    write_csv(run_dir / "metrics.csv", LOG_FIELDS, state["history"])
    write_csv(run_dir / "validation_per_image.csv", ["image_id", "label", "MSE", "PSNR", "SSIM"],
              state["validation_rows"])
    if state["best_epoch"] is not None:
        atomic_checkpoint(run_dir / "best.pt", {
            "schema_version": 1, "role": "validation_best_for_analysis",
            "epoch": state["best_epoch"], "config": state["config"],
            "model_state": state["best_model_state"],
            "validation_mse": state["best_validation_mse"],
            "source_sha256": state["manifest"]["source_sha256"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--resume", type=Path, help="Existing run directory containing last.pt")
    parser.add_argument("--run-dir", type=Path, help="Explicit new run directory, for the batch queue")
    args = parser.parse_args()
    if args.resume and args.run_dir:
        parser.error("--run-dir is for a new run only")
    root = args.root.resolve()
    if args.resume:
        run_dir = args.resume if args.resume.is_absolute() else root / args.resume
        run_dir = run_dir.resolve()
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        config = state["config"]
    else:
        path = args.config if args.config.is_absolute() else root / args.config
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        state = None
    validate_config(config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", config["name"]):
        raise ValueError("Use a short ASCII name with letters, digits, underscores or hyphens")

    # Configure determinism before initializing CUDA or creating any training tensors.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested = config["device"]
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    environment = runtime_info(device)

    train_data, validation_data = load_training_partitions(root)
    train_limit = config["data"]["train_limit"]
    val_limit = config["data"]["validation_limit"]
    train_ids = selected_indices(train_data, train_limit)
    val_ids = selected_indices(validation_data, val_limit)
    train_subset = train_data if train_limit is None else Subset(train_data, range(train_limit))
    val_subset = validation_data if val_limit is None else Subset(validation_data, range(val_limit))
    split_path = root / "data" / "processed" / "split_v1.npz"
    current_sources = source_hashes(root)
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()

    if state is not None:
        manifest = state["manifest"]
        if state["schema_version"] != 1 or manifest["source_sha256"] != current_sources:
            raise ValueError("Run source version differs; resume requires the recorded code")
        if manifest["split_sha256"] != split_hash:
            raise ValueError("Run data split differs")
        previous_environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        if (previous_environment["packages"] != environment["packages"]
                or previous_environment["device"] != environment["device"]):
            raise ValueError("Package versions/device differ from this run; restore the original environment to resume")
        with np.load(run_dir / "sample_indices.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["train"], train_ids) or not np.array_equal(saved["validation"], val_ids):
                raise ValueError("Saved run sample indices differ")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = ((args.run_dir if args.run_dir.is_absolute() else root / args.run_dir)
                   if args.run_dir else root / "outputs" / "runs" / f"{config['name']}_seed{config['seed']}_{timestamp}")
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": current_sources, "split_sha256": split_hash,
                    "noise_protocol": "mnist-noise-v1", "test_set_used": False,
                    "primary_checkpoint": "last.pt", "purpose": config["purpose"]}
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "environment.json", environment)
        atomic_json(run_dir / "manifest.json", manifest)
        np.savez_compressed(run_dir / "sample_indices.npz", train=train_ids, validation=val_ids)
        snapshot = run_dir / "source_snapshot"
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copy2(root / "src" / name, snapshot / name)
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60)
        if frozen.returncode:
            raise RuntimeError(f"Could not record dependencies: {frozen.stderr}")
        (run_dir / "pip_freeze.txt").write_text(frozen.stdout, encoding="utf-8")

    loader_generator = torch.Generator().manual_seed(config["seed"])
    # Separate RNG streams prevent model initialization or plotting from changing noise/shuffling.
    noise_generator = torch.Generator(device=device).manual_seed(config["seed"] + 100000)
    validation_generator = torch.Generator().manual_seed(config["seed"] + 200000)
    loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True,
                        num_workers=0, generator=loader_generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"], shuffle=False,
                            num_workers=0, generator=validation_generator)
    model = build_model(config["model"]).to(device)
    if state is None:
        atomic_json(run_dir / "model.json", model_description(model))
        (run_dir / "model.txt").write_text(str(model) + "\n", encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    if state is None:
        state = make_checkpoint(model, optimizer, loader_generator, noise_generator,
                                epoch=0, history=[], best_state=cpu_model_state(model), best_epoch=None,
                                best_mse=None, config=config, manifest=manifest, validation_rows=[])
        atomic_checkpoint(run_dir / "last.pt", state)
    else:
        restore_checkpoint(state, model, optimizer, loader_generator, noise_generator)
    export_records(run_dir, state)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Purpose: {config['purpose']} | Device: {device} | Train: {len(train_subset)} | Validation: {len(val_subset)}", flush=True)
    if config["purpose"] == "smoke":
        print("LOCAL SMOKE TEST: results are excluded from the paper.", flush=True)
    elif config["purpose"] == "pilot":
        print("VALIDATION SELECTION RUN: no test-set evaluation.", flush=True)

    epochs = config["training"]["epochs"]
    committed_epoch = state["epoch"]
    try:
        atomic_json(run_dir / "status.json", {"status": "running", "last_completed_epoch": committed_epoch})
        for epoch in range(committed_epoch + 1, epochs + 1):
            training_values, training_seconds = train_epoch(model, loader, optimizer, config["training"],
                device, noise_generator, loss_config=config.get("loss"), return_components=True)
            training_mse = training_values["mse"]
            scores, rows, validation_seconds = validate_epoch(model, val_loader,
                config["validation"]["sigma"], config["validation"]["noise_seed"], device)
            history = state["history"] + [{"epoch": epoch, "train_loss": training_values["loss"],
                "train_mse": training_mse, "train_gradient_l1": training_values["gradient_l1"],
                "val_mse": scores["MSE"], "val_psnr": scores["PSNR"], "val_ssim": scores["SSIM"],
                "train_seconds": training_seconds, "validation_seconds": validation_seconds}]
            improved = state["best_validation_mse"] is None or scores["MSE"] < state["best_validation_mse"]
            state = make_checkpoint(model, optimizer, loader_generator, noise_generator,
                epoch=epoch, history=history,
                best_state=cpu_model_state(model) if improved else state["best_model_state"],
                best_epoch=epoch if improved else state["best_epoch"],
                best_mse=scores["MSE"] if improved else state["best_validation_mse"],
                config=config, manifest=manifest, validation_rows=rows)
            atomic_checkpoint(run_dir / "last.pt", state)
            committed_epoch = epoch
            export_records(run_dir, state)
            atomic_json(run_dir / "status.json", {"status": "running", "last_completed_epoch": epoch})
            print(f"Epoch {epoch}/{epochs} | train MSE={training_mse:.6f} | val MSE={scores['MSE']:.6f} | "
                  f"PSNR={scores['PSNR']:.3f} dB | SSIM={scores['SSIM']:.4f}", flush=True)

        plot_history(state["history"], run_dir / "curves.png", config["purpose"])
        plot_validation_preview(model, val_loader, config["validation"]["sigma"],
                                config["validation"]["noise_seed"], device,
                                run_dir / "validation_preview.png", config["purpose"])
        atomic_json(run_dir / "summary.json", {
            "purpose": config["purpose"], "completed_epochs": committed_epoch,
            "train_size": len(train_subset), "validation_size": len(val_subset),
            "test_set_evaluated": False, "parameters": sum(p.numel() for p in model.parameters()),
            "primary_checkpoint": "last.pt", "best_epoch_for_analysis": state["best_epoch"],
            "last_epoch": state["history"][-1],
            "note": "Smoke metrics are debug only. Validation metrics are not final test results.",
        })
        atomic_json(run_dir / "status.json", {"status": "completed", "last_completed_epoch": committed_epoch})
    except KeyboardInterrupt:
        atomic_json(run_dir / "status.json", {"status": "interrupted", "last_completed_epoch": committed_epoch})
        print(f"\nStopped. Resume from the last completed epoch: python src/train.py --resume \"{run_dir}\"", flush=True)
        return 130
    except Exception as exc:
        atomic_json(run_dir / "status.json", {"status": "failed", "last_completed_epoch": committed_epoch,
                                              "error": f"{type(exc).__name__}: {exc}"})
        raise
    print("\nSMOKE RUN COMPLETED" if config["purpose"] == "smoke" else "TRAINING COMPLETED")
    print(f"Artifacts: {run_dir}")
    print("Main checkpoint: last.pt | Curves: curves.png | Preview: validation_preview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
