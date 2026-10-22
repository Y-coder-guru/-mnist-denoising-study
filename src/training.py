"""Training, validation, and epoch-boundary recovery primitives."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from metrics import per_image_metrics, summarize_metrics
from losses import reconstruction_loss, validate_loss
from models import predict_clean
from noise import corrupt_training, fixed_evaluation_noise, sample_mixed_sigmas


LOG_FIELDS = ["epoch", "train_loss", "train_mse", "train_gradient_l1", "val_mse", "val_psnr", "val_ssim",
              "train_seconds", "validation_seconds"]


def validate_config(config):
    if config.get("purpose") not in ("smoke", "formal", "pilot"):
        raise ValueError("purpose must be smoke, formal, or pilot")
    validate_loss(config.get("loss", {"type": "mse", "gradient_weight": 0.0}))
    model_type = config["model"]["type"]
    if model_type not in ("fc", "cae", "cae_conditioned"):
        raise ValueError("Unsupported model type")
    if type(config["model"].get("residual", False)) is not bool:
        raise ValueError("model.residual must be a boolean")
    if model_type == "fc" and config["model"].get("residual", False):
        raise ValueError("Residual ablation is defined for the convolutional backbone")
    model_size = config["model"]["latent_dim"] if model_type == "fc" else config["model"]["base_channels"]
    integers = [config["seed"], config["cpu_threads"], model_size,
                config["training"]["epochs"], config["training"]["batch_size"],
                config["validation"]["noise_seed"]]
    if any(type(value) is not int for value in integers):
        raise ValueError("Seeds, dimensions, epochs, batch size, and threads must be integers")
    if integers[0] < 0 or integers[-1] < 0 or min(integers[1:-1]) < 1:
        raise ValueError("Invalid seed or positive-integer setting")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    training = config["training"]
    if not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if training["noise_mode"] not in ("fixed", "mixed"):
        raise ValueError("noise_mode must be fixed or mixed")
    if training["noise_mode"] == "fixed":
        if not math.isfinite(training["sigma"]) or training["sigma"] < 0:
            raise ValueError("Invalid training sigma")
    elif not (0 <= training["sigma_low"] < training["sigma_high"] <= 1):
        raise ValueError("Require 0 <= sigma_low < sigma_high <= 1")
    if not math.isfinite(config["validation"]["sigma"]) or config["validation"]["sigma"] < 0:
        raise ValueError("Invalid validation sigma")
    for key, maximum in (("train_limit", 55000), ("validation_limit", 5000)):
        value = config["data"][key]
        if value is not None and (type(value) is not int or not 1 <= value <= maximum):
            raise ValueError(f"Invalid {key}")
        if config["purpose"] in ("formal", "pilot") and value is not None:
            raise ValueError("Formal/pilot runs must use the full fixed training/validation partitions")
    if config["purpose"] == "smoke" and any(config["data"][key] is None
                                            for key in ("train_limit", "validation_limit")):
        raise ValueError("Smoke runs must explicitly limit both data partitions")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, config, device, noise_generator, *,
                loss_config=None, return_components=False):
    model.train()
    synchronize(device)
    started = time.perf_counter()
    total, total_loss, total_edge, count = 0.0, 0.0, 0.0, 0
    for batch in loader:
        clean = batch["clean"].to(device)
        if config["noise_mode"] == "fixed":
            sigma = config["sigma"]
        else:
            sigma = sample_mixed_sigmas(len(clean), low=config["sigma_low"],
                                       high=config["sigma_high"], generator=noise_generator,
                                       device=device)
        noisy = corrupt_training(clean, sigma, generator=noise_generator)
        optimizer.zero_grad(set_to_none=True)
        loss, mse, edge = reconstruction_loss(predict_clean(model, noisy, sigma), clean, loss_config)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        gradient_checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if gradient_checks and not torch.stack(gradient_checks).all():
            raise RuntimeError("Non-finite training gradient")
        optimizer.step()
        mse_value, loss_value, edge_value = torch.stack([mse.detach(), loss.detach(), edge.detach()]).cpu().tolist()
        total += mse_value * len(clean)
        total_loss += loss_value * len(clean)
        total_edge += edge_value * len(clean)
        count += len(clean)
    synchronize(device)
    if not count:
        raise ValueError("Empty training loader")
    values = {"mse": total / count, "loss": total_loss / count, "gradient_l1": total_edge / count}
    return (values if return_components else values["mse"]), time.perf_counter() - started


@torch.no_grad()
def validate_epoch(model, loader, sigma, noise_seed, device):
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    collected = {key: [] for key in ("MSE", "PSNR", "SSIM")}
    rows = []
    for batch in loader:
        clean = batch["clean"].to(device)
        noisy = fixed_evaluation_noise(clean, batch["image_id"], sigma, seed=noise_seed)
        prediction = predict_clean(model, noisy, sigma).clamp(0, 1)
        values = per_image_metrics(clean, prediction)
        for key in collected:
            collected[key].append(values[key])
        for index, (image_id, label) in enumerate(zip(batch["image_id"], batch["label"])):
            rows.append({"image_id": int(image_id), "label": int(label),
                         **{key: float(values[key][index]) for key in collected}})
    synchronize(device)
    if not rows:
        raise ValueError("Empty validation loader")
    summary = summarize_metrics({key: np.concatenate(parts) for key, parts in collected.items()})
    return summary, rows, time.perf_counter() - started


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path: Path, state):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_csv(path: Path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_checkpoint(model, optimizer, loader_generator, noise_generator, *,
                    epoch, history, best_state, best_epoch, best_mse, config, manifest,
                    validation_rows):
    return {
        "schema_version": 1, "epoch": epoch, "config": config, "manifest": manifest,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "loader_rng": loader_generator.get_state(), "noise_rng": noise_generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else [],
        "history": history, "best_model_state": best_state, "best_epoch": best_epoch,
        "best_validation_mse": best_mse, "validation_rows": validation_rows,
    }


def restore_checkpoint(state, model, optimizer, loader_generator, noise_generator):
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    loader_generator.set_state(state["loader_rng"])
    noise_generator.set_state(state["noise_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("A CUDA run must be resumed with CUDA available")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def cpu_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
