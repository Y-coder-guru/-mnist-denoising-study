"""Noisy input and validation-tuned Gaussian filtering reference methods."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from experiments import SIGMAS, file_hash
from noise import fixed_evaluation_noise
from training import atomic_json


BLUR_WIDTHS = (0.0, 0.3, 0.5, 0.7, 1.0, 1.3)


def gaussian_denoise(noisy, width):
    if width < 0 or not np.isfinite(width):
        raise ValueError("Invalid filter width")
    values = noisy.detach().cpu().numpy() if isinstance(noisy, torch.Tensor) else np.asarray(noisy)
    values = values.astype(np.float32, copy=False)
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("Expected N1HW grayscale batch")
    # Do not blur across the batch/channel axes.
    result = values.copy() if width == 0 else gaussian_filter(values, sigma=(0, 0, width, width), mode="reflect")
    return result.clip(0, 1)


def select_filter_width(scores):
    return min(scores, key=lambda row: (row["MSE"], row["width"]))["width"]


def tune_gaussian(root, work_dir, validation_loader):
    path = work_dir / "gaussian_selection.json"
    protocol = {name: file_hash(root / "src" / name)
                for name in ("references.py", "metrics.py", "noise.py")}
    split_hash = file_hash(root / "data/processed/split_v1.npz")
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["source_sha256"] != protocol or result["split_sha256"] != split_hash:
            raise ValueError("Gaussian selection has different source/data provenance")
        return result
    chosen, candidates = {}, {}
    for sigma in SIGMAS:
        totals = {width: 0.0 for width in BLUR_WIDTHS}
        count = 0
        for batch in validation_loader:
            clean = batch["clean"]
            noisy = fixed_evaluation_noise(clean, batch["image_id"], sigma, seed=9300)
            reference = clean.numpy().astype(np.float64)
            for width in BLUR_WIDTHS:
                predicted = gaussian_denoise(noisy, width).astype(np.float64)
                totals[width] += float(np.square(reference - predicted).mean(axis=(1, 2, 3)).sum())
            count += len(clean)
        if count != 5000:
            raise ValueError("Gaussian selection must use all 5000 validation images")
        scores = [{"width": width, "MSE": total / count} for width, total in totals.items()]
        chosen[str(sigma)] = select_filter_width(scores)
        candidates[str(sigma)] = scores
    result = {"width_by_sigma": chosen, "candidates": candidates,
              "rule": "Minimum validation MSE per known input noise strength; ties use smaller width.",
              "known_noise_strength": True, "test_set_used": False, "validation_noise_seed": 9300,
              "validation_count": 5000, "source_sha256": protocol, "split_sha256": split_hash}
    atomic_json(path, result)
    return result
