"""Explicit grayscale MSE/PSNR/SSIM protocol, with per-image aggregation."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _images_array(images) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float64)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    elif images.ndim == 2:
        images = images[None]
    if images.ndim != 3 or len(images) == 0 or min(images.shape[1:]) < 7:
        raise ValueError("Expected nonempty grayscale images with height/width at least 7")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Metrics require finite pixel values in [0, 1]; clip predictions explicitly before evaluation")
    return images


def per_image_metrics(clean, prediction) -> dict[str, np.ndarray]:
    reference = _images_array(clean)
    restored = _images_array(prediction)
    if reference.shape != restored.shape:
        raise ValueError("Clean and predicted images must have the same shape")
    mse = np.mean(np.square(reference - restored), axis=(1, 2))
    psnr = np.full(mse.shape, np.inf, dtype=np.float64)
    nonzero = mse > 0
    psnr[nonzero] = -10.0 * np.log10(mse[nonzero])  # data_range = 1
    ssim = np.array([
        structural_similarity(x, y, data_range=1.0, win_size=7,
                              gaussian_weights=False, use_sample_covariance=True,
                              K1=0.01, K2=0.03)
        for x, y in zip(reference, restored)
    ], dtype=np.float64)
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


def summarize_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    # For a full dataset, concatenate per-image arrays first; do not average batch means.
    return {key: float(np.mean(value)) for key, value in values.items()}


def per_image_gradient_error(clean, prediction):
    reference, restored = _images_array(clean), _images_array(prediction)
    if reference.shape != restored.shape:
        raise ValueError("Image shapes differ")
    error = restored - reference
    return 0.5 * (np.abs(np.diff(error, axis=2)).mean(axis=(1, 2))
                  + np.abs(np.diff(error, axis=1)).mean(axis=(1, 2)))
