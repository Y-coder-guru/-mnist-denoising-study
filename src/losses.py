"""Pixel MSE and a horizontal/vertical finite-difference L1 constraint."""

import math

import torch
import torch.nn.functional as F


def validate_loss(config):
    kind = config.get("type", "mse")
    weight = config.get("gradient_weight", 0.0)
    if kind not in ("mse", "mse_gradient"):
        raise ValueError("loss.type must be mse or mse_gradient")
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("gradient_weight must be a finite nonnegative number")
    if (kind == "mse" and weight != 0) or (kind == "mse_gradient" and weight <= 0):
        raise ValueError("MSE needs zero gradient weight; mse_gradient needs positive weight")


def gradient_l1(prediction, clean):
    if prediction.shape != clean.shape or prediction.ndim != 4 or min(prediction.shape[-2:]) < 2:
        raise ValueError("Expected matching NCHW images with height and width >= 2")
    dx_pred = prediction[..., :, 1:] - prediction[..., :, :-1]
    dx_clean = clean[..., :, 1:] - clean[..., :, :-1]
    dy_pred = prediction[..., 1:, :] - prediction[..., :-1, :]
    dy_clean = clean[..., 1:, :] - clean[..., :-1, :]
    return 0.5 * (F.l1_loss(dx_pred, dx_clean) + F.l1_loss(dy_pred, dy_clean))


def reconstruction_loss(prediction, clean, config=None):
    config = config or {"type": "mse", "gradient_weight": 0.0}
    mse = F.mse_loss(prediction, clean)
    edge = gradient_l1(prediction, clean) if config["type"] == "mse_gradient" else mse.new_zeros(())
    total = mse + config.get("gradient_weight", 0.0) * edge
    return total, mse, edge
