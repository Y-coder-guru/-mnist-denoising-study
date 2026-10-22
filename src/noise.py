"""Dynamic training noise and image-indexed, batch-independent evaluation noise."""

from __future__ import annotations

import hashlib

import torch


def _check_clean(clean: torch.Tensor):
    if clean.ndim != 4 or tuple(clean.shape[1:]) != (1, 28, 28) or len(clean) == 0:
        raise ValueError("Expected a nonempty batch shaped (N, 1, 28, 28)")
    if not clean.is_floating_point() or not torch.isfinite(clean).all():
        raise ValueError("Clean images must contain finite floating-point values")
    if clean.min() < 0 or clean.max() > 1:
        raise ValueError("Clean image values must be in [0, 1]")


def _sigma_values(sigma, clean: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(sigma, dtype=clean.dtype, device=clean.device)
    if values.ndim == 0:
        values = values.expand(len(clean))
    if values.shape != (len(clean),) or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("sigma must be a nonnegative finite scalar or a length-N vector")
    return values.reshape(-1, 1, 1, 1)


def corrupt_training(clean: torch.Tensor, sigma, *, generator=None) -> torch.Tensor:
    _check_clean(clean)
    scales = _sigma_values(sigma, clean)
    # A supplied generator must belong to the same device as clean.
    standard = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device,
                           generator=generator)
    return (clean + scales * standard).clamp(0, 1)


def sample_mixed_sigmas(batch_size: int, *, low=0.1, high=0.5,
                        generator=None, device="cpu") -> torch.Tensor:
    if batch_size < 1 or not 0 <= low < high:
        raise ValueError("Require batch_size > 0 and 0 <= low < high")
    return low + (high - low) * torch.rand(batch_size, generator=generator, device=device)


def fixed_evaluation_noise(clean: torch.Tensor, image_ids, sigma: float,
                           *, seed: int) -> torch.Tensor:
    _check_clean(clean)
    scales = _sigma_values(sigma, clean)
    ids = torch.as_tensor(image_ids, device="cpu")
    if ids.shape != (len(clean),) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("image_ids must contain one integer ID per image")
    if (ids < 0).any() or not isinstance(seed, int) or seed < 0:
        raise ValueError("Image IDs and seed must be nonnegative integers")
    draws = []
    for image_id in ids.tolist():
        # Hashing separates the evaluation seed and image ID, avoiding additive collisions.
        key = f"mnist-noise-v1:{seed}:{image_id}".encode("ascii")
        per_image_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        generator = torch.Generator(device="cpu").manual_seed(per_image_seed & ((1 << 63) - 1))
        draws.append(torch.randn((1, 28, 28), generator=generator, dtype=torch.float32))
    # Draw on CPU for both CPU/GPU evaluation; sigma is intentionally absent from the seed.
    standard = torch.stack(draws).to(device=clean.device, dtype=clean.dtype)
    return (clean + scales * standard).clamp(0, 1)
