"""Local train/validation integration check, not a formal experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import load_training_partitions
from metrics import per_image_metrics, summarize_metrics
from models import FullyConnectedDAE
from noise import corrupt_training, fixed_evaluation_noise, sample_mixed_sigmas


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    torch.set_num_threads(2)
    torch.manual_seed(42)

    print("[1/5] Reading fixed train/validation partitions; no test evaluation...", flush=True)
    train, validation = load_training_partitions(root)
    train_batch = next(iter(DataLoader(train, batch_size=32, shuffle=True, num_workers=0,
                                      generator=torch.Generator().manual_seed(42))))
    val_batch = next(iter(DataLoader(validation, batch_size=32, shuffle=False, num_workers=0)))
    clean, ids = val_batch["clean"], val_batch["image_id"]

    print("[2/5] Checking fixed, dynamic, and mixed-strength noise...", flush=True)
    fixed = fixed_evaluation_noise(clean, ids, 0.3, seed=9300)
    repeated = fixed_evaluation_noise(clean, ids, 0.3, seed=9300)
    chunked = torch.cat([fixed_evaluation_noise(clean[:13], ids[:13], 0.3, seed=9300),
                         fixed_evaluation_noise(clean[13:], ids[13:], 0.3, seed=9300)])
    order = torch.arange(len(clean) - 1, -1, -1)
    reordered = fixed_evaluation_noise(clean[order], ids[order], 0.3, seed=9300)
    require(torch.equal(fixed, repeated), "Fixed noise changes on repetition")
    require(torch.equal(fixed, chunked), "Fixed noise depends on batch size")
    require(torch.equal(fixed[order], reordered), "Fixed noise depends on sample order")
    require(torch.equal(fixed_evaluation_noise(clean, ids, 0.0, seed=9300), clean), "Zero-noise identity failed")
    generator = torch.Generator().manual_seed(777)
    dynamic_a = corrupt_training(clean, 0.3, generator=generator)
    dynamic_b = corrupt_training(clean, 0.3, generator=generator)
    require(not torch.equal(dynamic_a, dynamic_b), "Training noise did not change")
    mixed = corrupt_training(clean, sample_mixed_sigmas(len(clean), generator=generator), generator=generator)
    require(mixed.shape == clean.shape, "Mixed noise changed image shape")

    print("[3/5] Checking MSE, PSNR, and SSIM against reference cases...", flush=True)
    reference = np.zeros((2, 28, 28))
    estimates = np.stack([np.full((28, 28), 0.1), np.full((28, 28), 0.2)])
    known = per_image_metrics(reference, estimates)
    require(np.allclose(known["MSE"], [0.01, 0.04], atol=1e-12), "MSE reference check failed")
    require(np.allclose(known["PSNR"], [20.0, 13.979400086720375], atol=1e-10), "PSNR reference check failed")
    identity = per_image_metrics(clean[:1], clean[:1])
    require(identity["MSE"][0] == 0 and np.isposinf(identity["PSNR"][0])
            and np.isclose(identity["SSIM"][0], 1), "Identity metric check failed")

    print("[4/5] Checking model shapes and 3 debug CPU parameter updates...", flush=True)
    parameter_counts = {}
    for latent_dim in (8, 16, 32, 64):
        candidate = FullyConnectedDAE(latent_dim)
        with torch.no_grad():
            prediction = candidate(train_batch["clean"])
        require(prediction.shape == train_batch["clean"].shape, "Output shape mismatch")
        require(torch.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1,
                "Model output must be finite and in [0, 1]")
        parameter_counts[str(latent_dim)] = sum(p.numel() for p in candidate.parameters())
    torch.manual_seed(42)
    model = FullyConnectedDAE(16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = [p.detach().clone() for p in model.parameters()]
    generator = torch.Generator().manual_seed(42)
    losses = []
    for _ in range(3):
        noisy = corrupt_training(train_batch["clean"], 0.3, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(noisy), train_batch["clean"])
        require(torch.isfinite(loss), "Non-finite loss")
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                "Missing/non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
    require(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())),
            "Optimizer did not update parameters")

    print("[5/5] Saving debug report; no formal model checkpoint...", flush=True)
    names = ["data_pipeline.py", "noise.py", "metrics.py", "models.py", "check_pipeline.py"]
    report = {
        "status": "pipeline_checks_passed", "purpose": "Debug only; not formal paper results.",
        "device": "cpu", "seed": 42, "batch_size": 32, "optimizer_steps": 3,
        "train_size": len(train), "validation_size": len(validation), "test_set_evaluated": False,
        "checks": {key: True for key in ["frozen_split", "fixed_noise_repeatability",
            "batch_independence", "order_independence", "zero_noise_identity", "dynamic_noise",
            "mixed_noise", "analytical_metrics", "model_shapes", "finite_gradients", "parameter_updates"]},
        "noise_protocol": "mnist-noise-v1",
        "parameter_counts_by_bottleneck": parameter_counts,
        "debug_losses_not_for_paper": losses,
        "validation_noisy_input_debug_metrics": summarize_metrics(per_image_metrics(clean, fixed)),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() for name in names},
    }
    output = root / "outputs" / "pipeline_check" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nPIPELINE CHECK PASSED")
    print(f"Report: {output}")
    print(f"FC latent=16 parameters: {parameter_counts['16']}")
    print("3 debug optimizer steps only; no test evaluation or saved model weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
