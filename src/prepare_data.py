"""Download and inspect MNIST. Does not train or evaluate a denoising model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import MNIST

from mnist_download import prepare_mnist_files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(dataset: MNIST, expected_size: int) -> dict:
    if tuple(dataset.data.shape) != (expected_size, 28, 28):
        raise ValueError(f"Unexpected image shape: {tuple(dataset.data.shape)}")
    if tuple(dataset.targets.shape) != (expected_size,):
        raise ValueError(f"Unexpected target shape: {tuple(dataset.targets.shape)}")
    if dataset.data.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 raw images, got {dataset.data.dtype}")
    labels, counts = torch.unique(dataset.targets, return_counts=True)
    if labels.tolist() != list(range(10)):
        raise ValueError(f"Unexpected labels: {labels.tolist()}")
    return {
        "size": len(dataset),
        "image_shape": [28, 28],
        "raw_dtype": str(dataset.data.dtype),
        "raw_min": int(dataset.data.min()),
        "raw_max": int(dataset.data.max()),
        "class_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
    }


def prepare_split(path: Path, seed: int, train_count: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(train_count)
    train_indices = permutation[:55000].astype(np.int64)
    validation_indices = permutation[55000:].astype(np.int64)
    if not np.array_equal(np.sort(np.concatenate([train_indices, validation_indices])), np.arange(train_count)):
        raise ValueError("Split must contain every training index exactly once")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            matches = (
                int(saved["split_seed"].item()) == seed
                and np.array_equal(saved["train_indices"], train_indices)
                and np.array_equal(saved["validation_indices"], validation_indices)
            )
        if not matches:
            raise FileExistsError(f"Existing split differs from requested split: {path}. Preserve it and use a new project copy for a different protocol.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train_indices=train_indices,
                            validation_indices=validation_indices,
                            split_seed=np.array(seed, dtype=np.int64))
    return train_indices, validation_indices


def save_preview(dataset: MNIST, train_indices: np.ndarray, path: Path) -> list[int]:
    # Use training examples only; choose one fixed representative per digit.
    train_labels = dataset.targets[torch.from_numpy(train_indices)].numpy()
    representatives = [int(train_indices[np.flatnonzero(train_labels == digit)[0]]) for digit in range(10)]
    clean = dataset.data[representatives].to(torch.float32).div(255.0)
    generator = torch.Generator().manual_seed(9300)
    standard_noise = torch.randn(clean.shape, generator=generator)
    rows = [clean, (clean + 0.3 * standard_noise).clamp(0, 1),
            (clean + 0.5 * standard_noise).clamp(0, 1)]
    row_labels = ["Clean", "Noise sigma=0.3", "Noise sigma=0.5"]
    figure, axes = plt.subplots(3, 10, figsize=(13, 4.6))
    for row, images in enumerate(rows):
        for digit, image in enumerate(images):
            axis = axes[row, digit]
            axis.imshow(image.numpy(), cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(str(digit))
            if digit == 0:
                axis.set_ylabel(row_labels[row], fontsize=9)
    figure.suptitle("MNIST corruption check | fixed training examples | not denoising results")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return representatives


def runtime_report() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-image", "scipy", "tqdm", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root; defaults to the folder containing src/.")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Bypass proxy settings for this script's downloads only.")
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "data"
    check_dir = root / "outputs" / "data_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading MNIST (download if missing)...", flush=True)
    downloads = prepare_mnist_files(data_dir, MNIST.resources, no_proxy=args.no_proxy)
    official_train = MNIST(root=str(data_dir), train=True, download=False)
    official_test = MNIST(root=str(data_dir), train=False, download=False)

    print("[2/5] Checking images and labels...", flush=True)
    train_info = inspect_dataset(official_train, 60000)
    test_info = inspect_dataset(official_test, 10000)

    print("[3/5] Saving/verifying the fixed 55000/5000 split...", flush=True)
    split_path = data_dir / "processed" / "split_v1.npz"
    train_indices, validation_indices = prepare_split(split_path, seed=2026)

    print("[4/5] Saving corruption preview...", flush=True)
    preview_path = check_dir / "noise_examples.png"
    representatives = save_preview(official_train, train_indices, preview_path)
    raw_hashes = {
        filename.removesuffix(".gz"): file_sha256(
            Path(official_train.raw_folder) / filename.removesuffix(".gz"))
        for filename, _ in MNIST.resources
    }
    report = {
        "status": "data_checks_passed",
        "purpose": "Data preparation only; no model trained or tested.",
        "official_training_set": train_info,
        "official_test_set": test_info,
        "split": {
            "seed": 2026, "train_size": len(train_indices),
            "validation_size": len(validation_indices), "test_size": 10000,
            "file": str(split_path.relative_to(root)),
            "sha256": file_sha256(split_path),
            "overlap": 0,
            "method": "Seeded random permutation of official training indices; labels are not training targets for the denoiser.",
        },
        "raw_file_sha256": raw_hashes,
        "verified_archives": downloads,
        "download_proxy_mode": "direct" if args.no_proxy else "system_or_environment",
        "preview_training_indices_by_digit": representatives,
        "preview_noise_seed": 9300,
        "corruption": "clean=uint8/255; noisy=clip(clean+sigma*standard_normal,0,1)",
        "noise_note": "After clipping, the effective corruption is not an unbounded Gaussian; sigma describes the pre-clipping draw.",
    }
    (check_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[5/5] Recording the local environment...", flush=True)
    (check_dir / "environment_local.json").write_text(
        json.dumps(runtime_report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if frozen.returncode:
        raise RuntimeError(f"pip freeze failed: {frozen.stderr}")
    (check_dir / "pip_freeze_local_cpu.txt").write_text(frozen.stdout, encoding="utf-8")
    print("\nDATA CHECK PASSED", flush=True)
    print("Training: 55000 | Validation: 5000 | Test: 10000")
    print(f"Preview: {preview_path}")
    print(f"Report: {check_dir / 'data_report.json'}")
    print("No training was performed. Keep the split and reports for subsequent stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
