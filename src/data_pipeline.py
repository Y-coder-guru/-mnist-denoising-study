"""Read the frozen MNIST partitions; never download or reshuffle their membership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def verify_raw_integrity(root: Path):
    report = json.loads((root / "outputs/data_check/data_report.json").read_text(encoding="utf-8"))
    expected_names = {name.removesuffix(".gz") for name, _ in MNIST.resources}
    if set(report["raw_file_sha256"]) != expected_names:
        raise ValueError("Unexpected raw-file manifest")
    for name, expected in report["raw_file_sha256"].items():
        path = root / "data/MNIST/raw" / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Raw MNIST file differs from the verified data report: {name}")


def load_saved_split(root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = root / "data" / "processed" / "split_v1.npz"
    report_path = root / "outputs" / "data_check" / "data_report.json"
    if not path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Run python src/prepare_data.py first and retain its split and report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["split"]["sha256"]:
        raise ValueError("Saved split differs from the verified data report; stop and inspect it.")
    with np.load(path, allow_pickle=False) as saved:
        train = saved["train_indices"].copy()
        validation = saved["validation_indices"].copy()
        if saved["split_seed"].item() != 2026:
            raise ValueError("Unexpected split seed")
    if train.dtype != np.int64 or validation.dtype != np.int64:
        raise ValueError("Partition indices must be int64")
    if train.shape != (55000,) or validation.shape != (5000,):
        raise ValueError("Unexpected partition sizes")
    if not np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(60000)):
        raise ValueError("Partitions overlap, omit indices, or contain invalid indices")
    # Reuse the serialized, hashed order rather than regenerating it in another
    # NumPy version on Kaggle. The split report binds its exact bytes.
    return train, validation


class MNISTPartition(Dataset):
    """Labels are diagnostic metadata; the denoiser's target is always clean pixels."""

    def __init__(self, source: MNIST, indices: np.ndarray, partition: str):
        self.source = source
        self.indices = indices.copy()
        self.partition = partition

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image_id = int(self.indices[position])
        return {
            "clean": self.source.data[image_id].unsqueeze(0).to(torch.float32).div(255.0),
            "label": int(self.source.targets[image_id]),
            "image_id": image_id,
        }


def load_training_partitions(root: Path) -> tuple[MNISTPartition, MNISTPartition]:
    train_indices, validation_indices = load_saved_split(root)
    source = MNIST(root=str(root / "data"), train=True, download=False)
    if len(source) != 60000:
        raise ValueError("Unexpected official training set size")
    return (MNISTPartition(source, train_indices, "train"),
            MNISTPartition(source, validation_indices, "validation"))


def load_test_partition(root: Path) -> MNISTPartition:
    # Called by final evaluation, not by model-selection or pipeline-check code.
    load_saved_split(root)
    source = MNIST(root=str(root / "data"), train=False, download=False)
    if len(source) != 10000:
        raise ValueError("Unexpected official test set size")
    return MNISTPartition(source, np.arange(10000, dtype=np.int64), "test")
