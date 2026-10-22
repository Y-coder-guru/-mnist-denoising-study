"""Offline tests of epoch-boundary recovery and dataset-level aggregation."""

import copy
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import per_image_metrics, summarize_metrics
from models import build_model
from training import (atomic_checkpoint, cpu_model_state, make_checkpoint,
                      restore_checkpoint, train_epoch, validate_config, validate_epoch)


class SyntheticImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"clean": self.images[index], "label": index % 10, "image_id": 500 + index}


class ZeroPrediction(torch.nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_uneven_validation_batches_preserve_per_image_average(self):
        images = torch.tensor([0.1, 0.2, 0.4]).reshape(3, 1, 1, 1).expand(3, 1, 28, 28)
        dataset = SyntheticImages(images)
        expected = summarize_metrics(per_image_metrics(images, torch.zeros_like(images)))
        for batch_size in (2, 3):
            actual, rows, _ = validate_epoch(ZeroPrediction(), DataLoader(dataset, batch_size=batch_size),
                                              0.0, 9300, torch.device("cpu"))
            self.assertEqual(len(rows), 3)
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        base = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(dir=base)
        directory = Path(temporary.name)
        self.assertEqual(directory.resolve().parent, base)
        self.addCleanup(temporary.cleanup)
        images = torch.rand(7, 1, 28, 28, generator=torch.Generator().manual_seed(999))
        dataset = SyntheticImages(images)
        torch.manual_seed(123)
        initial_model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Sigmoid())
        initial = cpu_model_state(initial_model)

        def setup():
            model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Sigmoid())
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            shuffle = torch.Generator().manual_seed(42)
            noise = torch.Generator().manual_seed(100042)
            loader = DataLoader(dataset, batch_size=3, shuffle=True, generator=shuffle, num_workers=0)
            return model, optimizer, shuffle, noise, loader

        settings = {"noise_mode": "mixed", "sigma_low": 0.1, "sigma_high": 0.5}
        device = torch.device("cpu")
        model, optimizer, shuffle, noise, loader = setup()
        train_epoch(model, loader, optimizer, settings, device, noise)
        state = make_checkpoint(model, optimizer, shuffle, noise, epoch=1, history=[],
            best_state=cpu_model_state(model), best_epoch=1, best_mse=0.1,
            config={"purpose": "smoke"}, manifest={}, validation_rows=[])
        path = directory / "last.pt"
        atomic_checkpoint(path, state)
        uninterrupted_loss, _ = train_epoch(model, loader, optimizer, settings, device, noise)
        expected = cpu_model_state(model)

        resumed, resumed_optimizer, resumed_shuffle, resumed_noise, resumed_loader = setup()
        loaded = torch.load(path, weights_only=True, map_location="cpu")
        restore_checkpoint(loaded, resumed, resumed_optimizer, resumed_shuffle, resumed_noise)
        resumed_loss, _ = train_epoch(resumed, resumed_loader, resumed_optimizer, settings, device, resumed_noise)
        self.assertEqual(uninterrupted_loss, resumed_loss)
        for key, value in resumed.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)
        self.assertTrue(torch.equal(noise.get_state(), resumed_noise.get_state()))
        self.assertTrue(torch.equal(shuffle.get_state(), resumed_shuffle.get_state()))

    def test_formal_runs_reject_debug_subset_limits(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "configs" / "fc_smoke.yaml").read_text(encoding="utf-8"))
        validate_config(config)
        wrong = copy.deepcopy(config)
        wrong["purpose"] = "formal"
        with self.assertRaises(ValueError):
            validate_config(wrong)
        for name in ("fc_baseline_s03_z16.yaml", "fc_baseline_s05_z16.yaml"):
            validate_config(yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8")))

    def test_convolutional_configs_and_nonpositive_width(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("cae_smoke.yaml", "cae_baseline_s03.yaml", "cae_baseline_s05.yaml"):
            config = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
            validate_config(config)
        invalid = copy.deepcopy(config)
        invalid["model"]["base_channels"] = 0
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_extension_configs_and_conditioned_training_path(self):
        root = Path(__file__).resolve().parents[1]
        names = ["fc_mixed.yaml", "cae_mixed.yaml", "cae_conditioned_mixed.yaml",
                 "fc_mixed_smoke.yaml", "cae_mixed_smoke.yaml", "cae_conditioned_smoke.yaml"]
        for name in names:
            validate_config(yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8")))
        model = build_model({"type": "cae_conditioned", "base_channels": 4})
        images = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(567))
        loader = DataLoader(SyntheticImages(images), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        settings = {"noise_mode": "mixed", "sigma_low": 0.1, "sigma_high": 0.5}
        loss, _ = train_epoch(model, loader, optimizer, settings, torch.device("cpu"),
                              torch.Generator().manual_seed(42))
        self.assertTrue(np.isfinite(loss))
        scores, rows, _ = validate_epoch(model, loader, 0.3, 9300, torch.device("cpu"))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(np.isfinite(value) for value in scores.values()))

    def test_extension_runner_stops_after_failed_child(self):
        import run_extension_checks
        with patch("run_extension_checks.subprocess.run", return_value=Mock(returncode=7)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_extension_checks.main(), 7)
        self.assertEqual(run.call_count, 1)

    def test_extension_runner_processes_three_smoke_configs(self):
        import run_extension_checks
        with patch("run_extension_checks.subprocess.run", return_value=Mock(returncode=0)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_extension_checks.main(), 0)
        self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
