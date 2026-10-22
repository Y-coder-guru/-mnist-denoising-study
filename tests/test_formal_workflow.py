"""Synthetic-only checks of matrix, selection, losses, evaluation and orchestration."""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import discover_jobs, save_method, score_method
from experiments import (make_jobs, pilot_jobs, resolve_config, select_gradient_weight,
                         variants, verified_selection)
from losses import gradient_l1, reconstruction_loss, validate_loss
from metrics import per_image_gradient_error
from models import ConvolutionalDAE
from references import gaussian_denoise, select_filter_width
from summarize import aggregate, paired_ablation, save_ablation_plot, save_bottleneck_plot, save_noise_plot


class Images(Dataset):
    def __init__(self, n=4, offset=0):
        self.images = torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(789+offset))
        self.indices = np.arange(offset, offset+n, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {"clean": self.images[i], "label": i % 10, "image_id": int(self.indices[i])}


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(dir=base)
        self.directory = Path(temporary.name)
        self.assertEqual(self.directory.resolve().parent, base)
        self.addCleanup(temporary.cleanup)
        torch.set_num_threads(1)


class MatrixSelectionTests(TemporaryTest):
    def test_matrix_is_19_variants_57_formal_and_3_pilots(self):
        jobs = make_jobs()
        self.assertEqual(len(variants()), 19)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({j["job_id"] for j in jobs}), 60)
        self.assertEqual(sum(j["config"]["purpose"] == "formal" for j in jobs), 57)
        baseline = [j for j in make_jobs([42]) if j["stage"] == "baseline"]
        self.assertEqual(len(baseline), 4)
        selection = {"weight": 0.05, "rule": "synthetic fixture"}
        for job in jobs:
            config = resolve_config(job, "cpu", selection, "fixture_hash")
            self.assertEqual(config["training"]["epochs"], 20)
            self.assertIsNone(config["data"]["train_limit"])

    def write_pilots(self):
        sources, split = {"model.py": "fixture"}, "split_fixture"
        for job, score, mse in zip(pilot_jobs(), (0.8, 0.81, 0.81), (0.03, 0.03, 0.02)):
            run = self.directory / "runs" / job["job_id"]
            run.mkdir(parents=True)
            documents = {
                "config.json": resolve_config(job, "cpu"),
                "status.json": {"status": "completed"},
                "manifest.json": {"source_sha256": sources, "split_sha256": split},
                "summary.json": {"purpose": "pilot", "completed_epochs": 20, "train_size": 55000,
                                 "validation_size": 5000, "last_epoch": {"val_ssim": score, "val_mse": mse}},
            }
            for name, value in documents.items():
                (run / name).write_text(json.dumps(value), encoding="utf-8")
        return sources, split

    def test_selection_uses_validation_rule_and_detects_tampering(self):
        sources, split = self.write_pilots()
        choice = select_gradient_weight(self.directory, sources, split)
        self.assertEqual(choice["weight"], 0.1)
        self.assertFalse(choice["test_set_used"])
        self.assertEqual(verified_selection(self.directory, sources, split), choice)
        path = self.directory / "runs" / pilot_jobs()[0]["job_id"] / "summary.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            verified_selection(self.directory, sources, split)

    def test_gradient_jobs_need_selection_and_test_is_held_out(self):
        job = next(j for j in make_jobs([42]) if j["config"]["loss"]["gradient_weight"] is None)
        with self.assertRaises(ValueError):
            resolve_config(job, "cpu")
        with self.assertRaisesRegex(ValueError, "all 57"):
            discover_jobs(self.directory, "test")


class LossReferenceTests(TemporaryTest):
    def test_residual_targets_effective_input_minus_clean(self):
        torch.manual_seed(456)
        direct, residual = ConvolutionalDAE(4), ConvolutionalDAE(4, residual=True)
        residual.load_state_dict(direct.state_dict())
        clean = torch.rand(2, 1, 28, 28)
        noisy = (clean + 0.7).clamp(0, 1)
        raw, restored = direct(noisy), residual(noisy)
        torch.testing.assert_close(restored + raw, noisy)
        a = torch.nn.functional.mse_loss(restored, clean)
        b = torch.nn.functional.mse_loss(raw, noisy-clean)
        torch.testing.assert_close(a, b)
        self.assertEqual(sum(p.numel() for p in direct.parameters()), sum(p.numel() for p in residual.parameters()))

    def test_gradient_definition_combined_loss_and_constant_bias(self):
        clean = torch.zeros(1, 1, 28, 28)
        ramp = (torch.arange(28, dtype=torch.float32)/100).reshape(1, 1, 1, 28).expand_as(clean).clone().requires_grad_()
        self.assertAlmostEqual(float(gradient_l1(ramp, clean).detach()), 0.005, places=7)
        total, mse, edge = reconstruction_loss(ramp, clean, {"type": "mse_gradient", "gradient_weight": 0.1})
        torch.testing.assert_close(total, mse+0.1*edge)
        total.backward()
        self.assertTrue(torch.isfinite(ramp.grad).all())
        self.assertAlmostEqual(float(gradient_l1(clean+0.2, clean)), 0.0)
        np.testing.assert_allclose(per_image_gradient_error(clean, ramp.detach()), [0.005], atol=1e-8)
        with self.assertRaises(ValueError):
            validate_loss({"type": "mse_gradient", "gradient_weight": 0})

    def test_filter_does_not_mix_images_and_ties_choose_small_width(self):
        images = torch.rand(2, 1, 28, 28)
        np.testing.assert_array_equal(gaussian_denoise(images, 0), images.numpy())
        together = gaussian_denoise(images, 0.7)
        separate = np.concatenate([gaussian_denoise(images[i:i+1], 0.7) for i in range(2)])
        np.testing.assert_array_equal(together, separate)
        self.assertEqual(select_filter_width([{"width": 0.7, "MSE": 0.1}, {"width": 0.3, "MSE": 0.1}]), 0.3)


class EvaluationReportingTests(TemporaryTest):
    def test_evaluation_identity_infinite_psnr_and_verified_cache(self):
        loader = DataLoader(Images(), batch_size=3)
        values, rows, _, _ = score_method(loader, 0, 9300, lambda noisy, sigma: noisy)
        self.assertEqual(values["MSE"], 0)
        self.assertTrue(np.isposinf(values["PSNR"]))
        self.assertEqual(len(rows), 4)
        output = self.directory / "evaluation"
        identity, protocol = {"variant": "fixture_identity"}, {"split": "validation", "noise_seed": 9300}
        with patch("evaluate.SIGMAS", (0.0, 0.3)), contextlib.redirect_stdout(io.StringIO()):
            save_method(output, identity, protocol, loader, lambda noisy, sigma: noisy)
            save_method(output, identity, protocol, loader, lambda *args: self.fail("Cached evaluation reran"))
            (output / "summary.csv").write_text("modified", encoding="utf-8")
            with self.assertRaises(ValueError):
                save_method(output, identity, protocol, loader, lambda noisy, sigma: noisy)

    def test_sample_sd_and_paired_factorial_effects(self):
        rows = []
        for seed, shift in ((42, 0), (2026, 1), (3407, 2)):
            for letter, delta, name in (("A", 0, "baseline"), ("B", 1, "residual"),
                                        ("C", 2, "gradient"), ("D", 4, "residual_gradient")):
                variant = "cae_baseline_s03_c16" if letter == "A" else f"cae_{name}_s03_c16"
                rows.append({"variant": variant, "stage": "baseline" if letter == "A" else "ablation",
                    "sigma": 0.3, "seed": seed, "parameters": 64593, "n_images": 10000,
                    "network_ms_per_image": 0.1, "MSE": 0.1-0.001*(shift+delta), "PSNR": 20+shift+delta,
                    "SSIM": 0.7+0.01*(shift+delta), "GradientMAE": 0.01-0.0001*(shift+delta)})
        table = pd.DataFrame(rows)
        means = aggregate(table)
        base = means[means["variant"] == "cae_baseline_s03_c16"].iloc[0]
        self.assertEqual(base["PSNR_mean"], 21)
        self.assertEqual(base["PSNR_sd"], 1)
        paired, factorial = paired_ablation(table)
        np.testing.assert_allclose(factorial["PSNR_interaction"], 1)
        np.testing.assert_allclose(factorial["PSNR_residual_effect"], 1.5)
        np.testing.assert_allclose(factorial["PSNR_gradient_effect"], 2.5)
        self.assertEqual(len(paired), 12)
        for function, filename, value in ((save_noise_plot, "noise.png", means),
                                          (save_bottleneck_plot, "bottleneck.png", means),
                                          (save_ablation_plot, "ablation.png", paired)):
            path = self.directory / filename
            function(value, path, "synthetic QA")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
