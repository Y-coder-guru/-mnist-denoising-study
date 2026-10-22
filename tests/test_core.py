"""Offline scientific-protocol checks on synthetic tensors only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import per_image_metrics, summarize_metrics
from models import FullyConnectedDAE, ConvolutionalDAE, build_model, model_description, predict_clean
from noise import corrupt_training, fixed_evaluation_noise, sample_mixed_sigmas


def setUpModule():
    torch.set_num_threads(1)


class MetricTests(unittest.TestCase):
    def test_known_error_and_per_image_psnr_aggregation(self):
        clean = np.zeros((2, 28, 28))
        restored = np.stack([np.full((28, 28), 0.1), np.full((28, 28), 0.2)])
        values = per_image_metrics(clean, restored)
        np.testing.assert_allclose(values["MSE"], [0.01, 0.04], atol=1e-12)
        np.testing.assert_allclose(values["PSNR"], [20.0, 13.979400086720375], atol=1e-10)
        self.assertAlmostEqual(summarize_metrics(values)["PSNR"], 16.989700043360187)
        self.assertGreater(abs(summarize_metrics(values)["PSNR"]
                               + 10 * np.log10(np.mean(values["MSE"]))), 0.9)

    def test_identity(self):
        clean = np.random.default_rng(123).random((2, 28, 28))
        values = per_image_metrics(clean, clean)
        np.testing.assert_array_equal(values["MSE"], 0)
        np.testing.assert_allclose(values["SSIM"], 1)
        self.assertTrue(np.isposinf(values["PSNR"]).all())

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            per_image_metrics(np.zeros((1, 28, 28)), np.full((1, 28, 28), 1.1))


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.clean = torch.full((4, 1, 28, 28), 0.5)
        self.ids = torch.tensor([31, 99, 5, 100])

    def test_batch_and_order_independence(self):
        whole = fixed_evaluation_noise(self.clean, self.ids, 0.3, seed=9300)
        pieces = torch.cat([fixed_evaluation_noise(self.clean[i:i+1], self.ids[i:i+1],
                                                  0.3, seed=9300) for i in range(4)])
        self.assertTrue(torch.equal(whole, pieces))
        order = torch.tensor([3, 0, 2, 1])
        reordered = fixed_evaluation_noise(self.clean[order], self.ids[order], 0.3, seed=9300)
        self.assertTrue(torch.equal(whole[order], reordered))

    def test_same_standard_draw_across_sigmas(self):
        a = fixed_evaluation_noise(self.clean, self.ids, 0.001, seed=9300)
        b = fixed_evaluation_noise(self.clean, self.ids, 0.002, seed=9300)
        torch.testing.assert_close((a - self.clean) / 0.001, (b - self.clean) / 0.002,
                                   atol=4e-5, rtol=1e-4)

    def test_separate_evaluation_seeds(self):
        a = fixed_evaluation_noise(self.clean, self.ids, 0.3, seed=9300)
        b = fixed_evaluation_noise(self.clean, self.ids, 0.3, seed=19300)
        self.assertFalse(torch.equal(a, b))

    def test_zero_sigma_and_dynamic_mixed_noise(self):
        self.assertTrue(torch.equal(fixed_evaluation_noise(self.clean, self.ids, 0.0, seed=9300), self.clean))
        generator = torch.Generator().manual_seed(42)
        sigmas = sample_mixed_sigmas(4, generator=generator)
        self.assertTrue(((sigmas >= 0.1) & (sigmas <= 0.5)).all())
        a = corrupt_training(self.clean, sigmas, generator=generator)
        b = corrupt_training(self.clean, sigmas, generator=generator)
        self.assertFalse(torch.equal(a, b))
        self.assertTrue(((a >= 0) & (a <= 1)).all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fixed_evaluation_noise(self.clean, self.ids.float(), 0.3, seed=9300)
        with self.assertRaises(ValueError):
            corrupt_training(self.clean, -0.1)


class ModelTests(unittest.TestCase):
    def test_bottleneck_output_shapes_and_ranges(self):
        clean = torch.rand(2, 1, 28, 28)
        for latent_dim in (8, 16, 32, 64):
            model = FullyConnectedDAE(latent_dim)
            with torch.no_grad():
                output = model(clean)
            self.assertEqual(output.shape, clean.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_convolutional_baseline_shape_and_finite_gradients(self):
        model = build_model({"type": "cae", "base_channels": 16})
        clean = torch.rand(2, 1, 28, 28)
        output = model(clean)
        self.assertEqual(output.shape, clean.shape)
        self.assertTrue(torch.isfinite(output).all())
        torch.nn.functional.mse_loss(output, clean).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        metadata = model_description(model)
        self.assertEqual(metadata["auxiliary_channel"], "constant_zero")
        self.assertEqual(metadata["training_output"], "linear_unclipped")

    def test_teacher_baseline_parameter_count_preserved(self):
        model = build_model({"type": "fc", "latent_dim": 16})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 437664)

    def test_invalid_model_settings_rejected(self):
        with self.assertRaises(ValueError):
            ConvolutionalDAE(base_channels=0)
        with self.assertRaises(ValueError):
            build_model({"type": "unsupported"})

    def test_conditioning_matches_capacity_and_zero_information_control(self):
        torch.manual_seed(123)
        control = ConvolutionalDAE(base_channels=4)
        conditioned = ConvolutionalDAE(base_channels=4, conditioned=True)
        conditioned.load_state_dict(control.state_dict())
        self.assertEqual(sum(p.numel() for p in control.parameters()),
                         sum(p.numel() for p in conditioned.parameters()))
        images = torch.rand(2, 1, 28, 28)
        with torch.no_grad():
            reference = control(images)
            self.assertTrue(torch.equal(reference, conditioned(images, sigma=0.0)))
            self.assertFalse(torch.equal(reference, conditioned(images, sigma=0.3)))
        self.assertTrue(model_description(conditioned)["requires_known_noise_strength"])

    def test_conditioned_model_supports_per_image_sigmas(self):
        model = ConvolutionalDAE(base_channels=4, conditioned=True)
        images = torch.rand(2, 1, 28, 28)
        with torch.no_grad():
            batch = predict_clean(model, images, torch.tensor([0.1, 0.5]))
            separate = torch.cat([model(images[:1], sigma=0.1), model(images[1:], sigma=0.5)])
        torch.testing.assert_close(batch, separate, atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            model(images)
        with self.assertRaises(ValueError):
            model(images, sigma=-0.1)


if __name__ == "__main__":
    unittest.main()
