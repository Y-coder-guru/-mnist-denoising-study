"""Tiny synthetic integration test of the actual CLI, serialization and recovery."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_formal_workflow import Images
import train


class TrainingEntryTests(unittest.TestCase):
    def test_cli_resume_matches_full_run_and_exports_images(self):
        base = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(dir=base)
        root = Path(temporary.name)
        self.assertEqual(root.resolve().parent, base)
        self.addCleanup(temporary.cleanup)
        (root / "src").mkdir()
        for name in train.SOURCES:
            shutil.copy2(base.parent / "src" / name, root / "src" / name)
        (root / "data/processed").mkdir(parents=True)
        (root / "data/processed/split_v1.npz").write_bytes(b"synthetic fixture split")
        config = yaml.safe_load((base.parent / "configs/fc_smoke.yaml").read_text(encoding="utf-8"))
        config["data"] = {"train_limit": 4, "validation_limit": 3}
        config["model"]["latent_dim"] = 8
        config["training"]["batch_size"] = 2
        config["cpu_threads"] = 1
        path = root / "fixture.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        datasets = (Images(4, 0), Images(3, 100))
        real_epoch = train.train_epoch
        calls = 0

        def interrupt_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return real_epoch(*args, **kwargs)

        def invoke(arguments, interrupt=False):
            with patch.object(sys, "argv", ["train.py", "--root", str(root), *arguments]), \
                    patch("train.load_training_partitions", return_value=datasets), \
                    patch("train.subprocess.run", return_value=Mock(returncode=0, stdout="fixture==1\n")), \
                    contextlib.redirect_stdout(io.StringIO()):
                if interrupt:
                    with patch("train.train_epoch", side_effect=interrupt_second):
                        return train.main()
                return train.main()

        resumed = root / "outputs/resumed"
        continuous = root / "outputs/continuous"
        self.assertEqual(invoke(["--config", str(path), "--run-dir", str(resumed)], interrupt=True), 130)
        self.assertEqual(json.loads((resumed / "status.json").read_text())["last_completed_epoch"], 1)
        self.assertEqual(invoke(["--resume", str(resumed)]), 0)
        self.assertEqual(invoke(["--config", str(path), "--run-dir", str(continuous)]), 0)
        a = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        b = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
        for key, value in a["model_state"].items():
            self.assertTrue(torch.equal(value, b["model_state"][key]), key)
        self.assertEqual(a["history"][-1]["train_mse"], b["history"][-1]["train_mse"])
        self.assertEqual(a["history"][-1]["val_ssim"], b["history"][-1]["val_ssim"])
        for name in ("curves.png", "validation_preview.png", "loss_components.png"):
            self.assertEqual((resumed / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(json.loads((resumed / "summary.json").read_text())["test_set_evaluated"], False)


if __name__ == "__main__":
    unittest.main()
