"""Queue lifecycle and allowlisted bundle checks; no real training/downloads."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build_bundle
import run_experiments
from experiments import file_hash, make_jobs
from train import SOURCES, source_hashes


class QueueBundleTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent
        temporary = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(temporary.name)
        self.assertEqual(self.root.resolve().parent, base)
        self.addCleanup(temporary.cleanup)
        (self.root / "src").mkdir()
        for name in SOURCES:
            shutil.copy2(base.parent / "src" / name, self.root / "src" / name)
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "data/processed/split_v1.npz").write_bytes(b"synthetic split fixture")

    def test_queue_new_skip_and_resume_lifecycle(self):
        job = next(j for j in make_jobs([42]) if j["stage"] == "baseline")
        commands = []

        def child(command, **kwargs):
            commands.append(command)
            if "--resume" in command:
                run = Path(command[command.index("--resume")+1])
                config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            else:
                run = Path(command[command.index("--run-dir")+1])
                config = yaml.safe_load(Path(command[command.index("--config")+1]).read_text(encoding="utf-8"))
                run.mkdir(parents=True)
            documents = {"status.json": {"status": "completed"}, "config.json": config,
                "manifest.json": {"source_sha256": source_hashes(self.root),
                                  "split_sha256": file_hash(self.root / "data/processed/split_v1.npz")},
                "summary.json": {"completed_epochs": 20, "purpose": "formal", "train_size": 55000,
                                 "validation_size": 5000}}
            for name, value in documents.items():
                (run / name).write_text(json.dumps(value), encoding="utf-8")
            for name in ("last.pt", "metrics.csv", "validation_per_image.csv"):
                (run / name).write_bytes(b"synthetic lifecycle fixture; not a real checkpoint")
            return Mock(returncode=0)

        def invoke():
            with patch.object(sys, "argv", ["run_experiments.py", "--root", str(self.root),
                                           "--stage", "baseline", "--seeds", "42", "--device", "cpu"]), \
                 patch("run_experiments.make_jobs", return_value=[job]), \
                 patch("run_experiments.load_saved_split"), patch("run_experiments.verify_raw_integrity"), \
                 patch("run_experiments.subprocess.run", side_effect=child), \
                 contextlib.redirect_stdout(io.StringIO()):
                return run_experiments.main()

        self.assertEqual(invoke(), 0)
        self.assertEqual(len(commands), 1)
        self.assertIn("--run-dir", commands[0])
        self.assertEqual(invoke(), 0)
        self.assertEqual(len(commands), 1)
        run = self.root / "outputs/formal/runs" / job["job_id"]
        (run / "status.json").write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
        self.assertEqual(invoke(), 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("--resume", commands[1])

    def test_bundle_contains_data_and_protocol_but_excludes_environments_and_runs(self):
        for name in ("run_experiments.py", "evaluate.py", "summarize.py", "collect_results.py"):
            (self.root / "src" / name).write_text("# synthetic bundle fixture\n", encoding="utf-8")
        for folder in ("configs", "docs", "outputs/data_check", "data/MNIST/raw", ".envs", "outputs/runs/old"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        for name in ("README.md", "requirements-runtime.txt", "outputs/data_check/data_report.json",
                     ".envs/private.txt", "outputs/runs/old/last.pt"):
            (self.root / name).write_text("fixture", encoding="utf-8")
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
            (self.root / "data/MNIST/raw" / name).write_bytes(b"synthetic data fixture")
        with patch.object(sys, "argv", ["build_bundle.py", "--root", str(self.root)]), \
                patch("build_bundle.load_saved_split"), patch("build_bundle.verify_raw_integrity"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build_bundle.main(), 0)
        archives = list((self.root / "dist").glob("*.zip"))
        self.assertEqual(len(archives), 1)
        with ZipFile(archives[0]) as archive:
            names = archive.namelist()
            self.assertIsNone(archive.testzip())
            self.assertIn("mnist_project/data/processed/split_v1.npz", names)
            self.assertIn("mnist_project/outputs/data_check/data_report.json", names)
            self.assertFalse(any(".envs" in name or "outputs/runs" in name for name in names))
            matrix = json.loads(archive.read("mnist_project/configs/task_matrix.json"))
            self.assertEqual(len(matrix), 60)


if __name__ == "__main__":
    unittest.main()
