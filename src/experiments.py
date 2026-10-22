"""The frozen 19-configuration matrix and validation-only loss-weight selection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from training import atomic_json, validate_config


SEEDS = (42, 2026, 3407)
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
WEIGHTS = (0.01, 0.05, 0.1)
STAGES = ("baseline", "bottleneck", "mixed", "pilot", "ablation")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_config(name, model, sigma=0.3, mixed=False):
    noise = {"noise_mode": "mixed", "sigma_low": 0.1, "sigma_high": 0.5} if mixed else {"noise_mode": "fixed", "sigma": sigma}
    return {"name": name, "purpose": "formal", "seed": 42, "device": "cuda", "cpu_threads": 2,
            "model": model, "training": {"epochs": 20, "batch_size": 128, "learning_rate": 0.001, **noise},
            "loss": {"type": "mse", "gradient_weight": 0.0},
            "data": {"train_limit": None, "validation_limit": None},
            "validation": {"sigma": sigma, "noise_seed": 9300}}


def variants():
    result = []
    for dim in (16, 8, 32, 64):
        for sigma in (0.3, 0.5):
            name = f"fc_baseline_s{int(sigma * 10):02d}_z{dim}"
            result.append({"stage": "baseline" if dim == 16 else "bottleneck",
                           "variant": name, "config": base_config(name, {"type": "fc", "latent_dim": dim}, sigma)})
    for sigma in (0.3, 0.5):
        name = f"cae_baseline_s{int(sigma * 10):02d}_c16"
        result.append({"stage": "baseline", "variant": name,
                       "config": base_config(name, {"type": "cae", "base_channels": 16}, sigma)})
    for name, model in (("fc_mixed_z16", {"type": "fc", "latent_dim": 16}),
                        ("cae_mixed_c16", {"type": "cae", "base_channels": 16}),
                        ("cae_conditioned_mixed_c16", {"type": "cae_conditioned", "base_channels": 16})):
        result.append({"stage": "mixed", "variant": name, "config": base_config(name, model, mixed=True)})
    for residual, gradient, label in ((True, False, "residual"), (False, True, "gradient"),
                                      (True, True, "residual_gradient")):
        for sigma in (0.3, 0.5):
            name = f"cae_{label}_s{int(sigma * 10):02d}_c16"
            config = base_config(name, {"type": "cae", "base_channels": 16, "residual": residual}, sigma)
            if gradient:
                config["loss"] = {"type": "mse_gradient", "gradient_weight": None}
            result.append({"stage": "ablation", "variant": name, "config": config})
    if len(result) != 19 or len({item["variant"] for item in result}) != 19:
        raise RuntimeError("Experiment matrix is inconsistent")
    return result


def pilot_jobs():
    result = []
    for weight in WEIGHTS:
        name = "pilot_gradient_w" + str(weight).replace(".", "p")
        config = base_config(name, {"type": "cae", "base_channels": 16})
        config["purpose"] = "pilot"
        config["loss"] = {"type": "mse_gradient", "gradient_weight": weight}
        result.append({"stage": "pilot", "variant": name, "seed": 42,
                       "job_id": name + "_seed42", "config": config})
    return result


def make_jobs(seeds=SEEDS):
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"Use distinct planned seeds from {SEEDS}")
    result = []
    for index, seed in enumerate(seeds):
        for stage in ("baseline", "bottleneck", "mixed", "ablation"):
            if stage == "ablation" and index == 0:
                result.extend(pilot_jobs())
            for item in variants():
                if item["stage"] != stage:
                    continue
                item = copy.deepcopy(item)
                item["config"]["seed"] = seed
                item.update(seed=seed, job_id=f"{item['variant']}_seed{seed}")
                result.append(item)
    return result


def verified_selection(work_dir, sources, split_hash):
    path = work_dir / "gradient_selection.json"
    if not path.exists():
        return None
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection["source_sha256"] != sources or selection["split_sha256"] != split_hash:
        raise ValueError("Gradient selection comes from different code or data")
    for job_id, expected in selection["pilot_summary_sha256"].items():
        if file_hash(work_dir / "runs" / job_id / "summary.json") != expected:
            raise ValueError("A selection pilot summary has changed")
    if selection["weight"] not in WEIGHTS:
        raise ValueError("Selected weight is outside the preregistered candidates")
    return selection


def select_gradient_weight(work_dir, sources, split_hash):
    existing = verified_selection(work_dir, sources, split_hash)
    if existing is not None:
        return existing
    candidates, hashes = [], {}
    for job in pilot_jobs():
        run = work_dir / "runs" / job["job_id"]
        if not (run / "summary.json").exists() or not (run / "status.json").exists():
            return None
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            return None
        if (summary["purpose"] != "pilot" or summary["completed_epochs"] != 20
                or summary["train_size"] != 55000 or summary["validation_size"] != 5000
                or manifest["source_sha256"] != sources or manifest["split_sha256"] != split_hash):
            raise ValueError("Invalid selection pilot provenance")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if config != resolve_config(job, config["device"]):
            raise ValueError("Selection pilot settings differ from the declared protocol")
        row = summary["last_epoch"]
        candidates.append({"job_id": job["job_id"], "weight": config["loss"]["gradient_weight"],
                           "validation_ssim": row["val_ssim"], "validation_mse": row["val_mse"]})
        hashes[job["job_id"]] = file_hash(run / "summary.json")
    winner = sorted(candidates, key=lambda c: (-c["validation_ssim"], c["validation_mse"], c["weight"]))[0]
    selection = {"weight": winner["weight"], "winner": winner["job_id"], "candidates": candidates,
                 "rule": "Maximum final-epoch validation SSIM; ties: lower MSE, then lower weight.",
                 "pilot_model": "direct_CAE_sigma0.3_seed42", "test_set_used": False,
                 "source_sha256": sources, "split_sha256": split_hash,
                 "pilot_summary_sha256": hashes}
    atomic_json(work_dir / "gradient_selection.json", selection)
    return selection


def resolve_config(job, device, selection=None, selection_hash=None):
    config = copy.deepcopy(job["config"])
    config["device"] = device
    if config["loss"]["gradient_weight"] is None:
        if selection is None:
            raise ValueError("Run the three validation pilots before gradient-loss ablations")
        config["loss"]["gradient_weight"] = selection["weight"]
        config["selection"] = {"artifact_sha256": selection_hash, "test_set_used": False,
                               "rule": selection["rule"]}
    config["experiment"] = {"job_id": job["job_id"], "stage": job["stage"], "variant": job["variant"]}
    validate_config(config)
    return config
