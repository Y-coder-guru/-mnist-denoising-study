"""Fixed representative examples and explicitly ranked failures for trained runs."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import build_model, predict_clean
from noise import fixed_evaluation_noise
from references import gaussian_denoise
from training import atomic_json


def load_model(run, device):
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    model = build_model(state["config"]["model"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def image_grid(rows, labels, image_ids, digits, path, title, error_last=False):
    fig, axes = plt.subplots(len(rows), len(image_ids), figsize=(max(7, len(image_ids) * 1.2), len(rows) * 1.25), squeeze=False)
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            axis = axes[r, c]
            axis.imshow(np.asarray(image).squeeze(), cmap="magma" if error_last and r == len(rows)-1 else "gray",
                        vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if c == 0:
                axis.set_ylabel(labels[r], fontsize=11)
            if r == 0:
                axis.set_title(f"digit {digits[c]}\nid {image_ids[c]}", fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def make_example_figures(root, work, evaluation, output, dataset, split, device):
    noise_seed = 19300 if split == "test" else 9300
    targets = dataset.source.targets[torch.from_numpy(dataset.indices)].numpy()
    positions = [int(np.flatnonzero(targets == digit)[0]) for digit in range(10)]
    items = [dataset[position] for position in positions]
    clean = torch.stack([item["clean"] for item in items])
    ids = torch.tensor([item["image_id"] for item in items])
    atomic_json(output / "representative_selection.json", {
        "rule": "First occurrence of each digit in the fixed partition order, independent of model scores.",
        "split": split, "image_ids_by_digit": ids.tolist(), "noise_seed": noise_seed})
    filters = json.loads((work / "gaussian_selection.json").read_text(encoding="utf-8"))
    for sigma in (0.3, 0.5):
        code = f"{int(sigma * 10):02d}"
        noisy = fixed_evaluation_noise(clean, ids, sigma, seed=noise_seed)
        rows = [clean[:, 0].numpy(), noisy[:, 0].numpy(),
                gaussian_denoise(noisy, filters["width_by_sigma"][str(sigma)])[:, 0]]
        labels = ["Clean", "Noisy", "Gaussian"]
        variants = [(f"fc_baseline_s{code}_z16", "FC z16"),
                    (f"cae_baseline_s{code}_c16", "CAE"),
                    (f"cae_residual_s{code}_c16", "Residual"),
                    (f"cae_gradient_s{code}_c16", "Gradient"),
                    (f"cae_residual_gradient_s{code}_c16", "Residual+gradient")]
        for variant, label in variants:
            run = work / "runs" / (variant + "_seed42")
            if not (run / "status.json").exists():
                continue
            model = load_model(run, device)
            restored = predict_clean(model, noisy.to(device), sigma).clamp(0, 1).cpu()
            rows.append(restored[:, 0].numpy())
            labels.append(label)
        image_grid(rows, labels, ids.tolist(), list(range(10)), output / f"representatives_sigma{code}.png",
                   f"{split.capitalize()} representatives | sigma={sigma} | same inputs for all methods")

    variant = "cae_residual_gradient_s05_c16"
    job_id = variant + "_seed42"
    metric_path = evaluation / job_id / "per_image.csv.gz"
    base_run = work / "runs" / "cae_baseline_s05_c16_seed42"
    if not metric_path.exists() or not (base_run / "last.pt").exists():
        return
    metrics = pd.read_csv(metric_path)
    worst = metrics[np.isclose(metrics["sigma"], 0.5)].sort_values(["SSIM", "image_id"], ascending=True).head(8)
    lookup = {int(image_id): position for position, image_id in enumerate(dataset.indices)}
    samples = [dataset[lookup[int(image_id)]] for image_id in worst["image_id"]]
    clean = torch.stack([sample["clean"] for sample in samples])
    ids = torch.tensor([sample["image_id"] for sample in samples])
    noisy = fixed_evaluation_noise(clean, ids, 0.5, seed=noise_seed)
    baseline = load_model(base_run, device)
    combined = load_model(work / "runs" / job_id, device)
    baseline_images = predict_clean(baseline, noisy.to(device), 0.5).clamp(0, 1).cpu()
    combined_images = predict_clean(combined, noisy.to(device), 0.5).clamp(0, 1).cpu()
    rows = [clean[:, 0].numpy(), noisy[:, 0].numpy(), baseline_images[:, 0].numpy(),
            combined_images[:, 0].numpy(), (combined_images-clean).abs()[:, 0].numpy()]
    image_grid(rows, ["Clean", "Noisy", "CAE", "Residual+gradient", "Absolute error"],
               ids.tolist(), [sample["label"] for sample in samples], output / "worst_cases_sigma05.png",
               f"{split.capitalize()}: 8 lowest-SSIM residual+gradient cases | error scale 0 to 1", error_last=True)
    atomic_json(output / "failure_selection.json", {
        "rule": "Eight lowest SSIM values of the residual+gradient model at sigma=0.5, seed42; ties use smaller image ID.",
        "job_id": job_id, "split": split, "noise_seed": noise_seed,
        "samples": worst[["image_id", "label", "SSIM", "PSNR"]].to_dict(orient="records")})
