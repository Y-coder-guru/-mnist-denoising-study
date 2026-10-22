"""Diagnostic figures generated from recorded runs, with explicit smoke labels."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from noise import fixed_evaluation_noise
from models import predict_clean


def plot_history(history, path, purpose):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(epochs, [r["train_mse"] for r in history], marker="o", label="Training")
    axes[0].plot(epochs, [r["val_mse"] for r in history], marker="o", label="Validation")
    axes[0].set_ylabel("MSE (lower is better)")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_psnr"] for r in history], marker="o", color="tab:blue")
    axes[1].set_ylabel("Validation PSNR (dB)")
    axes[2].plot(epochs, [r["val_ssim"] for r in history], marker="o", color="tab:green")
    axes[2].set_ylabel("Validation SSIM")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs if len(epochs) <= 10 else epochs[::2])
        axis.grid(alpha=0.2)
    label = "SMOKE TEST | NOT PAPER RESULTS" if purpose == "smoke" else "Training and validation history"
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if all("train_loss" in row for row in history):
        components, panels = plt.subplots(1, 2, figsize=(9, 3.4))
        panels[0].plot(epochs, [r["train_loss"] for r in history], label="Total objective")
        panels[0].plot(epochs, [r["train_mse"] for r in history], label="Pixel MSE", linestyle="--")
        panels[0].set_ylabel("Training objective")
        panels[0].legend()
        panels[1].plot(epochs, [r["train_loss"]-r["train_mse"] for r in history], color="tab:purple")
        panels[1].set_ylabel("Weighted gradient contribution")
        for axis in panels:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
        components.suptitle("SMOKE TEST | NOT PAPER RESULTS" if purpose == "smoke" else "Training loss components")
        components.tight_layout()
        components.savefig(path.with_name("loss_components.png"), dpi=200, bbox_inches="tight")
        plt.close(components)


@torch.no_grad()
def plot_validation_preview(model, loader, sigma, noise_seed, device, path, purpose):
    model.eval()
    batch = next(iter(loader))
    clean = batch["clean"][:10].to(device)
    ids = batch["image_id"][:10]
    noisy = fixed_evaluation_noise(clean, ids, sigma, seed=noise_seed)
    restored = predict_clean(model, noisy, sigma).clamp(0, 1)
    rows = [clean.cpu(), noisy.cpu(), restored.cpu(), (restored - clean).abs().cpu()]
    labels = ["Clean", f"Noisy sigma={sigma}", "Reconstructed", "Absolute error"]
    fig, axes = plt.subplots(4, len(clean), figsize=(max(5, len(clean) * 1.2), 5), squeeze=False)
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            axis = axes[r, c]
            axis.imshow(image[0].numpy(), cmap="magma" if r == 3 else "gray", vmin=0, vmax=1,
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if c == 0:
                axis.set_ylabel(labels[r], fontsize=9)
            if r == 0:
                axis.set_title(f"digit {int(batch['label'][c])}\nid {int(ids[c])}", fontsize=8)
    label = "SMOKE TEST | NOT PAPER RESULTS" if purpose == "smoke" else "Final-epoch validation examples"
    fig.suptitle(label + " | error color scale: 0 to 1")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
