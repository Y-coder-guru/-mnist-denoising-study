"""Source-verified tables, seed statistics, paired ablations and research figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline import load_test_partition, load_training_partitions
from evaluate import METRICS
from experiments import file_hash
from paper_figures import make_example_figures
from training import atomic_json


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return float(np.mean(values)), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def collect_tables(evaluation):
    completion = json.loads((evaluation / "completion.json").read_text(encoding="utf-8"))
    tables, classes = [], []
    for path in sorted(evaluation.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["request"]["protocol"] != completion["protocol"]:
            raise ValueError("Evaluation tables use different protocols")
        for name, expected in manifest["artifact_sha256"].items():
            if file_hash(path.parent / name) != expected:
                raise ValueError(f"Modified evaluation artifact: {path.parent / name}")
        table = pd.read_csv(path.parent / "summary.csv")
        tables.append(table)
        by_class = pd.read_csv(path.parent / "class_metrics.csv")
        for field in ("variant", "stage", "seed", "job_id"):
            by_class[field] = table.iloc[0][field]
        classes.append(by_class)
    if not tables:
        raise ValueError("No evaluation tables")
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated(["job_id", "sigma"]).any():
        raise ValueError("Duplicate evaluation records")
    expected_n = 10000 if completion["split"] == "test" else 5000
    if not (table["n_images"] == expected_n).all():
        raise ValueError("Unexpected evaluation sample count")
    if completion["split"] == "test":
        formal = table[table["stage"] != "reference"]
        if formal["job_id"].nunique() != 57 or len(formal) != 57 * 7:
            raise ValueError("Final test summary requires all 57 runs at all 7 noise levels")
    return table, pd.concat(classes, ignore_index=True), completion


def aggregate(table):
    rows = []
    for (variant, stage, sigma), group in table.groupby(["variant", "stage", "sigma"], sort=True):
        row = {"variant": variant, "stage": stage, "sigma": sigma, "n_runs": len(group),
               "parameters": int(group["parameters"].iloc[0]), "n_images_per_run": int(group["n_images"].iloc[0])}
        for metric in METRICS:
            row[metric + "_mean"], row[metric + "_sd"] = mean_sd(group[metric])
        row["network_ms_per_image_mean"] = group["network_ms_per_image"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ablation(table):
    paired, factorial = [], []
    for sigma in (0.3, 0.5):
        code = f"{int(sigma*10):02d}"
        names = {"A": f"cae_baseline_s{code}_c16", "B": f"cae_residual_s{code}_c16",
                 "C": f"cae_gradient_s{code}_c16", "D": f"cae_residual_gradient_s{code}_c16"}
        part = table[np.isclose(table["sigma"], sigma)]
        lookups = {letter: part[part["variant"] == name].set_index("seed") for letter, name in names.items()}
        if lookups["A"].empty:
            continue
        for letter, values in lookups.items():
            for seed in lookups["A"].index.intersection(values.index):
                base, value = lookups["A"].loc[seed], values.loc[seed]
                paired.append({"sigma": sigma, "variant": letter, "seed": seed,
                               **{metric + "_delta": float(value[metric]-base[metric]) for metric in METRICS}})
        if any(frame.empty for frame in lookups.values()):
            continue
        seeds = set.intersection(*(set(frame.index) for frame in lookups.values()))
        for seed in sorted(seeds):
            row = {"sigma": sigma, "seed": seed}
            for metric in METRICS:
                a, b, c, d = [lookups[letter].loc[seed, metric] for letter in "ABCD"]
                row[metric + "_residual_effect"] = 0.5 * ((b-a)+(d-c))
                row[metric + "_gradient_effect"] = 0.5 * ((c-a)+(d-b))
                row[metric + "_interaction"] = d-b-c+a
            factorial.append(row)
    return pd.DataFrame(paired), pd.DataFrame(factorial)


def save_noise_plot(average, path, split):
    names = {"fc_baseline_s03_z16": "FC fixed 0.3", "fc_baseline_s05_z16": "FC fixed 0.5",
             "cae_baseline_s03_c16": "CAE fixed 0.3", "cae_mixed_c16": "CAE mixed",
             "cae_conditioned_mixed_c16": "CAE mixed + sigma",
             "cae_residual_gradient_s03_c16": "Residual + gradient (0.3)",
             "noisy_input": "Noisy input", "gaussian_filter": "Gaussian filter"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for variant, label in names.items():
        part = average[(average["variant"] == variant) & (average["sigma"] > 0)].sort_values("sigma")
        if part.empty:
            continue
        for axis, metric in zip(axes, ("PSNR", "SSIM")):
            x = part["sigma"].to_numpy()
            y = part[metric + "_mean"].to_numpy()
            line, = axis.plot(x, y, marker="o", markersize=4, label=label)
            sd = part[metric + "_sd"].fillna(0).to_numpy()
            axis.fill_between(x, y-sd, y+sd, alpha=0.10, color=line.get_color())
    for axis, label in zip(axes, ("PSNR (dB)", "SSIM")):
        axis.set_xlabel("Input noise sigma")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.axvline(0.5, linestyle=":", color="gray", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{split.capitalize()} noise robustness | bands: sample SD across training seeds")
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_bottleneck_plot(average, path, split):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for sigma in (0.3, 0.5):
        code = f"{int(sigma*10):02d}"
        part = average[(average["variant"].str.startswith(f"fc_baseline_s{code}_z"))
                       & np.isclose(average["sigma"], sigma)].copy()
        if part.empty:
            continue
        part["dimension"] = part["variant"].str.extract(r"_z(\d+)$").astype(int)
        part = part.sort_values("dimension")
        for axis, metric in zip(axes, ("PSNR", "SSIM")):
            axis.errorbar(part["dimension"], part[metric+"_mean"], yerr=part[metric+"_sd"].fillna(0),
                          marker="o", capsize=3, label=f"Train/test sigma={sigma}")
    for axis, label in zip(axes, ("PSNR (dB)", "SSIM")):
        axis.set_xlabel("FC bottleneck dimension")
        axis.set_ylabel(label)
        axis.set_xticks([8, 16, 32, 64])
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle(f"{split.capitalize()} bottleneck comparison | mean and sample SD")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(paired, path, split):
    if paired.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sigma, offset in ((0.3, -0.06), (0.5, 0.06)):
        part = paired[np.isclose(paired["sigma"], sigma)]
        for axis, metric in zip(axes, ("PSNR", "SSIM")):
            x, means, sds = [], [], []
            for position, letter in enumerate("ABCD"):
                values = part[part["variant"] == letter][metric+"_delta"]
                if len(values):
                    mean, sd = mean_sd(values)
                    x.append(position+offset); means.append(mean); sds.append(sd if np.isfinite(sd) else 0)
            if x:
                axis.errorbar(x, means, yerr=sds, fmt="o", capsize=4, label=f"sigma={sigma}")
    for axis, label in zip(axes, ("Paired PSNR difference vs A (dB)", "Paired SSIM difference vs A")):
        axis.set_xticks(range(4), ["A: Base", "B: Residual", "C: Gradient", "D: Both"], rotation=15)
        axis.set_ylabel(label)
        axis.axhline(0, color="gray", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"{split.capitalize()} paired ablation | mean and sample SD across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/formal"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir if args.work_dir.is_absolute() else root / args.work_dir
    evaluation = work / "evaluation" / args.split
    table, classes, completion = collect_tables(evaluation)
    output = work / "report" / args.split
    output.mkdir(parents=True, exist_ok=True)
    average = aggregate(table)
    table.to_csv(output / "results_by_seed.csv", index=False, encoding="utf-8-sig")
    average.to_csv(output / "mean_std.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output / "per_digit_by_seed.csv", index=False, encoding="utf-8-sig")
    classes.groupby(["variant", "sigma", "label"])[list(METRICS)].agg(["mean", "std"]).to_csv(
        output / "per_digit_mean_std.csv", encoding="utf-8-sig")
    paired, factorial = paired_ablation(table)
    paired.to_csv(output / "paired_ablation.csv", index=False, encoding="utf-8-sig")
    factorial.to_csv(output / "factorial_effects.csv", index=False, encoding="utf-8-sig")
    save_noise_plot(average, output / "noise_robustness.png", args.split)
    save_bottleneck_plot(average, output / "bottleneck.png", args.split)
    save_ablation_plot(paired, output / "paired_ablation.png", args.split)
    dataset = load_test_partition(root) if args.split == "test" else load_training_partitions(root)[1]
    make_example_figures(root, work, evaluation, output, dataset, args.split, torch.device(args.device))
    atomic_json(output / "report_manifest.json", {"split": args.split,
        "formal_runs": completion["formal_runs_evaluated"], "source_completion_sha256": file_hash(evaluation / "completion.json"),
        "uncertainty": "Sample SD across training seeds on the same fixed evaluation images/noise; not noise-realization uncertainty.",
        "reference_runs": "Deterministic references evaluated once; sample SD is undefined for one run.",
        "zero_noise_psnr": "Perfect identity gives infinite PSNR; CSV retains inf and noise plots exclude sigma=0.",
        "ablation": "Differences paired by training seed; factorial interaction D-B-C+A. No significance claim from three seeds alone."})
    print(f"\nREPORT ARTIFACTS READY: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
