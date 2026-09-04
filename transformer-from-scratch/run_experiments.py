"""
Runs the controlled ablations agreed on for this project: vary ONE
hyperparameter at a time from a shared baseline, everything else held fixed,
so any change in val loss/perplexity can be attributed to that one axis.

Baseline (matches the confirmed baseline run in the README):
    d_model=128, n_heads=4, n_layers=4, block_size=128, lr=3e-4,
    max_iters=5000, seed=42

Axes:
    context   -> block_size in {32, 128, 256}
    depth     -> n_layers  in {2, 4, 6}
    lr        -> lr        in {1e-4, 3e-4, 1e-3}
    tie       -> tie_weights in {False, True}, params/perplexity trade-off
    seeds     -> baseline config re-run at seed in {42, 123, 2026}, reports mean/std PPL
    all       -> runs all five axes

Writes, per axis, one subfolder under --out_root (default "results/") with:
    - one loss_log_<tag>.csv and summary_<tag>.json per run (via train.run_training)
    - one loss.png overlay plot (via plot_results.py logic)
    - one summary.csv table (the row-per-config table for the README)

Usage:
    python run_experiments.py --axis context --max_iters 5000
    python run_experiments.py --axis all --max_iters 5000
"""
import argparse
import csv
import math
import os
from types import SimpleNamespace

import torch

import train as train_module
import plot_results


BASELINE = dict(
    d_model=128, n_heads=4, n_layers=4, block_size=128, batch_size=64,
    lr=3e-4, max_iters=5000, eval_interval=250, dropout=0.1,
    seed=42, tie_weights=False,
)


def make_args(overrides: dict, out_dir: str, tag: str) -> SimpleNamespace:
    cfg = {**BASELINE, **overrides, "out_dir": out_dir, "tag": tag}
    return SimpleNamespace(**cfg)


def run_axis(axis_name: str, configs: list, out_root: str, max_iters: int):
    """configs: list of (tag, overrides_dict). Runs each, plots overlay, writes summary.csv."""
    axis_dir = os.path.join(out_root, axis_name)
    os.makedirs(axis_dir, exist_ok=True)

    summaries = []
    for tag, overrides in configs:
        run_dir = os.path.join(axis_dir, tag)
        os.makedirs(run_dir, exist_ok=True)
        overrides = {**overrides, "max_iters": max_iters}
        args = make_args(overrides, run_dir, tag)
        print(f"\n--- [{axis_name}] running {tag}: {overrides} ---")
        summary = train_module.run_training(args)
        summaries.append(summary)

    # overlay plot across all configs in this axis
    csv_paths = [s["loss_log"] for s in summaries]
    labels = [s["tag"] for s in summaries]
    plot_args = argparse.Namespace(
        csv=csv_paths, labels=labels, out=os.path.join(axis_dir, "loss_comparison.png"), metric="both",
    )
    _plot_from_namespace(plot_args)

    # summary table
    table_path = os.path.join(axis_dir, "summary.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "n_params", "final_val_loss", "final_val_ppl", "training_time_s"])
        for s in summaries:
            writer.writerow([s["tag"], s["n_params"], s["final_val_loss"], s["final_val_ppl"], s["training_time_s"]])

    print(f"\n[{axis_name}] summary table -> {table_path}")
    print(f"[{axis_name}] comparison plot -> {axis_dir}/loss_comparison.png")
    return summaries


def _plot_from_namespace(ns):
    # reuse plot_results' plotting logic without re-parsing argv
    labels = ns.labels or [os.path.basename(c) for c in ns.csv]
    os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for csv_path, label in zip(ns.csv, labels):
        iters, train_losses, val_losses = plot_results.read_log(csv_path)
        axes[0].plot(iters, train_losses, "--", alpha=0.5, label=f"{label} (train)")
        axes[0].plot(iters, val_losses, "-", label=f"{label} (val)")
        axes[1].plot(iters, [math.exp(v) for v in val_losses], "-", label=f"{label}")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("perplexity"); axes[1].set_title("Val Perplexity"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ns.out, dpi=150)


def run_seeds(out_root: str, max_iters: int, seeds=(42, 123, 2026)):
    axis_dir = os.path.join(out_root, "seeds")
    os.makedirs(axis_dir, exist_ok=True)

    summaries = []
    for seed in seeds:
        run_dir = os.path.join(axis_dir, f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        args = make_args({"seed": seed, "max_iters": max_iters}, run_dir, f"seed{seed}")
        print(f"\n--- [seeds] running seed={seed} ---")
        summary = train_module.run_training(args)
        summaries.append(summary)

    ppls = [s["final_val_ppl"] for s in summaries]
    losses = [s["final_val_loss"] for s in summaries]
    mean_ppl, std_ppl = sum(ppls) / len(ppls), torch.tensor(ppls).std().item()
    mean_loss, std_loss = sum(losses) / len(losses), torch.tensor(losses).std().item()

    table_path = os.path.join(axis_dir, "summary.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "final_val_loss", "final_val_ppl"])
        for s in summaries:
            writer.writerow([s["seed"], s["final_val_loss"], s["final_val_ppl"]])
        writer.writerow(["mean", mean_loss, mean_ppl])
        writer.writerow(["std", std_loss, std_ppl])

    print(f"\n[seeds] val PPL: {[round(p, 2) for p in ppls]} -> mean {mean_ppl:.2f} +/- {std_ppl:.2f}")
    print(f"[seeds] summary table -> {table_path}")
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["context", "depth", "lr", "tie", "seeds", "all"], default="all")
    p.add_argument("--max_iters", type=int, default=5000, help="reduce for a fast CPU sanity pass, e.g. 500")
    p.add_argument("--out_root", type=str, default="results")
    args = p.parse_args()

    if args.axis in ("context", "all"):
        run_axis("context", [
            ("ctx32", {"block_size": 32}),
            ("ctx128", {"block_size": 128}),
            ("ctx256", {"block_size": 256}),
        ], args.out_root, args.max_iters)

    if args.axis in ("depth", "all"):
        run_axis("depth", [
            ("layers2", {"n_layers": 2}),
            ("layers4", {"n_layers": 4}),
            ("layers6", {"n_layers": 6}),
        ], args.out_root, args.max_iters)

    if args.axis in ("lr", "all"):
        run_axis("learning_rate", [
            ("lr1e-4", {"lr": 1e-4}),
            ("lr3e-4", {"lr": 3e-4}),
            ("lr1e-3", {"lr": 1e-3}),
        ], args.out_root, args.max_iters)

    if args.axis in ("tie", "all"):
        # the weight-tying axis, so the result can be
        # compared across character-level and subword tokenization
        run_axis("weight_tying", [
            ("untied", {"tie_weights": False}),
            ("tied", {"tie_weights": True}),
        ], args.out_root, args.max_iters)

    if args.axis in ("seeds", "all"):
        run_seeds(args.out_root, args.max_iters)


if __name__ == "__main__":
    main()
