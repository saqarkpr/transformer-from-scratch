"""
Turns loss_log_*.csv files (written by train.py) into PNG plots, so every
experiment gets a figure automatically instead of being plotted by hand.

Usage:
    # plot a single run
    python plot_results.py --csv checkpoints/loss_log_baseline.csv --out results/baseline/loss.png

    # overlay several runs on one plot (e.g. all context-length ablation runs)
    python plot_results.py --csv results/context/ctx32/loss_log_ctx32.csv \
                                  results/context/ctx128/loss_log_ctx128.csv \
                                  results/context/ctx256/loss_log_ctx256.csv \
                            --labels "block_size=32" "block_size=128" "block_size=256" \
                            --out results/context/loss_comparison.png
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    iters, train_losses, val_losses = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return iters, train_losses, val_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, nargs="+", required=True, help="one or more loss_log CSV paths")
    p.add_argument("--labels", type=str, nargs="+", default=None, help="legend label per CSV (defaults to filename)")
    p.add_argument("--out", type=str, required=True, help="output PNG path")
    p.add_argument("--metric", choices=["loss", "perplexity", "both"], default="both")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(c) for c in args.csv]
    assert len(labels) == len(args.csv), "--labels must match --csv in count"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_panels = 2 if args.metric == "both" else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for csv_path, label in zip(args.csv, labels):
        iters, train_losses, val_losses = read_log(csv_path)
        panel = 0
        if args.metric in ("loss", "both"):
            axes[panel].plot(iters, train_losses, "--", alpha=0.5, label=f"{label} (train)")
            axes[panel].plot(iters, val_losses, "-", label=f"{label} (val)")
            panel += 1
        if args.metric in ("perplexity", "both"):
            val_ppl = [math.exp(v) for v in val_losses]
            axes[panel].plot(iters, val_ppl, "-", label=f"{label} (val PPL)")

    panel = 0
    if args.metric in ("loss", "both"):
        axes[panel].set_xlabel("iteration")
        axes[panel].set_ylabel("cross-entropy loss")
        axes[panel].set_title("Training / Validation Loss")
        axes[panel].legend(fontsize=8)
        axes[panel].grid(alpha=0.3)
        panel += 1
    if args.metric in ("perplexity", "both"):
        axes[panel].set_xlabel("iteration")
        axes[panel].set_ylabel("perplexity")
        axes[panel].set_title("Validation Perplexity")
        axes[panel].legend(fontsize=8)
        axes[panel].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
