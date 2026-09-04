"""
Generates the figures used in this README from the committed result files.

Two figures, both regenerable from `results/`:

  fig_ablations.png     all four axes as final-perplexity bars, with the
                        measured seed sigma drawn as a band so "real vs noise"
                        is visible rather than asserted.

  fig_tying_budget.png  the headline result: the tied/untied perplexity ratio
                        across training. This is the one worth looking at --
                        it shows an ablation conclusion inverting as a function
                        of training budget, with one boolean flag as the only
                        difference between the two runs.

    python make_figures.py
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
OUT = "results"


def read_summary(axis):
    with open(f"{RESULTS}/{axis}/summary.csv") as f:
        return [r for r in csv.DictReader(f) if r["tag"] not in ("mean", "std")]


def read_curve(axis, tag):
    its, val = [], []
    with open(f"{RESULTS}/{axis}/{tag}/loss_log_{tag}.csv") as f:
        for r in csv.DictReader(f):
            its.append(int(r["iter"]))
            val.append(float(r["val_loss"]))
    return its, val


def seed_sigma():
    with open(f"{RESULTS}/seeds/summary.csv") as f:
        rows = list(csv.DictReader(f))
    return float(next(r["final_val_ppl"] for r in rows if r["seed"] == "std")), \
           float(next(r["final_val_ppl"] for r in rows if r["seed"] == "mean"))


def fig_ablations():
    sigma, mean = seed_sigma()
    axes_ = ["context", "depth", "learning_rate", "weight_tying"]
    titles = ["Context length", "Depth", "Learning rate", "Weight tying"]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, axis, title in zip(axs, axes_, titles):
        rows = read_summary(axis)
        tags = [r["tag"] for r in rows]
        ppls = [float(r["final_val_ppl"]) for r in rows]
        best = min(ppls)
        # colour by whether the gap to the best config exceeds 3 sigma
        cols = ["#4c72b0" if abs(p - best) < 3 * sigma else "#c44e52" for p in ppls]
        ax.bar(tags, ppls, color=cols)
        # the noise band, drawn around the best config
        ax.axhspan(best - sigma, best + sigma, color="grey", alpha=0.25, zorder=0)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("val perplexity" if axis == "context" else "")
        ax.set_ylim(min(ppls) - 0.6, max(ppls) + 0.4)
        ax.grid(alpha=0.3, axis="y")
        for i, p in enumerate(ppls):
            ax.text(i, p + 0.06, f"{p:.2f}", ha="center", fontsize=9)

    fig.suptitle(f"Ablations at 5000 iters — grey band = ±1σ seed noise "
                 f"(σ = {sigma:.2f} PPL); red = >3σ from best", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ablations.png", dpi=150)
    print(f"saved {OUT}/fig_ablations.png")


def fig_tying_budget():
    it_u, v_u = read_curve("weight_tying", "untied")
    it_t, v_t = read_curve("weight_tying", "tied")
    assert it_u == it_t

    ppl_u = [math.exp(v) for v in v_u]
    ppl_t = [math.exp(v) for v in v_t]
    ratio = [t / u for t, u in zip(ppl_t, ppl_u)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    axs[0].plot(it_u, ppl_u, "-", color="#4c72b0", label="untied")
    axs[0].plot(it_t, ppl_t, "-", color="#c44e52", label="tied")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("val perplexity (log scale)")
    axs[0].set_title("Tied vs untied: both converge")
    axs[0].legend()
    axs[0].grid(alpha=0.3, which="both")

    axs[1].plot(it_u, ratio, "o-", color="#c44e52")
    axs[1].axhline(1.0, ls="--", color="grey", lw=1)
    axs[1].set_xlabel("iteration")
    axs[1].set_ylabel("tied PPL / untied PPL")
    axs[1].set_title("The penalty is a function of training budget")
    axs[1].grid(alpha=0.3)
    for i in (1, 5, 20):
        axs[1].annotate(f"{ratio[i]:.2f}x", (it_u[i], ratio[i]),
                         textcoords="offset points", xytext=(6, 8), fontsize=9)

    fig.suptitle("Weight tying: catastrophic early, minor at convergence "
                 "(one boolean flag, everything else identical)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tying_budget.png", dpi=150)
    print(f"saved {OUT}/fig_tying_budget.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_ablations()
    fig_tying_budget()
