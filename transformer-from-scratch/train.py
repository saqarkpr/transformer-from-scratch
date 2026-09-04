"""
Training loop for the from-scratch decoder-only Transformer.

Usage:
    python train.py --n_layers 4 --n_heads 4 --d_model 128 --block_size 128 \
        --batch_size 64 --lr 3e-4 --max_iters 5000

Logs train/val loss to a CSV (loss_log.csv) so the README experiment
tables (context length / n_layers / lr) can be filled in after runs.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch

from data import download_corpus, CharTokenizer, prepare_dataset, get_batch
from model import TransformerFromScratch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data, block_size, batch_size, device, eval_iters=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--block_size", type=int, default=128, help="context length")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--tag", type=str, default="run1", help="experiment tag, used in log filenames")
    p.add_argument("--seed", type=int, default=42, help="random seed, for reproducibility / seed-variance study")
    p.add_argument("--tie_weights", action="store_true", help="tie lm_head weights to the token embedding")
    return p


def run_training(args) -> dict:
    """Runs one full training job from a populated args namespace and
    returns a summary dict. Factored out of main() so ablation/seed-sweep
    scripts can call this directly (in-process, no subprocess overhead)
    instead of duplicating the training loop."""
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    text = download_corpus()
    tok = CharTokenizer(text)
    train_data, val_data = prepare_dataset(text, tok)

    model = TransformerFromScratch(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.block_size,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M | device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = os.path.join(args.out_dir, f"loss_log_{args.tag}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "val_loss", "elapsed_s"])

    t0 = time.time()
    for it in range(1, args.max_iters + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0 or it == 1:
            val_loss = evaluate(model, val_data, args.block_size, args.batch_size, device)
            elapsed = time.time() - t0
            print(f"iter {it:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | {elapsed:.1f}s")
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([it, loss.item(), val_loss, elapsed])

    ckpt_path = os.path.join(args.out_dir, f"model_{args.tag}.pt")
    torch.save(
        {"model_state": model.state_dict(), "vocab": tok.stoi, "args": vars(args)},
        ckpt_path,
    )

    final_val_loss = evaluate(model, val_data, args.block_size, args.batch_size, device, eval_iters=50)
    final_val_ppl = math.exp(final_val_loss)
    total_elapsed = time.time() - t0

    summary = {
        "tag": args.tag,
        "seed": args.seed,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "block_size": args.block_size,
        "lr": args.lr,
        "tie_weights": args.tie_weights,
        "max_iters": args.max_iters,
        "n_params": n_params,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "training_time_s": total_elapsed,
        "checkpoint": ckpt_path,
        "loss_log": log_path,
    }
    summary_path = os.path.join(args.out_dir, f"summary_{args.tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved checkpoint to {ckpt_path}")
    print(f"loss log written to {log_path}")
    print(f"final val_loss {final_val_loss:.4f} | val_ppl {final_val_ppl:.2f} | "
          f"params {n_params/1e6:.2f}M | time {total_elapsed:.1f}s")
    return summary


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
