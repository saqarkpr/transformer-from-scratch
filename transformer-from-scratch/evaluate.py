"""
Loads a trained checkpoint and reports the standard metrics block:
parameter count, validation loss, perplexity, and (if a matching loss log
is found next to the checkpoint) training time. Also accepts a summary_*.json
written by train.py directly, which is faster since it skips recomputation.

Usage:
    python evaluate.py --ckpt checkpoints/model_baseline.pt
    python evaluate.py --summary checkpoints/summary_baseline.json
"""
import argparse
import json
import math
import os

import torch

from data import download_corpus, CharTokenizer, prepare_dataset, get_batch
from model import TransformerFromScratch


@torch.no_grad()
def compute_val_loss(model, val_data, block_size, batch_size, device, eval_iters=100):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(val_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def print_summary(d: dict):
    print("```")
    print(f"Model:        Decoder-only Transformer")
    print(f"Parameters:   {d['n_params']:,}")
    print(f"Layers:       {d.get('n_layers', '?')}")
    print(f"Heads:        {d.get('n_heads', '?')}")
    print(f"d_model:      {d.get('d_model', '?')}")
    print(f"Context:      {d.get('block_size', '?')}")
    print(f"Tied weights: {d.get('tie_weights', False)}")
    print(f"Seed:         {d.get('seed', '?')}")
    print(f"Dataset:      Tiny Shakespeare")
    print(f"Tokenizer:    Character-level")
    print(f"Iterations:   {d.get('max_iters', '?')}")
    print(f"Val Loss:     {d['final_val_loss']:.4f}")
    print(f"Perplexity:   {d['final_val_ppl']:.2f}")
    if "training_time_s" in d:
        print(f"Training:     {d['training_time_s']:.1f}s")
    print("```")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="path to model_*.pt checkpoint")
    p.add_argument("--summary", type=str, default=None, help="path to summary_*.json (skips recomputation)")
    p.add_argument("--eval_iters", type=int, default=100)
    args = p.parse_args()

    if args.summary:
        with open(args.summary) as f:
            d = json.load(f)
        print_summary(d)
        return

    if not args.ckpt:
        raise SystemExit("provide either --ckpt or --summary")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    stoi = ckpt["vocab"]

    text = download_corpus()
    tok = CharTokenizer(text)
    assert tok.stoi == stoi, "tokenizer vocab in checkpoint doesn't match current corpus tokenizer"
    _, val_data = prepare_dataset(text, tok)

    model = TransformerFromScratch(
        vocab_size=tok.vocab_size,
        d_model=margs["d_model"], n_heads=margs["n_heads"], n_layers=margs["n_layers"],
        max_len=margs["block_size"], dropout=0.0, tie_weights=margs.get("tie_weights", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p_.numel() for p_ in model.parameters())
    val_loss = compute_val_loss(model, val_data, margs["block_size"], margs.get("batch_size", 64), device, args.eval_iters)
    val_ppl = math.exp(val_loss)

    d = {**margs, "n_params": n_params, "final_val_loss": val_loss, "final_val_ppl": val_ppl}
    print_summary(d)


if __name__ == "__main__":
    main()
