"""
Sample text from a trained checkpoint.

Supports multiple temperatures in one call for a systematic qualitative
comparison (e.g. "effect of sampling temperature on generation quality"),
instead of eyeballing one sample at a time:

    python generate.py --ckpt checkpoints/model_baseline.pt --prompt "ROMEO:" \
        --temperature 0.7 1.0 1.3 --max_new_tokens 200 --seed 42
"""
import argparse
import torch

from model import TransformerFromScratch


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    stoi = ckpt["vocab"]
    itos = {i: ch for ch, i in stoi.items()}
    margs = ckpt["args"]

    model = TransformerFromScratch(
        vocab_size=len(stoi),
        d_model=margs["d_model"],
        n_heads=margs["n_heads"],
        n_layers=margs["n_layers"],
        max_len=margs["block_size"],
        dropout=0.0,
        tie_weights=margs.get("tie_weights", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stoi, itos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, nargs="+", default=[0.8],
                    help="one or more temperatures; generates one sample per value")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None, help="fix the sampling seed for reproducible comparisons")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.ckpt, device)
    idx = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=device)

    for temp in args.temperature:
        if args.seed is not None:
            torch.manual_seed(args.seed)  # same seed per temperature -> differences are due to temperature, not noise
        out = model.generate(idx, args.max_new_tokens, temperature=temp, top_k=args.top_k)
        text = "".join(itos[i] for i in out[0].tolist())
        if len(args.temperature) > 1:
            print(f"\n=== temperature = {temp} ===")
        print(text)


if __name__ == "__main__":
    main()
