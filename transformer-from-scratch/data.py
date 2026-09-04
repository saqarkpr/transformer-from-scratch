"""
Character-level tokenizer + batching for a small text corpus (tiny Shakespeare
by default). Character-level keeps the vocabulary tiny (~65 symbols) so the
project stays about the Transformer, not about a tokenizer.
"""
import os
import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "input.txt")


def download_corpus(path: str = LOCAL_PATH, url: str = DATA_URL) -> str:
    if not os.path.exists(path):
        import urllib.request

        print(f"Downloading corpus from {url} ...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def prepare_dataset(text: str, tokenizer: CharTokenizer, split: float = 0.9):
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(split * len(ids))
    return ids[:n], ids[n:]


if __name__ == "__main__":
    text = download_corpus()
    tok = CharTokenizer(text)
    print("corpus length:", len(text), "vocab size:", tok.vocab_size)
    train_data, val_data = prepare_dataset(text, tok)
    xb, yb = get_batch(train_data, block_size=32, batch_size=4)
    print("batch shapes:", xb.shape, yb.shape)
    print("sample decoded:", tok.decode(xb[0].tolist()))
