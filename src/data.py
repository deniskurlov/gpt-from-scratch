# src/data.py
import torch

from jaxtyping import Int64
from torch import Tensor


def load_corpus() -> str:
    with open("data/input.txt", "r", encoding="utf-8") as f:
        return f.read()


class Tokenizer:
    def __init__(self, corpus: str) -> None:
        vocab: list[str] = sorted(set(corpus))
        self.vocab = vocab
        self.vocab_size: int = len(vocab)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.vocab[t] for t in tokens])

    def encode_to_tensor(self, text: str) -> Int64[Tensor, "L"]:  # noqa: F821
        return torch.tensor(self.encode(text), dtype=torch.long)


class TokenizedDataset:
    def __init__(self, encoded: Int64[Tensor, "L"]) -> None:  # noqa: F821
        self.encoded = encoded

    def get_batch(
        self, B: int, T: int
    ) -> tuple[Int64[Tensor, "B T"], Int64[Tensor, "B T"]]:
        L = len(self.encoded)
        offsets = torch.randint(0, L - T, (B,))
        idx = offsets[:, None] + torch.arange(T)[None, :]
        x = self.encoded[idx]
        y = self.encoded[idx + 1]
        return x, y


if __name__ == "__main__":
    # torch.manual_seed(42)

    text = load_corpus()
    tok = Tokenizer(text)
    ds = TokenizedDataset(tok.encode_to_tensor(text))

    print(ds.get_batch(B=2, T=4))
