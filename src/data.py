# src/data.py
import torch 


def load_corpus() -> str:
    with open('data/input.txt', 'r', encoding='utf-8') as f:
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

    def encode_to_tensor(self, text: str) -> torch.LongTensor:
        return torch.tensor(self.encode(text), dtype=torch.long)


if __name__ == '__main__':
    text = load_corpus()
    tok = Tokenizer(text)

    print(tok.encode_to_tensor(text[:80]))