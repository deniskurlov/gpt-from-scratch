# src/model.py

import torch

from jaxtyping import Int64, Float32
from torch import Tensor, nn

from src.data import load_corpus, Tokenizer, TokenizedDataset


class TokenEmbedding(nn.Module):
    def __init__(self, V: int, d_model: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(V, d_model)
    def forward(self, ids: Int64[Tensor, "B T"]) -> Float32[Tensor, "B T d_model"]:
        return self.tok_emb(ids)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, T_max: int, d_model: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(T_max, d_model)
    def forward(self, positions: Int64[Tensor, "T"]) -> Float32[Tensor, "T d_model"]:
        return self.pos_emb(positions) 


if __name__ == '__main__':
    torch.manual_seed(42)

    text = load_corpus()
    tok = Tokenizer(text)
    ds = TokenizedDataset(tok.encode_to_tensor(text))

    emb = TokenEmbedding(V=tok.vocab_size, d_model=128)
    pos_emb = LearnedPositionalEmbedding(T_max=256, d_model=128)

    B, T = 2, 4
    
    x = ds.get_batch(B, T)[0]
    out = emb(x) + pos_emb(torch.arange(T))

    print(out.shape, out.dtype)