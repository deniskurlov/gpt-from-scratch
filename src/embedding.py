from jaxtyping import Int64, Float32
from torch import nn, Tensor


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
