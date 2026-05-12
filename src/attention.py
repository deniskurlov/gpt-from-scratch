import torch
import torch.nn.functional as F 

from math import sqrt
from jaxtyping import Float32
from torch import nn, Tensor


class Attention(nn.Module):
    def __init__(self, T_max: int, d_k: int, d_v: int, d_model: int) -> None:
        super().__init__()
        self.register_buffer('mask', torch.tril(torch.ones(T_max, T_max)).bool())
        self.d_k = d_k
        self.d_v = d_v
        self.qkv_proj = nn.Linear(d_model, 2*d_k + d_v)
        self.out_proj = nn.Linear(d_v, d_model)
    def forward(self, x: Float32[Tensor, "B T d_model"]) -> Float32[Tensor, "B T d_model"]:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.d_k, self.d_k, self.d_v], dim=-1)
        scores = q @ k.transpose(-1, -2) / sqrt(self.d_k)
        T = x.shape[1]
        scores = scores.masked_fill(self.mask[:T, :T].logical_not(), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        output = self.out_proj(out)
        return output

if __name__ == '__main__':
    torch.manual_seed(42)

    attn = Attention(T_max=256, d_k=128, d_v=128, d_model=128)
    x = torch.randn(2, 4, 128)
    out = attn(x)

    print(out.shape, out.dtype)