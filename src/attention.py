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

class MultiHeadAttention(nn.Module):
    def __init__(self, T_max: int, n_heads: int, d_model: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.register_buffer('mask', torch.tril(torch.ones(T_max, T_max)).bool())
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: Float32[Tensor, "B T d_model"]) -> Float32[Tensor, "B T d_model"]:
        B, T = x.shape[:2]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.d_model, self.d_model, self.d_model], dim=-1)  # each (B, T, d_model)
        q, k, v = (
            p.view(B, T, self.n_heads, self.head_dim).transpose(-2, -3) for p in (q, k, v)
            )  # each (B, n_heads, T, head_dim)
        scores = q @ k.transpose(-1, -2) / sqrt(self.head_dim)  # (B, n_heads, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        out = attn @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(-2, -3).reshape(B, T, self.d_model)  # (B, T, d_model)
        output = self.out_proj(out)
        return output


if __name__ == '__main__':
    torch.manual_seed(42)

    # attn = Attention(T_max=256, d_k=128, d_v=128, d_model=128)
    # x = torch.randn(2, 4, 128)
    # out = attn(x)

    mha = MultiHeadAttention(T_max=256, n_heads=4, d_model=128)
    x = torch.randn(2, 4, 128)
    out = mha(x)



    print(out.shape, out.dtype)