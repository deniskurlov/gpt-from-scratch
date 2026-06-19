import torch

from jaxtyping import Int64, Float32
from torch import nn, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, V: int, d_model: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(V, d_model)

    def forward(self, ids: Int64[Tensor, "B T"]) -> Float32[Tensor, "B T d_model"]:
        return self.tok_emb(ids)


class LearnedPositionalEmbedding(nn.Module):
    """Legacy reference; not used by the current GPT (which uses RoPE)."""
    def __init__(self, T_max: int, d_model: int) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(T_max, d_model)

    def forward(self, positions: Int64[Tensor, "T"]) -> Float32[Tensor, "T d_model"]:  # noqa: F821
        return self.pos_emb(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Legacy reference; not used by the current GPT (which uses RoPE)."""
    def __init__(self, d_model: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = base ** (-2 * torch.arange(d_model // 2) / d_model)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, positions: Int64[Tensor, "T"]) -> Float32[Tensor, "T d_model"]:  # noqa: F821
        angles = torch.outer(positions, self.inv_freq)
        sin, cos = angles.sin(), angles.cos()

        pos_emb = torch.stack([sin, cos], dim=-1).flatten(-2)

        return pos_emb


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, f'RoPE error: head_dim={head_dim} is not even.'
        inv_freq = base ** (-2 * torch.arange(head_dim // 2) / head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
    def forward(self, x: Float32[Tensor, "B n_heads T_new head_dim"], start_pos):
        B, n_heads, T_new, head_dim = x.shape

        positions = torch.arange(start_pos, start_pos + T_new, device=x.device).float()
        angles = torch.outer(positions, self.inv_freq)
        cos, sin = angles.cos(), angles.sin()

        x_pairs = x.view(B, n_heads, T_new, head_dim // 2, 2)
        a, b = x_pairs[..., 0], x_pairs[..., 1]
        a_rot = a * cos - b * sin
        b_rot = a * sin + b * cos
        x_rot = torch.stack([a_rot, b_rot], dim=-1).flatten(-2)
        
        return x_rot


if __name__ == '__main__':
    rope = RoPE(head_dim=32)                                                                
    q = torch.randn(1, 1, 1, 32)                                                                      
    k = torch.randn(1, 1, 1, 32)
                                                                                                        
    q_2 = rope(q, start_pos=2)                                                                        
    k_5 = rope(k, start_pos=5)
    score_25 = (q_2 * k_5).sum()                                                                      
                                                                
    q_7 = rope(q, start_pos=7)
    k_10 = rope(k, start_pos=10)
    score_710 = (q_7 * k_10).sum()                                                                    
    
    # Both pairs have m - n = -3, so scores should match                                              
    print(torch.allclose(score_25, score_710, atol=1e-5))
