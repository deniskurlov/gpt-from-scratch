import torch
import torch.nn.functional as F  

from jaxtyping import Float32
from torch import Tensor, nn


class MLP(nn.Module):

    def __init__(self, d_model: int, d_ff: int | None = None, bias: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4*d_model
        self.up_proj = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.down_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
    
    def forward(self, x: Float32[Tensor, "B T d_model"]) -> Float32[Tensor, "B T d_model"]:
        return self.down_proj(F.gelu(self.up_proj(x)))


if __name__ == '__main__':
    d_model = 128
    mlp = MLP(d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out = mlp(x)

    print(x_out.shape)
