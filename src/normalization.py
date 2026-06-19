import torch

from jaxtyping import Float32
from torch import nn, Tensor


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.bias = bias
        self.gamma = nn.Parameter(torch.ones(d_model))
        if bias:
            self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(
        self, x: Float32[Tensor, "B T d_model"]
    ) -> Float32[Tensor, "B T d_model"]:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(
            x, dim=-1, keepdim=True, unbiased=False
        )  # unbiased=False to match nn.LayerNorm
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias:
            y = self.gamma * x_norm + self.beta
        else:
            y = self.gamma * x_norm
        return y


if __name__ == "__main__":
    d_model = 128
    LN = LayerNormalization(d_model=d_model)
    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    x_out = LN(x)

    print(x.shape, x_out.shape)
    print(x.type, x_out.dtype)
    print(torch.mean(x_out, dim=-1, keepdim=True))
