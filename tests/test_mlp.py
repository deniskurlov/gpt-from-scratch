import pytest
import torch

from src.mlp import MLP


def test_mlp_shape_and_type():
    d_model = 128
    mlp = MLP(d_model=d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    x_out = mlp(x)

    assert x_out.shape == (B, T, d_model) and x_out.dtype == torch.float32

@pytest.mark.parametrize("d_ff", [m * 128 for m in (1, 2, 3, 4, 6)])
@pytest.mark.parametrize("bias", [False, True])
def test_mlp_parameter_count(d_ff, bias):
    d_model = 128
    mlp = MLP(d_model=d_model, d_ff=d_ff, bias=bias)

    param_count = 2 * d_model * d_ff + d_ff + d_model if bias else 2 * d_model * d_ff

    assert sum(p.numel() for p in mlp.parameters()) == param_count

def test_mlp_nonlinearity_applied():
    d_model = 128
    mlp = MLP(d_model=d_model, bias=False)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    a = 3
    assert not torch.allclose(mlp(a * x), a * mlp(x))
