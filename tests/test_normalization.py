import torch

from src.normalization import LayerNormalization


def test_layer_norm_shape_and_type():
    torch.manual_seed(42)

    d_model = 128
    LN = LayerNormalization(d_model=d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out = LN(x)

    assert x_out.shape == torch.Size([B, T, d_model]) and x_out.dtype == torch.float32

def test_layer_norm_mean_and_var():
    torch.manual_seed(42)

    d_model = 128
    LN = LayerNormalization(d_model=d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out = LN(x)

    assert (
        torch.allclose(torch.mean(x_out, dim=-1), torch.zeros(B, T), atol=1e-6)
        and
        torch.allclose(torch.var(x_out, dim=-1, unbiased=False), torch.ones(B, T), atol=1e-5)
    )

def test_layer_norm_param_count():
    d_model = 128
    LN_with_bias = LayerNormalization(d_model=d_model, bias=True)
    LN_no_bias = LayerNormalization(d_model=d_model, bias=False)

    assert (
        sum(p.numel() for p in LN_with_bias.parameters()) == 2*d_model
        and
        sum(p.numel() for p in LN_no_bias.parameters()) == d_model
    )