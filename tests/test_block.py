import torch

from src.attention import MultiHeadAttention
from src.mlp import MLP
from src.model import Block
from src.normalization import LayerNormalization


def test_block_shape_and_type():
    T_max, n_heads, d_model = 256, 4, 128
    block = Block(T_max=T_max, n_heads=n_heads, d_model=d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out = block(x)

    assert x_out.shape == (B, T, d_model) and x_out.dtype == torch.float32

def test_block_causality():
    T_max, n_heads, d_model = 256, 4, 128
    block = Block(T_max=T_max, n_heads=n_heads, d_model=d_model)
    block.eval()

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out = block(x)

    x_modified = x.clone()
    x_modified[:, -1, :] = 100 * torch.randn(B, d_model)

    x_modified_out = block(x_modified)

    assert torch.equal(x_out[:, :T-1, :], x_modified_out[:, :T-1, :])

def test_block_parameter_count():
    T_max, n_heads, d_model = 256, 4, 128
    d_ff = 6 * d_model
    block = Block(T_max=T_max, n_heads=n_heads, d_model=d_model, d_ff=d_ff)

    ln1_param_count = 2 * d_model
    ln2_param_count = 2 * d_model
    attn_param_count = 4 * d_model * (d_model + 1)
    mlp_param_count = 2 * d_ff * d_model
    total_param_count = ln1_param_count + ln2_param_count + attn_param_count + mlp_param_count

    assert sum(p.numel() for p in block.parameters()) == total_param_count

def test_block_residual_structure_at_p0():

    T_max, n_heads, d_model = 256, 4, 128
    d_ff = 6 * d_model
    block = Block(T_max=T_max, n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=0.0)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    expected = x + block.attn(block.ln1(x))
    expected = expected + block.mlp(block.ln2(expected))

    assert torch.allclose(block(x), expected)

def test_block_dropout_train_vs_eval():
    T_max, n_heads, d_model = 256, 4, 128
    block = Block(T_max=T_max, n_heads=n_heads, d_model=d_model)

    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    block.train()
    assert not torch.allclose(block(x), block(x))

    block.eval()
    assert torch.allclose(block(x), block(x))
