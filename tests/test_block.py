import torch

from src.attention import MultiHeadAttention
from src.mlp import MLP
from src.model import Block
from src.normalization import LayerNormalization


def test_block_shape_and_type():
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    block = Block(
        n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=None, dropout=0.1
        )

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out, _ = block(x)

    assert x_out.shape == (B, T, d_model) and x_out.dtype == torch.float32

def test_block_causality():
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    block = Block(
        n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=None, dropout=0.1
        )
    block.eval()

    B, T = 2, 4
    x = torch.randn(B, T, d_model)
    x_out, _ = block(x)

    x_modified = x.clone()
    x_modified[:, -1, :] = 100 * torch.randn(B, d_model)

    x_modified_out, _ = block(x_modified)

    assert torch.equal(x_out[:, :T-1, :], x_modified_out[:, :T-1, :])

def test_block_parameter_count():
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    d_ff = 6 * d_model
    block = Block(
        n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=d_ff, dropout=0.1
        )

    ln1_param_count = 2 * d_model
    ln2_param_count = 2 * d_model
    attn_param_count = 4 * d_model * (d_model + 1)
    mlp_param_count = 2 * d_ff * d_model
    total_param_count = ln1_param_count + ln2_param_count + attn_param_count + mlp_param_count

    assert sum(p.numel() for p in block.parameters()) == total_param_count

def test_block_residual_structure_at_p0():

    d_model, n_heads, rope_base = 256, 4, 10_000.0
    d_ff = 6 * d_model
    block = Block(
        n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=d_ff, dropout=0.0
        )

    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    expected = x + block.attn(block.ln1(x))[0]
    expected = expected + block.mlp(block.ln2(expected))

    assert torch.allclose(block(x)[0], expected)

def test_block_dropout_train_vs_eval():
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    block = Block(
        n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=None, dropout=0.1
        )

    B, T = 2, 4
    x = torch.randn(B, T, d_model)

    block.train()
    assert not torch.allclose(block(x)[0], block(x)[0])

    block.eval()
    assert torch.allclose(block(x)[0], block(x)[0])
