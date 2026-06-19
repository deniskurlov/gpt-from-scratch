import pytest
import torch

from src.data import TokenizedDataset
from math import log
from src.model import GPT


def test_model_shape_and_type(tok, text):
    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6
    rope_base=10_000.0
    d_ff=None
    dropout=0.1

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=rope_base, d_ff=d_ff, dropout=dropout
        )
    
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 2, 4
    x, _ = ds.get_batch(B, T)
    logits = gpt(x)
    assert x.shape == (B, T) and logits.shape == (B, T, V)
    assert x.dtype == torch.int64 and logits.dtype == torch.float32

@pytest.mark.parametrize("return_loss", [True, False])
def test_model_return_logits_loss(tok, text, return_loss):
    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6
    rope_base=10_000.0
    d_ff=None
    dropout=0.1

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=rope_base, d_ff=d_ff, dropout=dropout
        )
    
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 2, 4
    x, y = ds.get_batch(B, T)
    
    if return_loss:
        _, loss = gpt(x, targets=y)
        assert loss.shape == torch.Size([])
        assert all(p.grad is None for p in gpt.parameters())
        loss.backward()
        assert all(p.grad is not None for p in gpt.parameters())
    else:
        assert gpt(x).shape == (B, T, V)

def test_model_initial_loss(tok, text):
    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6
    rope_base=10_000.0
    d_ff=None
    dropout=0.1

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=rope_base, d_ff=d_ff, dropout=dropout
        )
    
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 2, 4
    x, y = ds.get_batch(B, T)

    _, loss = gpt(x, targets=y)

    assert 0.8 * log(V) <= loss.item() <= 1.2 * log(V)

@pytest.mark.parametrize("d_ff", [128, 256, 512, 1024])
@pytest.mark.parametrize("n_layers", [4, 5, 6, 8])
def test_model_param_count(tok, d_ff, n_layers):
    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    rope_base=10_000.0
    dropout=0.1

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=rope_base, d_ff=d_ff, dropout=dropout
        )
    
    tok_emb_param_count = V * d_model
    # pos_emb_param_count = T_max * d_model
    pos_emb_param_count = 0  #  using RoPE -- no learned params
    attn_param_count = 4 * d_model ** 2 + 4 * d_model
    mlp_param_count = 2 * d_model * d_ff
    ln1_param_count = 2 * d_model
    ln2_param_count = 2 * d_model
    layer_param_count = (
        attn_param_count + mlp_param_count + ln1_param_count + ln2_param_count
    )
    final_ln_param_count = 2 * d_model
    lm_head_param_count = 0  # tied embedding -- no new params

    model_param_count = (
        tok_emb_param_count + pos_emb_param_count + n_layers * layer_param_count 
        + final_ln_param_count + lm_head_param_count
    )

    assert sum(p.numel() for p in gpt.parameters()) == model_param_count

def test_model_causality(tok, text):
    
    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6
    rope_base=10_000.0
    d_ff=None
    dropout=0.1

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=rope_base, d_ff=d_ff, dropout=dropout
        )
    gpt.eval()  # prevent random dropouts 

    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 2, 4
    x, _ = ds.get_batch(B, T)

    x_modified = x.clone()
    x_modified[:, -1] = (x[:, -1] + torch.randint(low=0, high=V, size=(B,))) % V

    logits = gpt(x)
    logits_modified = gpt(x_modified)

    assert torch.equal(logits[:, :T-1, :], logits_modified[:, :T-1, :])