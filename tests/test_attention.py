import torch

from  src.attention import Attention, MultiHeadAttention


def test_attn_shape_and_type():
    torch.manual_seed(42)
    T_max = 256
    B, T, d_model = 6, 10, 128
    x = torch.randn(B, T, d_model)
    attn = Attention(T_max=T_max, d_k=d_model, d_v=d_model, d_model=d_model)
    out = attn(x)
    assert out.shape == (B, T, d_model) and out.dtype == torch.float32

def test_mha_shape_and_type():
    torch.manual_seed(42)
    B, T, d_model = 6, 10, 128
    n_heads = 4
    rope_base = 10_000.0
    x = torch.randn(B, T, d_model)
    attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model, rope_base=rope_base)
    out, _ = attn(x)
    assert out.shape == (B, T, d_model) and out.dtype == torch.float32

def test_attn_qkv_parameter_count():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    d_k=d_model // 4
    d_v=d_model // 2
    attn = Attention(T_max=T_max, d_k=d_k, d_v=d_v, d_model=d_model)
    qkv = attn.qkv_proj
    assert qkv.weight.numel() == d_model * (2*d_k + d_v) and qkv.bias.numel() == 2*d_k + d_v

def test_mha_qkv_parameter_count():
    torch.manual_seed(42)
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, rope_base=rope_base)
    qkv = mha.qkv_proj
    assert sum(p.numel() for p in qkv.parameters()) == 3 * d_model * (d_model + 1)  

def test_attn_total_parameter_count():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    d_k=d_model // 4
    d_v=d_model // 2
    attn = Attention(T_max=T_max, d_k=d_k, d_v=d_v, d_model=d_model)
    qkv_proj_weight_num = d_model * (2*d_k + d_v)
    qkv_proj_bias_num = 2*d_k + d_v
    out_proj_weight_num = d_v * d_model 
    out_proj_bias_num = d_model
    total_attn_param_num = (
        qkv_proj_weight_num + qkv_proj_bias_num + out_proj_weight_num + out_proj_bias_num
    )
    assert sum(p.numel() for p in attn.parameters()) == total_attn_param_num

def test_mha_total_parameter_count():
    torch.manual_seed(42)
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, rope_base=rope_base)
    assert sum(p.numel() for p in mha.parameters()) == 4 * d_model * (d_model + 1)  

def test_attn_causality():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    attn = Attention(T_max=T_max, d_k=d_model, d_v=d_model, d_model=d_model)
    
    B, T = 6, 10
    x = torch.randn(B, T, d_model)
    out = attn(x)
    x_modified = x.clone()
    x_modified[:, T-1, :] = 100 * torch.randn(B, d_model)
    out_modified = attn(x_modified)

    assert torch.equal(out[:, :T-1, :], out_modified[:, :T-1, :])

def test_mha_causality():
    torch.manual_seed(42)
    d_model, n_heads, rope_base = 256, 4, 10_000.0
    mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, rope_base=rope_base)

    B, T  = 6, 10
    x = torch.randn(B, T, d_model)
    out, _ = mha(x)
    x_modified = x.clone()
    x_modified[:, T-1, :] = 100 * torch.randn(B, d_model) 
    out_modified, _ = mha(x_modified)

    assert torch.equal(out[:, :T-1, :], out_modified[:, :T-1, :])
    
def test_attn_mask_is_non_parameter():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    d_k=d_model // 2
    d_v=d_model // 4
    attn = Attention(T_max=T_max, d_k=d_k, d_v=d_v, d_model=d_model)
    assert 'mask' in attn.state_dict() and not any(p is attn.mask for p in attn.parameters())
