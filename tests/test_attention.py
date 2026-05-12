import torch

from  src.attention import Attention


def test_attn_shape_and_type():
    torch.manual_seed(42)
    T_max = 256
    B, T, d_model = 6, 10, 128
    x = torch.randn(B, T, d_model)
    attn = Attention(T_max=T_max, d_k=d_model, d_v=d_model, d_model=d_model)
    out = attn(x)
    assert out.shape == (B, T, d_model) and out.dtype == torch.float32

def test_qkv_parameter_count():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    d_k=d_model // 4
    d_v=d_model // 2
    attn = Attention(T_max=T_max, d_k=d_k, d_v=d_v, d_model=d_model)
    qkv = attn.qkv_proj
    assert qkv.weight.numel() == d_model * (2*d_k + d_v) and qkv.bias.numel() == 2*d_k + d_v

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
    assert sum([p.numel() for p in attn.parameters()]) == total_attn_param_num

def test_attn_causality():
    torch.manual_seed(42)
    T_max = 256
    B, T, d_model = 6, 10, 128
    x = torch.randn(B, T, d_model)
    attn = Attention(T_max=T_max, d_k=d_model, d_v=d_model, d_model=d_model)
    out = attn(x)
    
    x_modified = x.clone()
    x_modified[:, T-1, :] = torch.randn(B, d_model) * 100
    out_modified = attn(x_modified)

    assert torch.equal(out[:, :T-1, :], out_modified[:, :T-1, :])
    

def test_attn_mask_is_non_parameter():
    torch.manual_seed(42)
    T_max, d_model = 256, 128
    d_k=d_model // 2
    d_v=d_model // 4
    attn = Attention(T_max=T_max, d_k=d_k, d_v=d_v, d_model=d_model)
    assert 'mask' in attn.state_dict() and not any(p is attn.mask for p in attn.parameters())