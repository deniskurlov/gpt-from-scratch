import pytest
import torch

from src.data import TokenizedDataset
from src.embedding import LearnedPositionalEmbedding, TokenEmbedding


def test_token_embedding_shape(tok, text):
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T, d_model = 100, 20, 128
    x, _ = ds.get_batch(B, T)
    V = tok.vocab_size
    emb = TokenEmbedding(V=V, d_model=d_model)
    assert emb(x).shape == (B, T, d_model)


def test_token_embedding_parameter_count(tok):
    d_model = 128
    V = tok.vocab_size
    emb = TokenEmbedding(V=V, d_model=d_model)
    assert sum(p.numel() for p in emb.parameters()) == V * d_model


@pytest.mark.parametrize("T", [1, 8, 32, 128, 256])
def test_learned_positional_embedding_shape(T):
    T_max = 256
    d_model = 128
    pos_emb = LearnedPositionalEmbedding(T_max=T_max, d_model=d_model)
    assert pos_emb(torch.arange(T)).shape == (T, d_model)


def test_learned_positional_embedding_param_count():
    T_max = 256
    d_model = 128
    pos_emb = LearnedPositionalEmbedding(T_max=T_max, d_model=d_model)
    assert sum(p.numel() for p in pos_emb.parameters()) == T_max * d_model


def test_learned_positional_embedding_out_of_range():
    T_max = 256
    d_model = 128
    pos_emb = LearnedPositionalEmbedding(T_max, d_model)
    with pytest.raises(IndexError):
        pos_emb(torch.tensor([T_max]))
