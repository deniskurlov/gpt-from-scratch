import torch

from src.data import TokenizedDataset


def test_encoding_roundtrip(text, tok):
    assert tok.decode(tok.encode(text)) == text

def test_vocab_size(tok, text):
    assert tok.vocab_size == len(set(text))

def test_vocab(tok, text):
    assert tok.vocab == sorted(set(text))

def test_batch_size(tok, text):
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 100, 20
    x, y = ds.get_batch(B, T)
    assert x.shape == (B, T) and y.shape == (B, T)

def test_batch_type(tok, text):
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 100, 20
    x,y = ds.get_batch(B, T)
    assert x.dtype == torch.long and y.dtype == torch.long

def test_batch_indices_range(tok, text):
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    B, T = 100, 20
    x, y = ds.get_batch(B, T)
    V = tok.vocab_size
    assert ((x >= 0) & (x < V) & (y >= 0) & (y < V)).all()

def test_batch_shift_by_1_invariant(tok, text):
    ds = TokenizedDataset(tok.encode_to_tensor(text))
    x, y = ds.get_batch(B=100, T=20)
    assert torch.all(y[:, :-1] == x[:, 1:])
