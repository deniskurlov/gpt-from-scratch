# src/model.py

import torch

from jaxtyping import Float32
from torch import Tensor, nn

from src.attention import MultiHeadAttention
# from src.embedding import LearnedPositionalEmbedding, TokenEmbedding
# from src.data import load_corpus, Tokenizer, TokenizedDataset
from src.normalization import LayerNormalization
from src.mlp import MLP


class Block(nn.Module):
    def __init__(self, T_max: int, n_heads: int, d_model: int, 
                 d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)
        self.attn = MultiHeadAttention(T_max=T_max, n_heads=n_heads, d_model=d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: Float32[Tensor, "B T d_model"]) -> Float32[Tensor, "B T d_model"]:
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x

if __name__ == '__main__':
    # torch.manual_seed(42)

    # text = load_corpus()
    # tok = Tokenizer(text)
    # ds = TokenizedDataset(tok.encode_to_tensor(text))

    # emb = TokenEmbedding(V=tok.vocab_size, d_model=128)
    # pos_emb = LearnedPositionalEmbedding(T_max=256, d_model=128)

    # B, T = 2, 4
    
    # x = ds.get_batch(B, T)[0]
    # out = emb(x) + pos_emb(torch.arange(T))

    # print(out.shape, out.dtype)
    torch.manual_seed(42)                                                              
    block = Block(T_max=256, n_heads=4, d_model=128)
    x = torch.randn(2, 4, 128)                                                         
    out = block(x)                                                                     
    print(out.shape, out.dtype)
