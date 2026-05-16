# src/model.py

import torch
import torch.nn.functional as F

from jaxtyping import Int64, Float32
from torch import Tensor, nn

from src.attention import MultiHeadAttention
from src.embedding import LearnedPositionalEmbedding, TokenEmbedding
from src.data import load_corpus, Tokenizer, TokenizedDataset
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

class GPT(nn.Module):
    def __init__(self, V: int, T_max: int, n_heads: int, d_model: int, n_layers: int,
                 d_ff: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.tok_emb = TokenEmbedding(V=V, d_model=d_model)
        self.pos_emb = LearnedPositionalEmbedding(T_max=T_max, d_model=d_model)
        self.blocks = nn.ModuleList(
            [Block(T_max=T_max, n_heads=n_heads, d_model=d_model, 
                    d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_ln = LayerNormalization(d_model)
        self.lm_head = nn.Linear(d_model, V, bias=False)
        self.lm_head.weight = self.tok_emb.tok_emb.weight  # tied weights
        # Re-init after tying so both embedding and lm_head share the small N(0, 0.02) scale.
        # Default N(0, 1) gives logit std ~√d_model → loss ~70 instead of log(V) ≈ 4.17. GPT-2 convention.
        nn.init.normal_(self.tok_emb.tok_emb.weight, mean=0.0, std=0.02)      
        
    def forward(self,
                ids: Int64[Tensor, "B T"],
                targets: Int64[Tensor, "B T"] | None = None
        ) -> Float32[Tensor, "B T V"] | tuple[Float32[Tensor, "B T V"], Float32[Tensor, ""]]:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(positions)
        for block in self.blocks: 
            x = block(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            return logits, loss
        

if __name__ == '__main__':
    torch.manual_seed(42)

    text = load_corpus()
    tok = Tokenizer(text)
    ds = TokenizedDataset(tok.encode_to_tensor(text))

    V = tok.vocab_size
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6

    B, T = 2, 4

    emb = TokenEmbedding(V=V, d_model=d_model)
    pos_emb = LearnedPositionalEmbedding(T_max=T_max, d_model=d_model)

    gpt = GPT(V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
    
    x, y = ds.get_batch(B, T)
    logits, loss = gpt(x, targets=y)

    print(loss.item())
