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
        self.V = V
        self.T_max = T_max
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
            loss = F.cross_entropy(logits.view(-1, self.V), targets.view(-1))
            return logits, loss

    @staticmethod
    def apply_top_k_(logits: Float32[Tensor, "B V"], k: int) -> Float32[Tensor, "B V"]:
        values, _ = torch.topk(logits, k)
        logits[logits < values[:, -1:]] = -torch.inf
        return logits

    @staticmethod
    def apply_top_p_(logits: Float32[Tensor, "B V"], p: float) -> Float32[Tensor, "B V"]:
        sorted_logits, sorted_ids = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        sorted_mask = cum > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(sorted_mask).scatter_(dim=-1, index=sorted_ids, src=sorted_mask)
        logits[mask] = -torch.inf
        return logits

    def generate(
        self, 
        ids: Int64[Tensor, "B T"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None
        ) -> Int64[Tensor, "B T+max_new_tokens"]:
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ids_in = ids[:, -self.T_max:]
                logits = self(ids_in)[:, -1, :]  # shape (B, V)
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        self.apply_top_k_(logits, top_k)
                    if top_p is not None:
                        self.apply_top_p_(logits, top_p)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_token], dim=-1)
        self.train()
        return ids


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

    B, T = 1, 4

    gpt = GPT(V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
    
    x, y = ds.get_batch(B, T)
    logits, loss = gpt(x, targets=y)

    # print(loss.item())

    # print(x)
    # print(gpt.generate(x, temperature=0.0, max_new_tokens=500).shape[-1] == 500 + T)

    # print(
    #     gpt._apply_top_k(
    #         torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    #         k=2
    #         )
    #     )

    
    print('\noriginal logits:')
    l = torch.tensor([[1.0, 5.0, 2.0, 6.0, 3.0]])
    print(l)
    print('\nsorted logits and indices:')
    sorted_vals, sorted_ids = torch.sort(l, dim=-1, descending=True)
    print('\tsorted_vals:  ', sorted_vals)
    print('\tsorted_ids:   ', sorted_ids)
    print('\nProbabilities:')
    probs = F.softmax(sorted_vals, dim=-1)
    print('\tprobs: ', probs)
    print('\nCumulative sums:')
    cum = probs.cumsum(dim=-1)
    print('\tcum: ', cum)
    p = 0.9
    print(f'\nSorted mask (cum > {p}):')
    sorted_mask = cum > p
    print('\tsorted_mask: ', sorted_mask)    
    print("\nShifted sorted mask `sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()`:")
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    print('\tsorted_mask: ', sorted_mask)
    print("\nThe first element must be included, set `sorted_mask[0] = False`:")
    sorted_mask[..., 0] = False 
    print('\tsorted_mask: ', sorted_mask)
    print('\nWrite the mask for the original (unsorted) logits:')
    mask = torch.zeros_like(sorted_mask)
    print('\tpre-build a zero mask:                   ', mask)
    mask.scatter_(dim=-1, index=sorted_ids, src=sorted_mask)
    print('\tpopulate (scatter) by the actual values: ', mask)
    print('\nMasked top-p logits: ')
    l[mask] = -torch.inf
    print('\tmasked logits: ', l)
