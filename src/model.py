# src/model.py

import torch
import torch.nn.functional as F

from jaxtyping import Int64, Float32
from torch import Tensor, nn
from typing import Iterator

from src.attention import MultiHeadAttention
from src.cache import KVCache
from src.embedding import TokenEmbedding
from src.data import load_corpus, Tokenizer, TokenizedDataset
from src.normalization import LayerNormalization
from src.mlp import MLP


class Block(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        rope_base: float,
        d_ff: int | None,
        dropout: float
        ) -> None:
        super().__init__()
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)
        self.attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model, rope_base=rope_base)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Float32[Tensor, "B T d_model"],
        cache: KVCache | None = None
        ) -> tuple[Float32[Tensor, "B T d_model"], KVCache | None]:
        attn_out, cache = self.attn(self.ln1(x), cache)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x, cache


class GPT(nn.Module):
    def __init__(
        self,
        V: int,
        T_max: int,
        n_heads: int,
        d_model: int,
        n_layers: int,
        rope_base: float,
        d_ff: int | None,
        dropout: float
        ) -> None:
        super().__init__()
        self.V = V
        self.T_max = T_max
        self.tok_emb = TokenEmbedding(V=V, d_model=d_model)
        # self.pos_emb = LearnedPositionalEmbedding(T_max=T_max, d_model=d_model)
        self.blocks = nn.ModuleList(
            [Block(n_heads=n_heads, d_model=d_model, rope_base=rope_base, d_ff=d_ff, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.final_ln = LayerNormalization(d_model)
        self.lm_head = nn.Linear(d_model, V, bias=False)
        self.lm_head.weight = self.tok_emb.tok_emb.weight  # tied weights
        # Re-init after tying so both embedding and lm_head share the small N(0, 0.02) scale.
        # Default N(0, 1) gives logit std ~√d_model → loss ~70 instead of log(V) ≈ 4.17. GPT-2 convention.
        nn.init.normal_(self.tok_emb.tok_emb.weight, mean=0.0, std=0.02)      
        
    def forward(self,
                ids: Int64[Tensor, "B T"],
                targets: Int64[Tensor, "B T"] | None = None,
                cache: list[KVCache] | None = None
        ) -> Float32[Tensor, "B T V"] | tuple[Float32[Tensor, "B T V"], Float32[Tensor, ""]]:
        assert cache is None or len(cache) == len(self.blocks)
        T = ids.shape[-1]
        # start_pos = 0 if cache is None else len(cache[0])
        # positions = torch.arange(start_pos, start_pos + T, device=ids.device)
        # x = self.tok_emb(ids) + self.pos_emb(positions)
        x = self.tok_emb(ids)
        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[i]
            x, _ = block(x, layer_cache)  # cache is mutated in place, discard the return (_)
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

    def stream(
        self, 
        ids: Int64[Tensor, "B T"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True
        ) -> Iterator[Int64[Tensor, "1"]]:
        if ids.shape[0] != 1:
            raise ValueError("stream requires B=1; use generate for batch")
        yield from self._self_iterator(ids=ids, max_new_tokens=max_new_tokens, temperature=temperature,
                                       top_k=top_k, top_p=top_p, use_cache=use_cache)

    def _self_iterator(
        self, 
        ids: Int64[Tensor, "B T"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True
        ) -> Iterator[Int64[Tensor, "B 1"]]:
        # the canonical generator — no B guard, internal use only
        self.eval()
        try:
            with torch.no_grad():
                cache = [KVCache(max_size=self.T_max) for _ in range(len(self.blocks))] if use_cache else None
                for step in range(max_new_tokens):
                    if use_cache:
                        ids_in = ids if step == 0 else next_token
                    else:
                        ids_in = ids[:, -self.T_max:]
                    logits = self(ids_in, cache=cache)[:, -1, :]  # shape (B, V)
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
                    yield next_token
        finally: 
            self.train()

    def generate(
        self, 
        ids: Int64[Tensor, "B T"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True
        ) -> Int64[Tensor, "B T+max_new_tokens"]:
        for next_token in self._self_iterator(ids, max_new_tokens, 
                                      temperature=temperature,
                                      top_k=top_k, top_p=top_p,
                                      use_cache=use_cache):
            ids = torch.cat([ids, next_token], dim=-1)
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

    gpt = GPT(
        V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers,
        rope_base=10_000.0, d_ff=None, dropout=0.1
        )
    
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