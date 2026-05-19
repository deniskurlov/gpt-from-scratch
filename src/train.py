import torch
import torch.nn.functional as F

from math import cos, pi

from src.data import load_corpus, Tokenizer, TokenizedDataset
from src.model import GPT


def make_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))
        return lr_lambda


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

    B, T = 64, 256
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = GPT(V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1
    )

    warmup_steps = 100
    total_steps = 5000
    min_lr_ratio = 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=make_lr_lambda(warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio)
        )

    for step in range(total_steps + 1):
        x, y = ds.get_batch(B, T)
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            # print(f"step: {step}    loss: {loss.item()}")
            print(f"step: {step}  lr: {optimizer.param_groups[0]['lr']:.2e}  loss: {loss.item():.4f}")