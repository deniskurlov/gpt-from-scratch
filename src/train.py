import pathlib
import torch

from math import cos, pi
from typing import Callable

from src.data import load_corpus, Tokenizer, TokenizedDataset
from src.model import GPT


def make_lr_lambda(
    warmup_steps: int, total_steps: int, min_lr_ratio: float
    ) -> Callable[[int], float] :
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))
        return lr_lambda

def eval_loss(
    model: GPT, ds_val: TokenizedDataset, B: int, T: int, eval_iters: int, device: str
    ) -> float:
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            x, y = ds_val.get_batch(B, T)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            losses[i] = loss
    model.train()
    return losses.mean().item()

def main() -> None:    
    seed = 42
    T_max = 256
    d_model = 128
    n_heads = 4
    n_layers = 6

    B, T = 64, 256
    
    warmup_steps = 100
    total_steps = 5000
    min_lr_ratio = 0.1

    eval_iters = 20

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    torch.manual_seed(seed)

    text = load_corpus()
    tok = Tokenizer(text)
    encoded_text = tok.encode_to_tensor(text)
    ds_train = TokenizedDataset(
        encoded_text[:int(0.9 * len(encoded_text))]
        )
    ds_val = TokenizedDataset(
        encoded_text[int(0.9 * len(encoded_text)):]
    )

    V = tok.vocab_size
    
    model = GPT(V=V, T_max=T_max, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=make_lr_lambda(warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio)
        )

    for step in range(total_steps):
        if step % 200 == 0:
            val_loss = eval_loss(
                model=model, ds_val=ds_val, B=B, T=T, eval_iters=eval_iters, device=device
                )
        x, y = ds_train.get_batch(B, T)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            print(f"step: {step}  lr: {optimizer.param_groups[0]['lr']:.2e}  train_loss: {loss.item():.4f} val_loss: {val_loss:.4f}")

    final_val_loss = eval_loss(
        model=model, ds_val=ds_val, B=B, T=T, eval_iters=eval_iters, device=device
    )
    print(f"final val_loss: {final_val_loss}")
    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model.pt')
if __name__ == '__main__':
    main()