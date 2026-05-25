import pathlib
import torch

from dataclasses import asdict
from math import cos, pi
from typing import Callable

from src.config import GPTConfig, TrainConfig
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

    cfg = TrainConfig(
        GPTConfig(V=V),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    model = GPT(
        V=V,
        T_max=cfg.model.T_max,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        rope_base=cfg.model.rope_base,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=make_lr_lambda(
            warmup_steps=cfg.warmup_steps, 
            total_steps=cfg.total_steps,
            min_lr_ratio=cfg.min_lr_ratio
            )
        )

    print("=== config ===")
    for k, v in asdict(cfg).items():                                                                  
        if isinstance(v, dict):                               
            for sub_k, sub_v in v.items():
                print(f"  {k}.{sub_k}: {sub_v}")                                                      
        else:
            print(f"  {k}: {v}")                                                                      
    print("===")          
    for step in range(cfg.total_steps):
        if step % cfg.eval_interval == 0:
            val_loss = eval_loss(
                model=model, 
                ds_val=ds_val, 
                B=cfg.B, 
                T=cfg.T, 
                eval_iters=cfg.eval_iters, 
                device=device
                )
        x, y = ds_train.get_batch(cfg.B, cfg.T)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if step % cfg.eval_interval == 0:
            print(f"step: {step}  lr: {optimizer.param_groups[0]['lr']:.2e}  train_loss: {loss.item():.4f} val_loss: {val_loss:.4f}")

    final_val_loss = eval_loss(
        model=model, ds_val=ds_val, B=cfg.B, T=cfg.T, eval_iters=cfg.eval_iters, device=device
    )
    print(f"final val_loss: {final_val_loss}")
    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': asdict(cfg)
    }, 'checkpoints/model.pt')
if __name__ == '__main__':
    main()