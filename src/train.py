import os
import subprocess
import sys
import torch
import uuid

from dataclasses import asdict
from datetime import datetime
from math import cos, pi
from pathlib import Path
from typing import Callable

from src.config import GPTConfig, TrainConfig
from src.data import load_corpus, Tokenizer, TokenizedDataset
from src.model import GPT


def make_lr_lambda(
    warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> Callable[[int], float]:
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


def save_checkpoint(
    path: str,
    step: int,
    val_loss: float,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    config: dict,
) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(
            {
                "step": step,
                "val_loss": val_loss,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "scheduler_state_dict": scheduler_state,
                "config": config,
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main() -> None:

    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = out_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}.txt"

    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log(f"{datetime.now().isoformat()}")

    text = load_corpus()
    tok = Tokenizer(text)
    encoded_text = tok.encode_to_tensor(text)
    ds_train = TokenizedDataset(encoded_text[: int(0.9 * len(encoded_text))])
    ds_val = TokenizedDataset(encoded_text[int(0.9 * len(encoded_text)) :])
    V = tok.vocab_size

    cfg = TrainConfig(
        GPTConfig(V=V),
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"]).decode()
    log(f"git commit: {commit}", console=False)
    if diff:
        log(f"=== uncommitted diff ===\n{diff}", console=False)
        log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running Torch {torch.__version__}", console=False)
    log(f"Device: {device}")
    log("=" * 100, console=False)

    model = GPT(
        V=V,
        T_max=cfg.model.T_max,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        rope_base=cfg.model.rope_base,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=make_lr_lambda(
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
            min_lr_ratio=cfg.min_lr_ratio,
        ),
    )

    log("=== config ===")
    for k, v in asdict(cfg).items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                log(f"  {k}.{sub_k}: {sub_v}")
        else:
            log(f"  {k}: {v}")
    log("=== training ===")
    min_val_loss = float("inf")
    for step in range(cfg.total_steps):
        if step % cfg.eval_interval == 0:
            val_loss = eval_loss(
                model=model,
                ds_val=ds_val,
                B=cfg.B,
                T=cfg.T,
                eval_iters=cfg.eval_iters,
                device=device,
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
            log(
                f"step: {step}  lr: {optimizer.param_groups[0]['lr']:.2e}  train_loss: {loss.item():.4f} val_loss: {val_loss:.4f}"
            )
            save_checkpoint(
                path="checkpoints/latest.pt",
                step=step,
                val_loss=val_loss,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                config=asdict(cfg),
            )
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_checkpoint(
                    path="checkpoints/best.pt",
                    step=step,
                    val_loss=val_loss,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    config=asdict(cfg),
                )

    latest_val_loss = eval_loss(
        model=model,
        ds_val=ds_val,
        B=cfg.B,
        T=cfg.T,
        eval_iters=cfg.eval_iters,
        device=device,
    )
    log(f"Final val_loss: {latest_val_loss}")

    save_checkpoint(
        path="checkpoints/latest.pt",
        step=step,
        val_loss=latest_val_loss,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        config=asdict(cfg),
    )
    if latest_val_loss < min_val_loss:
        save_checkpoint(
            path="checkpoints/best.pt",
            step=step,
            val_loss=latest_val_loss,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            config=asdict(cfg),
        )

    log(f"{datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
