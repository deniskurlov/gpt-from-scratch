from dataclasses import dataclass


@dataclass
class GPTConfig:
    V: int
    T_max: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int | None = None
    dropout: float = 0.1

@dataclass
class TrainConfig:
    model: GPTConfig
    # optimizer
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.1
    # schedule
    warmup_steps: int = 100
    total_steps: int = 5000
    min_lr_ratio: float = 0.1
    # training
    B: int = 64
    T: int = 256
    eval_iters: int = 20
    eval_interval: int = 200
    grad_clip_norm: float = 1.0
    seed: int = 42
