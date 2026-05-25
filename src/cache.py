import torch

from jaxtyping import Float32
from torch import Tensor


class KVCache():
    def __init__(self, max_size: int | None = None):
        self.K = None
        self.V = None
        self.max_size = max_size
        self.total_appended = 0

    def append(
        self,
        k_new: Float32[Tensor, "B n_heads T_new head_dim"],
        v_new: Float32[Tensor, "B n_heads T_new head_dim"]
        ) -> None:
        T_new = k_new.shape[-2]
        if self.K is None:
            self.K = k_new
            self.V = v_new
        else:
            self.K = torch.cat([self.K, k_new], dim=-2)
            self.V = torch.cat([self.V, v_new], dim=-2)
        self.total_appended += T_new
        if self.max_size is not None and self.K.shape[-2] > self.max_size:
            self.K = self.K[..., -self.max_size:, :]
            self.V = self.V[..., -self.max_size:, :]

    def get(self) -> tuple[
            Float32[Tensor, "B n_heads T_cached head_dim"] | None,
            Float32[Tensor, "B n_heads T_cached head_dim"] | None
            ]:
        return (self.K, self.V)

    def __len__(self) -> int:
        if self.K is None:
            return 0
        else:
            return self.K.shape[-2]

    @property
    def window_start(self) -> int:
        return self.total_appended - (self.K.shape[-2] if self.K is not None else 0)


if __name__ == '__main__':
    cache = KVCache()
    assert len(cache) == 0
    cache.append(torch.zeros(1, 4, 3, 16), torch.zeros(1, 4, 3, 16))
    assert len(cache) == 3
    cache.append(torch.zeros(1, 4, 1, 16), torch.zeros(1, 4, 1, 16))  # T_new=1
    assert len(cache) == 4
    K, V = cache.get()
    assert K.shape == (1, 4, 4, 16)