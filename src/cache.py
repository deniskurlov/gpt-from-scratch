import torch

from jaxtyping import Float32
from torch import Tensor


class KVCache():
    def __init__(self):
        self.K = None
        self.V = None
    
    def append(
        self,
        k_new: Float32[Tensor, "B n_heads T_new head_dim"],
        v_new: Float32[Tensor, "B n_heads T_new head_dim"]
        ) -> None:
        if self.K is None:
            self.K = k_new
            self.V = v_new
        else:
            self.K = torch.cat([self.K, k_new], dim=-2)
            self.V = torch.cat([self.V, v_new], dim=-2)

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


if __name__ == '__main__':
    cache = KVCache()
    assert len(cache) == 0
    cache.append(torch.zeros(1, 4, 3, 16), torch.zeros(1, 4, 3, 16))
    assert len(cache) == 3
    cache.append(torch.zeros(1, 4, 1, 16), torch.zeros(1, 4, 1, 16))  # T_new=1
    assert len(cache) == 4
    K, V = cache.get()
    assert K.shape == (1, 4, 4, 16)