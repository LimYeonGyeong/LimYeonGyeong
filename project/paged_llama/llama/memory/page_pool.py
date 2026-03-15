# paged_llama/llama/memory/page_pool.py
import torch

class PagePool:
    def __init__(
        self,
        num_layers,
        num_blocks,
        num_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # [num_layers, num_blocks, num_heads, block_size, head_dim]
        self.k_cache = torch.zeros(
            (num_layers, num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )

        self.free_blocks = list(range(num_blocks))

    def allocate(self):
        if not self.free_blocks:
            raise RuntimeError("PagePool에 남은 블록이 없습니다.")
        return self.free_blocks.pop(0)

    def free(self, block_idx):
        self.free_blocks.append(block_idx)

    def get_memory_usage(self):
        element_size = self.k_cache.element_size()
        total_elements = self.k_cache.nelement() + self.v_cache.nelement()
        return (total_elements * element_size) / (1024 * 1024)