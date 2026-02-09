# paged_llama/llama/memory/page_pool.py
import torch

class PagePool:
    def __init__(self, num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=torch.bfloat16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # KV Cache를 위한 거대 텐서 미리 할당 (Zero-copy를 위한 공간)
        # Shape: [물리 블록 개수, 헤드 수, 블록 크기, 헤드 차원]
        self.k_cache = torch.zeros(
            (int(num_blocks), int(num_heads), int(block_size), int(head_dim)), 
            device=device, 
            dtype=dtype
        )
        self.v_cache = torch.zeros(
            (int(num_blocks), int(num_heads), int(block_size), int(head_dim)), 
            device=device, 
            dtype=dtype
        )
        
        # 블록 사용 여부 관리
        self.free_blocks = list(range(num_blocks))

    def allocate(self):
        """빈 블록 하나를 할당받습니다."""
        if not self.free_blocks:
            raise MemoryError("PagePool에 남은 블록이 없습니다!")
        return self.free_blocks.pop(0)

    def free(self, block_idx):
        """블록을 반납합니다."""
        self.free_blocks.append(block_idx)

    def get_memory_usage(self):
        """현재 할당된 VRAM 사용량을 MB 단위로 반환합니다."""
        element_size = self.k_cache.element_size()
        total_elements = self.k_cache.nelement() + self.v_cache.nelement()
        return (total_elements * element_size) / (1024 * 1024)