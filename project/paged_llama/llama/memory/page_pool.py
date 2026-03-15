# paged_llama/llama/memory/page_pool.py
import torch

class PagePool:
    def __init__(self, num_blocks, num_heads, block_size, head_dim, device="cuda", dtype=torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype # 타입 저장

        # [★핵심] k_cache와 v_cache를 주입받은 dtype으로 생성
        # 차원 순서: [블록 개수, 헤드 수, 블록 내 토큰 수, 헤드 차원]
        self.k_cache = torch.zeros(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device
        )
        
        # 자유 블록 리스트 (빈 블록 관리)
        self.free_blocks = list(range(num_blocks))

    def allocate(self):
        if not self.free_blocks:
            raise RuntimeError("PagePool에 남은 블록이 없습니다.")
        return self.free_blocks.pop(0)

    def free(self, block_idx):
        """블록을 반납합니다."""
        self.free_blocks.append(block_idx)

    def get_memory_usage(self):
        """현재 할당된 VRAM 사용량을 MB 단위로 반환합니다."""
        element_size = self.k_cache.element_size()
        total_elements = self.k_cache.nelement() + self.v_cache.nelement()
        return (total_elements * element_size) / (1024 * 1024)