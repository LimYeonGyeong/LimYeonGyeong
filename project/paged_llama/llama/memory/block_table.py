import torch

class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size
        self.physical_blocks = []  # 실제 블록 번호 리스트
        self._tensor_cache = None  # 연산용 텐서 캐시

    def add_block(self, physical_block_idx):
        self.physical_blocks.append(physical_block_idx)
        self._tensor_cache = None  # 블록이 추가되면 캐시 초기화

    def to_tensor(self, device="cuda"):
        # 캐시가 없으면 새로 생성하여 보관 (성능 최적화)
        if self._tensor_cache is None:
            if not self.physical_blocks:
                return torch.zeros((1, 1), dtype=torch.long, device=device)
            self._tensor_cache = torch.tensor(self.physical_blocks, dtype=torch.long, device=device).unsqueeze(0)
        
        # 디바이스가 다를 경우에만 이동
        if self._tensor_cache.device != torch.device(device):
            return self._tensor_cache.to(device)
        return self._tensor_cache

    def __getitem__(self, idx):
        # 이제 텐서가 아니라 "물리 블록 번호(int)"를 직접 반환합니다.
        return self.physical_blocks[idx]

    def __len__(self):
        return len(self.physical_blocks)

    def __repr__(self):
        return f"BlockTable(size={len(self.physical_blocks)}, blocks={self.physical_blocks})"