import torch

class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size
        self.physical_blocks = []  # 실제 블록 번호 리스트
        self._tensor_cache = None  # 매번 생성하지 않도록 캐싱
        self._is_dirty = True      # 리스트가 변했는지 확인하는 플래그

    def add_block(self, physical_block_idx):
        self.physical_blocks.append(physical_block_idx)
        self._is_dirty = True      # 블록이 추가되면 텐서 갱신 필요

    def to_tensor(self, device="cuda"):
        # 리스트가 변했거나 캐시가 없으면 새로 생성
        if self._is_dirty or self._tensor_cache is None:
            if not self.physical_blocks:
                # 비어있을 경우 안전 장치
                return torch.zeros((1, 1), dtype=torch.long, device=device)
            
            self._tensor_cache = torch.tensor(
                self.physical_blocks, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0) # (1, num_blocks) 형태 유지
            self._is_dirty = False
            
        # 디바이스가 다르면 옮겨서 반환
        if self._tensor_cache.device != torch.device(device):
            return self._tensor_cache.to(device)
            
        return self._tensor_cache

    def __getitem__(self, idx):
        # 이제 텐서가 아니라 "실제 블록 번호"를 반환합니다.
        # modeling_llama.py에서 block_table[0, idx]로 접근하는 케이스와 충돌하지 않도록 주의해야 합니다.
        return self.physical_blocks[idx]

    def __len__(self):
        return len(self.physical_blocks)

    def __repr__(self):
        return f"BlockTable(size={len(self.physical_blocks)}, blocks={self.physical_blocks})"