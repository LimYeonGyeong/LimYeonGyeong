import torch

class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size
        self.physical_blocks = [] # 할당된 물리 블록 번호 리스트

    def add_block(self, physical_block_idx):
        self.physical_blocks.append(physical_block_idx)

    # modeling_llama.py의 .size(1) 및 [0, idx] 접근을 지원하기 위한 메서드
    def to_tensor(self, device="cuda"):
        if not self.physical_blocks:
            # 비어있을 경우 안전하게 기본 블록 할당 (에러 방지)
            return torch.zeros((1, 1), dtype=torch.long, device=device)
        # (1, num_blocks) 형태의 2차원 텐서로 반환해야 .size(1) 연산이 가능합니다.
        return torch.tensor(self.physical_blocks, device=device).unsqueeze(0)

    def __getitem__(self, idx):
        # 텐서처럼 동작하게 하려면 내부 리스트를 텐서로 바꿔서 반환
        return self.to_tensor()