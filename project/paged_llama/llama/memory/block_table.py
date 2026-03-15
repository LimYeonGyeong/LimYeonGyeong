import torch

class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size
        self.physical_blocks = []  # 실제 블록 번호 리스트

    def add_block(self, physical_block_idx):
        self.physical_blocks.append(physical_block_idx)

    def to_tensor(self, device="cuda"):
        if not self.physical_blocks:
            return torch.zeros((1, 1), dtype=torch.long, device=device)
        return torch.tensor(self.physical_blocks, dtype=torch.long, device=device).unsqueeze(0)

    def __getitem__(self, idx):
        # 이제 텐서가 아니라 "물리 블록 번호(int)"를 직접 반환합니다.
        return self.physical_blocks[idx]

    def __len__(self):
        return len(self.physical_blocks)

    def __repr__(self):
        return f"BlockTable(size={len(self.physical_blocks)}, blocks={self.physical_blocks})"