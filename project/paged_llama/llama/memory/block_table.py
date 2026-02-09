# paged_llama/llama/memory/block_table.py
class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size
        self.physical_blocks = [] # 할당된 물리 블록 번호 리스트

    def add_block(self, physical_block_idx):
        self.physical_blocks.append(physical_block_idx)

    def get_physical_block_idx(self, logical_token_idx):
        """토큰 번호(예: 50번째)를 주면 해당 토큰이 담길 물리 블록 번호를 알려줍니다."""
        block_idx = logical_token_idx // self.block_size
        if block_idx < len(self.physical_blocks):
            return self.physical_blocks[block_idx]
        return None

    def get_all_blocks(self):
        return self.physical_blocks