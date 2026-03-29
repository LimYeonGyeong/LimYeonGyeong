from paged_llama.llama.memory.block_table import BlockTable


class SimpleScheduler:
    def __init__(self, page_pool, block_size=16):
        self.page_pool = page_pool
        self.block_size = block_size
        self.request_tables = {}  # request_id -> BlockTable

    def allocate_for_request(self, request_id, total_tokens):
        needed_blocks = (total_tokens + self.block_size - 1) // self.block_size
        block_table = BlockTable(block_size=self.block_size)

        for _ in range(needed_blocks):
            physical_block = self.page_pool.allocate()
            block_table.add_block(physical_block)

        self.request_tables[request_id] = block_table
        return block_table

    def get_block_table(self, request_id):
        return self.request_tables[request_id]

    def release_request(self, request_id):
        block_table = self.request_tables[request_id]

        if hasattr(block_table, "to_tensor"):
            block_ids = block_table.to_tensor(device="cpu")[0].tolist()
        else:
            block_ids = block_table.block_table[0].tolist()

        for bid in block_ids:
            self.page_pool.free(bid)

        del self.request_tables[request_id]