import torch
from .cache_utils import Cache


class PagedCache(Cache):
    """
    HF generate가 기대하는 Cache 인터페이스를 흉내내는 shim.
    실제 KV 저장은 attention 내부의 page_pool / block_table에서 처리한다.
    """

    def __init__(self, config=None, page_pool=None, block_table=None, request_state=None):
        super().__init__()
        self.config = config
        self.page_pool = page_pool
        self.block_table = block_table
        self.request_state = request_state
        self.seq_len = 0 if request_state is None else int(request_state.get("seq_len", 0))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self.request_state is not None:
            return int(self.request_state.get("seq_len", 0))
        return self.seq_len

    def get_max_cache_shape(self):
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        # beam search용. 지금은 single batch 테스트라 그대로 둠
        return self

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        HF 내부 호출 호환용.
        실제 KV 저장은 PagedLlamaAttention.forward() 안에서 page_pool에 write 하므로
        여기서는 seq_len만 갱신하고 원본 반환.
        """
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            cp = cache_kwargs["cache_position"]
            if cp is not None and cp.numel() > 0:
                new_len = int(cp[-1].item()) + 1
                self.seq_len = max(self.seq_len, new_len)
                if self.request_state is not None:
                    self.request_state["seq_len"] = max(int(self.request_state.get("seq_len", 0)), new_len)
        else:
            if key_states is not None:
                self.seq_len += key_states.shape[-2]
                if self.request_state is not None:
                    self.request_state["seq_len"] = max(int(self.request_state.get("seq_len", 0)), self.seq_len)

        return key_states, value_states

    def to_legacy_cache(self):
        return None

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        return cls()
