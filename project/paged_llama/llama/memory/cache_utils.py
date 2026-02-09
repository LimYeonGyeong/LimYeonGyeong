import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import Cache 

class DynamicCache(Cache):
    def __init__(self):
        self.key_cache = {}
        self.value_cache = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        pass