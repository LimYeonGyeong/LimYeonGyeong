# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# paged_llama/llama/modeling_llama.py

# paged_llama/llama/modeling/modeling_llama.py

import torch
from torch import nn
import math
from typing import Optional, Callable, Any, Union, List

from ..activations import ACT2FN
from ..memory.cache_utils import Cache, DynamicCache
from ..generation.generation import GenerationMixin
from ..memory.masking_utils import create_causal_mask
from ..memory.paged_cache import PagedCache

# 같은 modeling 폴더 내 파일 참조
from .modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from .modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
)
from .modeling_utils import PreTrainedModel

# Transformers 표준 출력 사용
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# utils 폴더로 이동된 파일 참조
from ..utils.processing_utils import Unpack
# modeling_llama.py 상단 수정 (이전에 드린 것과 동일)

from ..utils.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    use_kernel_forward_from_hub,
    use_kernelized_func,
    use_kernel_func_from_hub,
    check_model_inputs,
)
try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
from ..config.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

class PagedCacheShim:
    """
    HF generate()가 요구하는 최소 cache 인터페이스만 흉내내는 객체.
    실제 KV는 저장하지 않고, '현재까지 본 토큰 수'만 관리한다.
    """
    def __init__(self, seen_tokens: int = 0):
        self.seen_tokens = seen_tokens

    def get_seq_length(self):
        return self.seen_tokens

    def update_seen_tokens(self, new_len: int):
        self.seen_tokens = new_len

    def __repr__(self):
        return f"PagedCacheShim(seen_tokens={self.seen_tokens})"

@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

from torch.amp import autocast
class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: LlamaConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

# 기존 modeling_llama.py 상단에 있는 함수들을 활용합니다.
# (apply_rotary_pos_emb, repeat_kv 등은 이미 파일에 있다고 가정)
class PagedLlamaAttention(nn.Module):
    def __init__(self, config, layer_idx=0, page_pool=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # 외부에서 주입받을 PagedPool
        self.page_pool = page_pool

        # projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        block_table: Optional[torch.Tensor] = None,
        use_cache: bool | None = False,
        **kwargs,
    ):

        # 1. dtype / device
        original_dtype = hidden_states.dtype
        target_dtype = self.q_proj.weight.dtype
        target_device = self.q_proj.weight.device

        hidden_states = hidden_states.to(dtype=target_dtype, device=target_device)
        bsz, q_len, _ = hidden_states.size()

        # 2. page pool
        pool = self.page_pool if self.page_pool is not None else getattr(self, "pool", None)
        if pool is None:
            raise ValueError(f"Layer {self.layer_idx} | PagePool is not initialized.")

        # 3. block table 확보
        target_block_table = block_table
        if target_block_table is None:
            target_block_table = getattr(self, "block_table", None)

        if target_block_table is None:
            raise ValueError(f"Layer {self.layer_idx} | block_table is None")

        if hasattr(target_block_table, "to_tensor"):
            block_table_tensor = target_block_table.to_tensor(device=target_device)
        else:
            block_table_tensor = target_block_table.to(device=target_device)

        # 4. q, k, v projection
        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]

        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [bsz, num_kv_heads, q_len, head_dim]

        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # [bsz, num_kv_heads, q_len, head_dim]

        # 5. RoPE
        if position_embeddings is None and "position_embeddings" in kwargs:
            position_embeddings = kwargs["position_embeddings"]

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos.to(target_device),
                sin.to(target_device),
            )

        # 6. 현재 step 위치 정보
        start_pos = position_ids.reshape(-1)[0].item() if position_ids is not None else 0
        total_seq_len = start_pos + q_len
        num_needed_blocks = (total_seq_len + pool.block_size - 1) // pool.block_size

        # 7. WRITE
        for i in range(q_len):
            abs_pos = start_pos + i
            block_idx_logic = abs_pos // pool.block_size
            block_offset = abs_pos % pool.block_size

            if block_idx_logic >= block_table_tensor.size(1):
                raise IndexError(
                    f"Layer {self.layer_idx} | block_idx_logic {block_idx_logic} "
                    f"out of range (size {block_table_tensor.size(1)})"
                )

            physical_block_idx = block_table_tensor[0, block_idx_logic].item()

            if i == 0 or abs_pos % pool.block_size == 0:
                print(
                    f"[VERIFY-WRITE] Layer {self.layer_idx} | "
                    f"Pos {abs_pos} (Logic {block_idx_logic}) -> Physical {physical_block_idx}"
                )

            pool.k_cache[self.layer_idx, physical_block_idx, :, block_offset, :] = \
                key_states[0, :, i, :].to(pool.k_cache.dtype)

            pool.v_cache[self.layer_idx, physical_block_idx, :, block_offset, :] = \
                value_states[0, :, i, :].to(pool.v_cache.dtype)

        # 8. block table 크기 확인
        if num_needed_blocks > block_table_tensor.size(1):
            raise RuntimeError(
                f"Layer {self.layer_idx} | Block table size too small! "
                f"Need {num_needed_blocks} blocks but only have {block_table_tensor.size(1)}"
            )

        # 9. READ 준비
        active_indices = block_table_tensor[0, :num_needed_blocks].to(target_device)

        layer_k_cache = pool.k_cache[self.layer_idx]   # [num_blocks, num_kv_heads, block_size, head_dim]
        layer_v_cache = pool.v_cache[self.layer_idx]

        k_blocks = layer_k_cache.index_select(0, active_indices)
        v_blocks = layer_v_cache.index_select(0, active_indices)

        print(f"[VERIFY-READ-1] Layer {self.layer_idx}")
        print(f"  active_indices = {active_indices.tolist()}")
        print(f"  k_blocks.shape = {k_blocks.shape}")
        print(f"  v_blocks.shape = {v_blocks.shape}")
        print(f"  total_seq_len = {total_seq_len}")
        print(f"  num_needed_blocks = {num_needed_blocks}")

        # 10. flatten
        k_flat = k_blocks.transpose(0, 1).reshape(self.num_key_value_heads, -1, self.head_dim)
        v_flat = v_blocks.transpose(0, 1).reshape(self.num_key_value_heads, -1, self.head_dim)

        # repeat 전 상태 보관 (검증용)
        full_key_states_before_repeat = k_flat[:, :total_seq_len, :].unsqueeze(0)
        full_value_states_before_repeat = v_flat[:, :total_seq_len, :].unsqueeze(0)

        print(f"[VERIFY-READ-2] Layer {self.layer_idx}")
        print(f"  k_flat.shape = {k_flat.shape}")
        print(f"  v_flat.shape = {v_flat.shape}")
        print(f"  full_key_states_before_repeat.shape = {full_key_states_before_repeat.shape}")
        print(f"  full_value_states_before_repeat.shape = {full_value_states_before_repeat.shape}")

        # 11. WRITE vs READ 직접 비교
        check_pos = total_seq_len - 1
        check_logic_block = check_pos // pool.block_size
        check_offset = check_pos % pool.block_size
        check_physical_block = block_table_tensor[0, check_logic_block].item()

        written_k = pool.k_cache[self.layer_idx, check_physical_block, :, check_offset, :]
        written_v = pool.v_cache[self.layer_idx, check_physical_block, :, check_offset, :]

        print(f"[VERIFY-READ-3] Layer {self.layer_idx}")
        print(f"  check_pos = {check_pos}")
        print(f"  logic_block = {check_logic_block}")
        print(f"  physical_block = {check_physical_block}")
        print(f"  offset = {check_offset}")
        print(f"  written_k sample = {written_k[0, :8].detach().cpu()}")
        print(f"  written_v sample = {written_v[0, :8].detach().cpu()}")

        read_k = full_key_states_before_repeat[0, :, check_pos, :]
        read_v = full_value_states_before_repeat[0, :, check_pos, :]

        print(f"[VERIFY-READ-4] Layer {self.layer_idx}")
        print(f"  read_k sample = {read_k[0, :8].detach().cpu()}")
        print(f"  read_v sample = {read_v[0, :8].detach().cpu()}")

        k_diff = (written_k - read_k).abs().max().item()
        v_diff = (written_v - read_v).abs().max().item()

        print(f"  max |written_k - read_k| = {k_diff}")
        print(f"  max |written_v - read_v| = {v_diff}")

        # 12. GQA 확장
        full_key_states = full_key_states_before_repeat.to(dtype=query_states.dtype)
        full_value_states = full_value_states_before_repeat.to(dtype=query_states.dtype)

        full_key_states = repeat_kv(full_key_states, self.num_key_value_groups)
        full_value_states = repeat_kv(full_value_states, self.num_key_value_groups)

        full_key_states = full_key_states.to(dtype=query_states.dtype)
        full_value_states = full_value_states.to(dtype=query_states.dtype)

        # 13. attention
        attn_weights = torch.matmul(
            query_states, full_key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            mask_slice = attention_mask[:, :, :, :total_seq_len].to(target_dtype)
            attn_weights = attn_weights + mask_slice

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(target_dtype)

        attn_output = torch.matmul(attn_weights, full_value_states)

        print(f"[VERIFY-ATTN] Layer {self.layer_idx}")
        print(f"  attn_weights nan = {torch.isnan(attn_weights).any().item()}")
        print(f"  attn_weights inf = {torch.isinf(attn_weights).any().item()}")
        print(f"  attn_output nan = {torch.isnan(attn_output).any().item()}")
        print(f"  attn_output inf = {torch.isinf(attn_output).any().item()}")
        print(f"  attn_output abs max = {attn_output.abs().max().item()}")

        # 14. output
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output).to(original_dtype)
        print(f"[VERIFY-O-PROJ] Layer {self.layer_idx}")
        print(f"  output nan = {torch.isnan(attn_output).any().item()}")
        print(f"  output inf = {torch.isinf(attn_output).any().item()}")
        print(f"  output abs max = {attn_output.abs().max().item()}")

        return attn_output, None
    
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PagedLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_table=getattr(past_key_values, "block_table", None),
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.page_pool = None
        self.block_table = None

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if self.page_pool is not None and self.block_table is not None:
                past_key_values = PagedCacheShim(
                    config=self.config,
                    page_pool=self.page_pool,
                    block_table=self.block_table,
                )
            else:
                past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if use_cache and isinstance(past_key_values, PagedCacheShim):
            # 이번 forward까지 처리한 총 길이 기록
            past_key_values.update_seen_tokens(int(cache_position[-1].item()) + 1)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]