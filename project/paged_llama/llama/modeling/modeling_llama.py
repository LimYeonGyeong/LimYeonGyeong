import torch
from torch import nn
import math
from typing import Optional, Callable, Any, Union, List

# llama 바로 아래
from ..activations import ACT2FN

# memory / generation
from ..memory.cache_utils import Cache
from ..memory.masking_utils import create_causal_mask
from ..memory.page_cache import PagedCache
from ..generation.generation import GenerationMixin

# modeling 내부
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

# transformers 표준 출력
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# utils (llama/utils)
from ..utils.processing_utils import Unpack
from ..utils.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    use_kernel_forward_from_hub,
    use_kernelized_func,
    use_kernel_func_from_hub,
    check_model_inputs,
)
from ..utils import logging

# config
from ..config.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

def _tensor_debug(name, x, layer_idx=None, step_info=""):
    if x is None:
        print(f"[DBG] {name}: None")
        return

    with torch.no_grad():
        x_detached = x.detach()
        is_floating = torch.is_floating_point(x_detached)

        nan_count = torch.isnan(x_detached).sum().item() if is_floating else 0
        inf_count = torch.isinf(x_detached).sum().item() if is_floating else 0

        if x_detached.numel() > 0:
            x_float = x_detached.float() if is_floating else x_detached.to(torch.float32)
            min_val = x_float.min().item()
            max_val = x_float.max().item()
            mean_val = x_float.mean().item()
            abs_max = x_float.abs().max().item()
        else:
            min_val = max_val = mean_val = abs_max = 0.0

        prefix = f"[DBG][Layer {layer_idx}] " if layer_idx is not None else "[DBG] "
        print(
            f"{prefix}{name} {step_info}"
            f" | shape={tuple(x_detached.shape)} dtype={x_detached.dtype} device={x_detached.device}"
            f" | min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f} absmax={abs_max:.6f}"
            f" | nan={nan_count} inf={inf_count}"
        )


def _assert_no_nan(name, x, layer_idx=None, step_info=""):
    if x is None or not torch.is_floating_point(x):
        return
    if torch.isnan(x).any():
        _tensor_debug(name, x, layer_idx=layer_idx, step_info=step_info)
        raise RuntimeError(f"[NUMERIC ERROR] {name} has NaN at layer={layer_idx} {step_info}")


def _assert_no_posinf(name, x, layer_idx=None, step_info=""):
    if x is None or not torch.is_floating_point(x):
        return
    if torch.isposinf(x).any():
        _tensor_debug(name, x, layer_idx=layer_idx, step_info=step_info)
        raise RuntimeError(f"[NUMERIC ERROR] {name} has +Inf at layer={layer_idx} {step_info}")

    

class PagedCacheShim:
    """
    HF generate()가 요구하는 최소 cache 인터페이스만 흉내내는 객체.
    실제 KV는 저장하지 않고, '현재까지 본 토큰 수'만 관리한다.
    """
    def __init__(
        self,
        config=None,
        page_pool=None,
        block_table=None,
        request_state=None,
        request_states=None,
        seen_tokens: int = 0,
    ):
        self.config = config
        self.page_pool = page_pool
        self.block_table = block_table
        self.request_state = request_state
        self.request_states = request_states
        self.seen_tokens = int(seen_tokens)
        self.debug = False

        if self.request_states:
            max_seen = max(int(rs.get("seq_len", 0)) for rs in self.request_states)
            self.seen_tokens = int(max_seen)
        elif self.request_state is not None:
            self.seen_tokens = int(self.request_state.get("seq_len", self.seen_tokens))

        if self.debug:
            print(
                f"[PagedCacheShim.__init__] id={id(self)} "
                f"seen_tokens={self.seen_tokens} "
                f"request_state_id={id(self.request_state) if self.request_state is not None else None} "
                f"num_request_states={len(self.request_states) if self.request_states is not None else None}"
            )

    def get_seq_length(self, layer_idx: int = 0):
        if self.request_states:
            seq = max(int(rs.get("seq_len", 0)) for rs in self.request_states)
            if self.debug:
                print(
                    f"[PagedCacheShim.get_seq_length] id={id(self)} "
                    f"return(request_states_max)={seq}"
                )
            return seq

        if self.request_state is not None:
            seq = int(self.request_state.get("seq_len", self.seen_tokens))
            if self.debug:
                print(
                    f"[PagedCacheShim.get_seq_length] id={id(self)} "
                    f"return(request_state)={seq} "
                    f"request_state_id={id(self.request_state)}"
                )
            return seq

        if self.debug:
            print(f"[PagedCacheShim.get_seq_length] id={id(self)} return(seen_tokens)={self.seen_tokens}")
        return self.seen_tokens

    def get_max_cache_shape(self):
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx):
        return self

    def update_seen_tokens(self, new_len):
        if self.debug:
            print(
                f"[PagedCacheShim.update_seen_tokens] id={id(self)} "
                f"before seen_tokens={self.seen_tokens} "
                f"new_len={new_len}"
            )

        if isinstance(new_len, (list, tuple)):
            if self.request_states is None:
                raise RuntimeError("[CACHE ERROR] new_len is list/tuple but request_states is None")
            for rs, nl in zip(self.request_states, new_len):
                rs["seq_len"] = int(nl)
            self.seen_tokens = max(int(nl) for nl in new_len) if len(new_len) > 0 else 0
        elif torch.is_tensor(new_len) and new_len.dim() > 0:
            values = [int(x.item()) for x in new_len.reshape(-1)]
            if self.request_states is not None and len(values) == len(self.request_states):
                for rs, nl in zip(self.request_states, values):
                    rs["seq_len"] = int(nl)
            elif self.request_state is not None and len(values) > 0:
                self.request_state["seq_len"] = int(values[0])
            self.seen_tokens = max(values) if len(values) > 0 else self.seen_tokens
        else:
            self.seen_tokens = int(new_len)

            # multi-request에서는 scalar overwrite 금지
            if self.request_states:
                pass

            elif self.request_state is not None:
                self.request_state["seq_len"] = int(new_len)

        if self.debug:
            print(
                f"[PagedCacheShim.update_seen_tokens] id={id(self)} "
                f"after seen_tokens={self.seen_tokens}"
            )

    def __repr__(self):
        return (
            f"PagedCacheShim(id={id(self)}, "
            f"seen_tokens={self.seen_tokens}, "
            f"seq_len={self.get_seq_length()})"
        )

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

        rope_params = getattr(self.config, "rope_parameters", None)
        if rope_params is None:
            rope_params = {
                "rope_type": "default",
                "rope_theta": getattr(self.config, "rope_theta", 10000.0),
            }

        self.rope_type = rope_params.get("rope_type", "default")

        # config 안에도 넣어둬서 아래 compute_default_rope_parameters에서 그대로 쓰게 함
        self.config.rope_parameters = rope_params

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
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params is None:
            base = getattr(config, "rope_theta", 10000.0)
        else:
            base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
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
    **kwargs,
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        print(
            f"[ATTN SHAPE][Layer {self.layer_idx}] "
            f"hidden={hidden_states.shape} "
            f"query={query_states.shape} "
            f"key={key_states.shape}"
        )
        
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
        
        self.scaling = self.head_dim**-0.5
        
        self.page_pool = page_pool

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.debug = False
        self.debug_verbose = False
        self.debug_stop_on_nonfinite = True

    def _normalize_block_tables(self, block_table, bsz, target_device):
        target_block_table = block_table
        if target_block_table is None:
            target_block_table = getattr(self, "block_table", None)

        if target_block_table is None:
            raise ValueError(f"Layer {self.layer_idx} | block_table is None")

        if isinstance(target_block_table, (list, tuple)):
            if len(target_block_table) != bsz:
                raise RuntimeError(
                    f"Layer {self.layer_idx} | block_table list size {len(target_block_table)} != batch size {bsz}"
                )
            tables = []
            for bt in target_block_table:
                if hasattr(bt, "to_tensor"):
                    tables.append(bt.to_tensor(device=target_device))
                else:
                    tables.append(bt.to(device=target_device))
            return tables

        if hasattr(target_block_table, "to_tensor"):
            single = target_block_table.to_tensor(device=target_device)
        else:
            single = target_block_table.to(device=target_device)

        return [single for _ in range(bsz)]

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
        print("ATTN INPUT")
        print("hidden_states.shape =", hidden_states.shape)
        target_device = self.q_proj.weight.device
        bsz, q_len, _ = hidden_states.size()
        pool = self.page_pool
        print("bsz =", bsz)
        print("q_len =", q_len)
        # 1. Projection
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(target_device), sin.to(target_device))
        print("hidden_states.shape =", hidden_states.shape)

        print("query_states.shape =", query_states.shape)

        print("key_states.shape =", key_states.shape)

        print("value_states.shape =", value_states.shape)

        print("bsz =", bsz)
        print("q_len =", q_len)
        # 2. block_table 텐서 변환
        if isinstance(block_table, list):
            processed_tables = []
            for bt in block_table:
                if hasattr(bt, "to_tensor"): processed_tables.append(bt.to_tensor(device=target_device))
                else: processed_tables.append(bt.to(target_device) if hasattr(bt, 'to') else torch.tensor(bt, device=target_device))
            bt_tensor = torch.stack(processed_tables).squeeze(1)
        else:
            bt_tensor = block_table.to_tensor(device=target_device) if hasattr(block_table, "to_tensor") else block_table.to(target_device)

        print("block_table type =", type(block_table))
        print("bt_tensor.shape =", bt_tensor.shape)
        print("bt_tensor =", bt_tensor)

        # 3. WRITE 로직
        print("\n========== DEBUG ==========")
        print("cache_position.shape =", cache_position.shape)
        print("cache_position =", cache_position)
        print("q_len =", q_len)

        abs_pos = cache_position.unsqueeze(1) + torch.arange(q_len, device=target_device).unsqueeze(0)

        print("abs_pos.shape =", abs_pos.shape)
        print("===========================\n")
        block_idx_logic = abs_pos // pool.block_size
        block_offset = abs_pos % pool.block_size
        batch_indices = torch.arange(bsz, device=target_device).unsqueeze(1)
        physical_block_idx_per_token = bt_tensor[batch_indices, block_idx_logic]

        pool.k_cache[self.layer_idx, physical_block_idx_per_token, :, block_offset, :] = key_states.transpose(1, 2).to(pool.k_cache.dtype)
        pool.v_cache[self.layer_idx, physical_block_idx_per_token, :, block_offset, :] = value_states.transpose(1, 2).to(pool.v_cache.dtype)

        # 4. READ 및 차원 정렬
        # pool.k_cache shape: [num_layers, max_num_blocks, num_heads, block_size, head_dim]
        # bt_tensor shape: [bsz, max_blocks]
        # k_blocks 인덱싱 후 shape: [bsz, max_blocks, num_heads, block_size, head_dim]
        print("bt_tensor.shape =", bt_tensor.shape)

        k_blocks = pool.k_cache[self.layer_idx, bt_tensor]
        v_blocks = pool.v_cache[self.layer_idx, bt_tensor]

        print("k_blocks.shape =", k_blocks.shape)
        print("v_blocks.shape =", v_blocks.shape)
        # [핵심] 배치 차원이 0번에 오도록 확실히 고정합니다.
        # k_blocks: [bsz, max_blocks, num_heads, block_size, head_dim]
        # bsz를 0번으로 유지하고, num_heads를 1번으로 가져오기 위해 permute
        k_blocks = k_blocks.permute(0, 2, 1, 3, 4)
        v_blocks = v_blocks.permute(0, 2, 1, 3, 4)

        print("k_blocks after permute =", k_blocks.shape)

        # 이제 bsz가 0번에 있으므로 reshape이 안전합니다.
        # -1은 자동으로 max_blocks * block_size를 계산합니다.
        k_blocks_flat = k_blocks.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)
        v_blocks_flat = v_blocks.reshape(bsz, self.num_key_value_heads, -1, self.head_dim)

        # 5. 헤드 확장 (repeat_kv)
        k_blocks_flat = repeat_kv(k_blocks_flat, self.num_key_value_groups)
        v_blocks_flat = repeat_kv(v_blocks_flat, self.num_key_value_groups)

        # 6. Attention 연산
        attn_weights = torch.matmul(query_states, k_blocks_flat.transpose(-2, -1)) * self.scaling
        
        # 5. 요청별 동적 마스킹
        # [bsz, 1, 1, total_kv_len]
        total_kv_len = k_blocks_flat.size(2)
        token_idx = torch.arange(total_kv_len, device=target_device).view(1, 1, 1, -1)
        # 요청별 seq_len 사용 (cache_position은 요청별 현재 토큰 위치)
        mask = token_idx < (cache_position.unsqueeze(1) + q_len).view(-1, 1, 1, 1)
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 6. Value Accumulation
        # 7. Value Accumulation 및 결과 반환
        attn_output = torch.matmul(attn_weights, v_blocks_flat)
        
        # --- 디버그 코드 추가 ---
        print(f"\n[DEBUG] attn_output.shape before reshape: {attn_output.shape}")
        print(f"[DEBUG] bsz={bsz}, q_len={q_len}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        print(f"[DEBUG] expected total size: {bsz * q_len * self.num_heads * self.head_dim}")
        # ----------------------
        print("attn_weights =", attn_weights.shape)
        print("value_states =", value_states.shape)
        print("attn_output =", attn_output.shape)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 2. bsz와 q_len을 유지하면서, 나머지 차원(num_heads * head_dim)을 하나로 합침
        # view 대신 flatten을 사용하여 [bsz, q_len, -1]로 만들면, 
        # 뒤의 2차원(num_heads * head_dim)이 자동으로 2048로 계산됩니다.
        attn_output = attn_output.flatten(2, 3) 
        
        return self.o_proj(attn_output), None
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
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        print("\n===== DECODER LAYER =====")
        print("layer hidden_states =", hidden_states.shape)
        print("attention_mask =", None if attention_mask is None else attention_mask.shape)
        print("position_ids =", None if position_ids is None else position_ids.shape)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            block_table=getattr(self.self_attn, "block_table", None),
            **kwargs,
        )
        print("after self_attn =", hidden_states.shape)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
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
        self.page_pool = None

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.page_pool = None
        self.block_table = None

        # Initialize weights and apply final processing
        if hasattr(self, "post_init"):
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
        **kwargs,
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        active_request_states = getattr(self, "active_request_states", None)
        active_request_state = getattr(self, "active_request_state", None)

        if use_cache:
            if past_key_values is None:
                past_key_values = PagedCacheShim(
                    config=self.config,
                    page_pool=self.page_pool,
                    block_table=self.block_table,
                    request_state=active_request_state,
                    request_states=active_request_states,
                )
            elif not isinstance(past_key_values, PagedCacheShim):
                raise RuntimeError(
                    f"[CACHE ERROR] expected PagedCacheShim, got {type(past_key_values)}"
                )
            else:
                past_key_values.request_state = active_request_state
                past_key_values.request_states = active_request_states
                past_key_values.block_table = self.block_table
                past_key_values.page_pool = self.page_pool

        if cache_position is None:
            if active_request_states:
                seq_len = inputs_embeds.shape[1]
                rows = []
                for rs in active_request_states:
                    start = int(rs.get("seq_len", 0))
                    rows.append(
                        torch.arange(start, start + seq_len, device=inputs_embeds.device, dtype=torch.long)
                    )
                cache_position = torch.stack(rows, dim=0)
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

        if position_ids is None:
            position_ids = cache_position if cache_position.dim() == 2 else cache_position.unsqueeze(0)

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
            if cache_position.dim() == 2:
                new_seen_tokens = [int(cache_position[row, -1].item()) + 1 for row in range(cache_position.shape[0])]
                past_key_values.update_seen_tokens(new_seen_tokens)
                if active_request_states:
                    for rs, new_len in zip(active_request_states, new_seen_tokens):
                        rs["seq_len"] = int(new_len)
            else:
                new_seen_tokens = int(cache_position[-1].item()) + 1
                past_key_values.update_seen_tokens(new_seen_tokens)
                if active_request_state is not None:
                    active_request_state["seq_len"] = new_seen_tokens

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
        if hasattr(self, "post_init"):
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
        **kwargs,
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