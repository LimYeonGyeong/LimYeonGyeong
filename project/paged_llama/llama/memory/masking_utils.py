# masking_utils.py

import torch

def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    """
    Creates a causal (lower triangular) mask for sequence of length (past + current).
    """
    batch_size, seq_length = input_ids_shape
    # total length = past + current
    total_length = past_key_values_length + seq_length

    # [1, 1, seq_length, total_length]
    mask = torch.full((1, 1, seq_length, total_length), float("-inf"), device=device)
    mask = torch.tril(mask, diagonal=past_key_values_length)
    return mask.to(dtype)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
    """
    Expands an attention_mask from shape [batch, seq] into [batch, 1, tgt_seq, src_seq]
    with 0 for keep and -inf for masked positions.
    """
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    if tgt_len is not None:
        mask = mask[:, :, :tgt_len, :]
    inverted = 1.0 - mask
    return inverted.to(dtype) * torch.finfo(dtype).min


def create_causal_mask(
    config,
    input_embeds=None,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
):
    """
    Build a causal mask for Llama-style autoregressive attention.

    Args:
        config: LlamaConfig
        input_embeds: Tensor of shape [batch, seq, hidden]
        attention_mask: Tensor of shape [batch, seq] with 1 for tokens to attend to
        cache_position: Tensor of shape [seq] representing positions already in cache
        past_key_values: DynamicCache, used to compute past length
        position_ids: Tensor of shape [batch, seq]

    Returns:
        Tensor of shape [batch, 1, seq, total_seq]
    """

    # Determine current and past lengths
    seq_length = input_embeds.shape[1]
    past_key_length = past_key_values.get_seq_length() if past_key_values is not None else 0

    # First, get the causal mask (lower triangular) including past states
    causal_mask = _make_causal_mask(
        (input_embeds.shape[0], seq_length),
        torch.float32,
        input_embeds.device,
        past_key_values_length=past_key_length,
    )

    # Next, optionally expand the given attention_mask
    if attention_mask is not None:
        # attention_mask shape: [batch, seq]
        expanded_attn_mask = _expand_mask(
            attention_mask,
            dtype=causal_mask.dtype,
            tgt_len=causal_mask.shape[-1],
        )
        # Combine
        causal_mask = causal_mask + expanded_attn_mask

    return causal_mask
