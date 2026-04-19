import os
import gc
import sys
import time
import builtins
from contextlib import contextmanager

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append('/LimYeonGyeong/project')

from paged_llama.llama.modeling.modeling_llama import (
    PagedLlamaAttention,
    LlamaForCausalLM as PagedLlamaForCausalLM,
)
from paged_llama.llama.config.configuration_llama import LlamaConfig
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 실행 중 너무 많이 찍히는 디버그 출력만 걸러낸다.
# - measure_paged_multi() 내부의 step-by-step 로그
# - modeling_llama.py 내부의 잔여 디버그 로그
# 최종 결과, 로딩 메시지, 성능 표는 그대로 둔다.
DEBUG_PREFIXES_TO_SUPPRESS = (
    '[MAIN-PREFILL]',
    '[MAIN-DECODE STEP',
    '[CACHE-ID]',
    '[CACHE-TYPE]',
    '[CACHE-SEQ]',
    '[CACHE-POS]',
    '[CACHE-UPDATE]',
    '[REQ-STATE]',
    '[REQ-STATE-ID]',
    '[POSITION-IDS]',
    '[CHECK]',
    '[MODEL-ENTRY]',
    '[MODEL-EXIT]',
    '[LM-HEAD-ENTRY]',
    '[LM-HEAD-EXIT]',
    '[PagedCacheShim',
    '[DBG]',
)


@contextmanager
def suppress_debug_prints(enabled: bool = True):
    if not enabled:
        yield
        return

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if not args:
            return original_print(*args, **kwargs)

        text = ' '.join(str(a) for a in args)
        if text.startswith(DEBUG_PREFIXES_TO_SUPPRESS):
            return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


def build_local_config_from_hf(hf_config):
    return LlamaConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
        hidden_act=hf_config.hidden_act,
        pad_token_id=hf_config.pad_token_id,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        attention_bias=getattr(hf_config, 'attention_bias', False),
        mlp_bias=getattr(hf_config, 'mlp_bias', False),
        attention_dropout=getattr(hf_config, 'attention_dropout', 0.0),
    )


def load_paged_model():
    print('>>> HF config / state_dict 로딩 중...')

    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
    )
    hf_model = hf_model.to('cpu')
    state_dict = hf_model.state_dict()

    print('>>> 로컬 paged 모델 생성 중...')

    local_config = build_local_config_from_hf(hf_config)
    model = PagedLlamaForCausalLM(local_config)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'[LOAD] missing keys: {len(missing)}')
    print(f'[LOAD] unexpected keys: {len(unexpected)}')

    del hf_model
    del state_dict
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    print('>>> 로컬 paged 모델 GPU 이동 중...')
    if device == 'cuda':
        model = model.half()
    model = model.to(device)

    model.config.use_cache = True
    return model


def make_prompt(topic: str, detail_level: str) -> str:
    if detail_level == 'short':
        return (
            '### Instruction:\n'
            f'Explain {topic} in one short sentence.\n'
            '### Response:\n'
        )

    if detail_level == 'medium':
        return (
            '### Instruction:\n'
            f'Explain {topic} clearly for a beginner.\n'
            'Use 3 to 4 simple sentences and include one example.\n'
            '### Response:\n'
        )

    if detail_level == 'long':
        return (
            '### Instruction:\n'
            f'Explain {topic} in detail for a student who is learning LLM systems.\n'
            'Your answer should include:\n'
            '1. a simple definition,\n'
            '2. why it matters,\n'
            '3. one technical detail,\n'
            '4. one practical example,\n'
            '5. one limitation.\n'
            'Write around 8 to 10 sentences.\n'
            '### Response:\n'
        )

    raise ValueError(f'Unknown detail_level: {detail_level}')


def build_mixed_prompts(n_requests: int = 20):
    topics = [
        'LLM', 'transformer attention', 'KV cache', 'paged attention', 'multi-head attention',
        'inference', 'training', 'prefill and decode', 'block table', 'page pool',
        'GPU memory fragmentation', 'sequence length', 'context window', 'RoPE embeddings',
        'beam search', 'greedy decoding', 'tokenization', 'causal mask', 'batch inference',
        'dynamic cache', 'static cache', 'request scheduling', 'latency', 'throughput',
        'memory efficiency',
    ]

    detail_pattern = (
        ['short'] * (n_requests // 3)
        + ['medium'] * (n_requests // 3)
        + ['long'] * (n_requests - 2 * (n_requests // 3))
    )

    prompts = []
    for i in range(n_requests):
        prompts.append(make_prompt(topics[i % len(topics)], detail_pattern[i]))
    return prompts


@torch.no_grad()
def measure_baseline_multi(model, tokenizer, prompts, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    t0 = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()
    generated_tokens = (outputs.shape[1] - inputs['input_ids'].shape[1]) * inputs['input_ids'].shape[0]

    if device == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    else:
        peak_allocated = current_allocated = current_reserved = max_reserved = 0.0

    return {
        'latency': t1 - t0,
        'throughput': generated_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0,
        'ram_mb': ram_end - ram_start,
        'ctx_switches': (ctx_end.voluntary - ctx_start.voluntary) + (ctx_end.involuntary - ctx_start.involuntary),
        'peak_vram_mb': peak_allocated,
        'alloc_vram_mb': current_allocated,
        'reserved_vram_mb': current_reserved,
        'max_reserved_vram_mb': max_reserved,
        'vram_per_token_kb': (peak_allocated * 1024 / generated_tokens) if generated_tokens > 0 else 0.0,
        'used_blocks': 0,
        'block_utilization': 0.0,
    }


def patch_model_with_paged_attention(model, page_pool, block_table, debug=False, debug_verbose=False):
    # model-level cache info 연결
    model.model.page_pool = page_pool
    model.model.block_table = block_table

    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        new_attn = PagedLlamaAttention(
            config=model.config,
            layer_idx=layer_idx,
            page_pool=page_pool,
        )

        new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
        new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

        new_attn.block_table = block_table
        new_attn.debug = debug
        new_attn.debug_verbose = debug_verbose
        new_attn.debug_stop_on_nonfinite = False

        ref_param = next(old_attn.parameters())
        new_attn = new_attn.to(device=ref_param.device, dtype=ref_param.dtype)
        layer.self_attn = new_attn

    return model


@torch.no_grad()
def measure_paged_multi(model, tokenizer, prompts, scheduler, pool, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    request_ids = []
    block_tables = []
    encoded_prompts = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors='pt')
        encoded_prompts.append({
            'input_ids': enc['input_ids'].to(model.device),
            'prompt_len': enc['input_ids'].shape[1],
        })

    for i, item in enumerate(encoded_prompts):
        rid = f'multi_req_{i}'
        request_ids.append(rid)
        total_tokens = item['prompt_len'] + max_new_tokens
        bt = scheduler.allocate_for_request(rid, total_tokens)
        scheduler.set_seq_len(rid, 0)
        block_tables.append(bt)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    t0 = time.time()
    results = []

    with suppress_debug_prints(enabled=True):
        for i, item in enumerate(encoded_prompts):
            request_id = request_ids[i]
            request_state = scheduler.get_request_state(request_id)

            for layer in model.model.layers:
                layer.self_attn.block_table = block_tables[i]
                layer.self_attn.page_pool = pool
                layer.self_attn.debug = False
                layer.self_attn.debug_verbose = False
            model.model.block_table = block_tables[i]
            model.model.page_pool = pool
            model.model.active_request_state = request_state

            generated = item['input_ids'].clone()
            prompt_len = item['prompt_len']
            past_key_values = None
            scheduler.set_seq_len(request_id, 0)

            outputs = model(
                input_ids=generated,
                use_cache=True,
                past_key_values=None,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            for _step in range(1, max_new_tokens):
                last_token = generated[:, -1:]
                current_pos = int(model.model.active_request_state['seq_len'])
                cache_position = torch.tensor([current_pos], device=model.device, dtype=torch.long)
                position_ids = cache_position.unsqueeze(0)

                outputs = model(
                    input_ids=last_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                    break

            results.append((generated, prompt_len))

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()
    generated_tokens = sum(gen.shape[1] - prompt_len for gen, prompt_len in results)

    if device == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    else:
        peak_allocated = current_allocated = current_reserved = max_reserved = 0.0

    used_blocks = 0
    for bt in block_tables:
        used_blocks += bt.to_tensor(device='cpu').shape[1] if hasattr(bt, 'to_tensor') else len(bt)
    block_utilization = used_blocks / pool.num_blocks if pool.num_blocks > 0 else 0.0

    for rid in request_ids:
        scheduler.release_request(rid)

    decoded_texts = [tokenizer.decode(gen[0], skip_special_tokens=True) for gen, _ in results]

    return {
        'latency': t1 - t0,
        'throughput': generated_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0,
        'ram_mb': ram_end - ram_start,
        'ctx_switches': (ctx_end.voluntary - ctx_start.voluntary) + (ctx_end.involuntary - ctx_start.involuntary),
        'peak_vram_mb': peak_allocated,
        'alloc_vram_mb': current_allocated,
        'reserved_vram_mb': current_reserved,
        'max_reserved_vram_mb': max_reserved,
        'vram_per_token_kb': (peak_allocated * 1024 / generated_tokens) if generated_tokens > 0 else 0.0,
        'used_blocks': used_blocks,
        'block_utilization': block_utilization,
        'texts': decoded_texts,
    }


def print_stats_table(title, stats_base, stats_paged, include_blocks=False):
    print(f'\n=== {title} ===')
    print(f"{'Metric':<25} | {'nomal':<15} | {'Paged':<15} |")
    print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
    print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")
    print(f"{'RAM Increase (MB)':<25} | {stats_base['ram_mb']:<15.1f} | {stats_paged['ram_mb']:<15.1f} |")
    print(f"{'Peak VRAM (MB)':<25} | {stats_base['peak_vram_mb']:<15.1f} | {stats_paged['peak_vram_mb']:<15.1f} |")
    print(f"{'Alloc VRAM (MB)':<25} | {stats_base['alloc_vram_mb']:<15.1f} | {stats_paged['alloc_vram_mb']:<15.1f} |")
    print(f"{'Reserved VRAM (MB)':<25} | {stats_base['reserved_vram_mb']:<15.1f} | {stats_paged['reserved_vram_mb']:<15.1f} |")
    print(f"{'Max Reserved VRAM (MB)':<25} | {stats_base['max_reserved_vram_mb']:<15.1f} | {stats_paged['max_reserved_vram_mb']:<15.1f} |")
    print(f"{'VRAM/token (KB)':<25} | {stats_base['vram_per_token_kb']:<15.2f} | {stats_paged['vram_per_token_kb']:<15.2f} |")
    print(f"{'Context Switch':<25} | {stats_base['ctx_switches']:<15} | {stats_paged['ctx_switches']:<15} |")
    if include_blocks:
        print(f"{'Used Blocks':<25} | {'-':<15} | {stats_paged['used_blocks']:<15} |")
        print(f"{'Block Utilization':<25} | {'-':<15} | {stats_paged['block_utilization']:<15.2%} |")


def multi_only_main():
    print('>>> Multi Request 비교 실행')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    num_requests = 20
    prompts = build_mixed_prompts(n_requests=num_requests)
    max_new_tokens = 20
    block_size = 16

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    print('>>> Baseline multi 로딩 중...')
    model_base_multi = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    stats_base_multi = measure_baseline_multi(model_base_multi, tokenizer, prompts, max_new_tokens=max_new_tokens)

    del model_base_multi
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    print('>>> Paged multi 로딩 중...')
    model_paged = load_paged_model()
    config = model_paged.config

    all_needed_blocks = 0
    for p in prompts:
        prompt_len = tokenizer(p, return_tensors='pt')['input_ids'].shape[1]
        needed_tokens = prompt_len + max_new_tokens
        needed_blocks = (needed_tokens + block_size - 1) // block_size
        all_needed_blocks += needed_blocks
    num_blocks = all_needed_blocks + 4

    pool = PagePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=model_paged.dtype,
    )
    scheduler = SimpleScheduler(page_pool=pool, block_size=block_size)

    dummy_prompt_len = tokenizer(prompts[0], return_tensors='pt')['input_ids'].shape[1]
    dummy_total_tokens = dummy_prompt_len + max_new_tokens
    dummy_block_table = scheduler.allocate_for_request('dummy_req', dummy_total_tokens)

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=dummy_block_table,
        debug=False,
        debug_verbose=False,
    )
    scheduler.release_request('dummy_req')

    pool.k_cache.zero_()
    pool.v_cache.zero_()

    stats_paged_multi = measure_paged_multi(
        model_paged, tokenizer, prompts, scheduler, pool, max_new_tokens=max_new_tokens
    )

    print('\n[OUTPUT] Multi Request Generation Results (first 5 only)')
    for i, text in enumerate(stats_paged_multi['texts'][:5]):
        print(f'\n--- Request {i+1} ---')
        print(text[:500])

    print_stats_table('Multi Request Result', stats_base_multi, stats_paged_multi, include_blocks=True)

    del model_paged
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    multi_only_main()
