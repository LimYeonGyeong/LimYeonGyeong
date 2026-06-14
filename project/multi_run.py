import os
import gc
import sys
import time
import math
import random
import argparse

import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append("/LimYeonGyeong/project")

from paged_llama.llama.modeling.modeling_llama import (
    LlamaForCausalLM as PagedLlamaForCausalLM,
)
from paged_llama.llama.config.configuration_llama import LlamaConfig
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

from main import patch_model_with_paged_attention

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Utils
# =========================================================

def model_device(model):
    return next(model.parameters()).device


def dtype_nbytes(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    if dtype == torch.float64:
        return 8
    if dtype in (torch.int8, torch.uint8, torch.bool):
        return 1
    if dtype in (torch.int16, torch.short):
        return 2
    if dtype in (torch.int32, torch.int):
        return 4
    if dtype in (torch.int64, torch.long):
        return 8
    return 2


def reset_cuda_stats_for_measurement():
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def read_cuda_stats_mb():
    if device != "cuda":
        return {
            "peak_vram_mb": 0.0,
            "alloc_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "max_reserved_vram_mb": 0.0,
        }

    torch.cuda.synchronize()

    return {
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "alloc_vram_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_vram_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_reserved_vram_mb": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }


def estimate_kv_cache_mb(num_layers, num_kv_heads, head_dim, tokens, dtype):
    # K와 V 두 개이므로 * 2
    bytes_used = (
        int(num_layers)
        * int(num_kv_heads)
        * int(head_dim)
        * int(tokens)
        * 2
        * dtype_nbytes(dtype)
    )
    return bytes_used / 1024 / 1024


def get_pool_cache_mb(pool):
    return (
        pool.k_cache.numel() * pool.k_cache.element_size()
        + pool.v_cache.numel() * pool.v_cache.element_size()
    ) / 1024 / 1024


def get_block_table_len(block_table):
    if hasattr(block_table, "to_tensor"):
        return int(block_table.to_tensor(device="cpu").shape[1])
    if hasattr(block_table, "block_table"):
        return int(len(block_table.block_table[0]))
    if hasattr(block_table, "physical_blocks"):
        return int(len(block_table.physical_blocks))
    return 0


# =========================================================
# Config / Model Load
# =========================================================

def build_local_config_from_hf(hf_config):
    return LlamaConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=getattr(
            hf_config,
            "num_key_value_heads",
            hf_config.num_attention_heads,
        ),
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        hidden_act=hf_config.hidden_act,
        pad_token_id=hf_config.pad_token_id,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        attention_bias=getattr(hf_config, "attention_bias", False),
        mlp_bias=getattr(hf_config, "mlp_bias", False),
        attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
    )


def load_baseline_model():
    print(">>> Normal baseline HF 모델 로딩 중...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    return model


def load_paged_model():
    print(">>> HF config / state_dict 로딩 중...")

    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    hf_model = hf_model.to("cpu")
    state_dict = hf_model.state_dict()

    print(">>> 로컬 paged 모델 생성 중...")
    local_config = build_local_config_from_hf(hf_config)
    model = PagedLlamaForCausalLM(local_config)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[LOAD] missing keys: {len(missing)}")
    print(f"[LOAD] unexpected keys: {len(unexpected)}")

    del hf_model
    del state_dict
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    print(">>> 로컬 paged 모델 GPU 이동 중...")
    if device == "cuda":
        model = model.half()

    model = model.to(device)
    model.eval()

    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    return model


# =========================================================
# Prompt Generation
# =========================================================

def make_prompt(question: str, style: str = "plain") -> str:
    if style == "short":
        extra = "Answer in one short sentence."
    elif style == "medium":
        extra = "Answer in three simple sentences."
    elif style == "long":
        extra = "Answer in six to eight clear sentences with one concrete example."
    else:
        extra = "Answer in clear plain English."

    return (
        "### Instruction:\n"
        f"{question}\n"
        f"{extra}\n"
        "Do not use markdown, bullet points, code blocks, or numbered lists.\n"
        "### Response:\n"
    )


def build_mixed_prompts(n_requests: int = 4, seed: int | None = None):
    rng = random.Random(seed)

    question_bank = [
        ("Explain LLM in one sentence.", "short"),
        ("Explain transformer attention in one sentence.", "short"),
        ("Explain why KV cache is useful during LLM inference.", "medium"),
        ("Explain paged attention in simple terms.", "medium"),
        ("Explain the difference between prefill and decode in LLM generation.", "medium"),
        ("Explain why GPU memory access can become a bottleneck during LLM serving.", "medium"),
        ("Explain request-level scheduling in LLM serving.", "medium"),
        ("Explain why batching improves LLM inference throughput.", "medium"),
        (
            "Explain paged attention, KV cache fragmentation, block allocation, and request scheduling in LLM serving.",
            "long",
        ),
        (
            "Explain how continuous batching helps when many users send requests with different prompt lengths and output lengths.",
            "long",
        ),
        (
            "Explain why reducing unnecessary attention computation can improve latency, throughput, and GPU memory efficiency.",
            "long",
        ),
        (
            "Explain how attention helps a transformer understand context across a long input sequence.",
            "long",
        ),
    ]

    if n_requests <= len(question_bank):
        selected = rng.sample(question_bank, n_requests)
    else:
        selected = [rng.choice(question_bank) for _ in range(n_requests)]

    prompts = [make_prompt(question, style) for question, style in selected]

    print("\n================ GENERATED QUESTIONS ================\n")
    for i, (question, style) in enumerate(selected):
        print(f"[Question {i + 1}] ({style}) {question}")
    print()

    return prompts


# =========================================================
# Debug Helper
# =========================================================

def _print_topk_logits(logits, tokenizer, tag: str, k: int = 5):
    topk = torch.topk(logits[:, -1, :], k=k, dim=-1)

    for row in range(topk.indices.shape[0]):
        print(f"\n[TOPK][{tag}] row={row}")
        for rank in range(k):
            token_id = int(topk.indices[row, rank].item())
            decoded = tokenizer.decode([token_id])
            value = float(topk.values[row, rank].item())
            print(
                f"rank={rank + 1} "
                f"token_id={token_id} "
                f"decoded={repr(decoded)} "
                f"logit={value:.4f}"
            )


# =========================================================
# Baseline Measurement
# =========================================================

@torch.no_grad()
def measure_baseline_multi(model, tokenizer, prompts, max_new_tokens=80):
    model.eval()
    process = psutil.Process(os.getpid())

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model_device(model))

    reset_cuda_stats_for_measurement()

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

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    prompt_width = inputs["input_ids"].shape[1]
    generated_tokens = (
        outputs.shape[1] - prompt_width
    ) * inputs["input_ids"].shape[0]

    prompt_tokens_total = int(inputs["attention_mask"].sum().item())
    total_tokens = prompt_tokens_total + int(generated_tokens)

    cuda_stats = read_cuda_stats_mb()
    peak_allocated = cuda_stats["peak_vram_mb"]

    decoded_texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    response_texts = []
    for row in range(outputs.shape[0]):
        response_ids = outputs[row, prompt_width:]
        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        response_texts.append(response_text)

    latency = t1 - t0

    # baseline도 같은 토큰 수 기준의 이론 KV 사용량을 같이 표시한다.
    cfg = model.config
    kv_used_mb = estimate_kv_cache_mb(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        tokens=total_tokens,
        dtype=next(model.parameters()).dtype,
    )

    return {
        "latency": latency,
        "throughput": generated_tokens / latency if latency > 0 else 0.0,
        "ram_mb": ram_end - ram_start,
        "ctx_switches": (
            ctx_end.voluntary - ctx_start.voluntary
            + ctx_end.involuntary - ctx_start.involuntary
        ),
        "peak_vram_mb": cuda_stats["peak_vram_mb"],
        "alloc_vram_mb": cuda_stats["alloc_vram_mb"],
        "reserved_vram_mb": cuda_stats["reserved_vram_mb"],
        "max_reserved_vram_mb": cuda_stats["max_reserved_vram_mb"],
        "vram_per_token_kb": (
            peak_allocated * 1024 / generated_tokens
            if generated_tokens > 0
            else 0.0
        ),
        "used_blocks": None,
        "allocated_blocks": None,
        "pool_blocks": None,
        "block_utilization": None,
        "pool_block_utilization": None,
        "kv_cache_capacity_mb": None,
        "theoretical_kv_used_mb": kv_used_mb,
        "kv_cache_waste_mb": None,
        "kv_cache_waste_ratio": None,
        "texts": decoded_texts,
        "responses": response_texts,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": int(generated_tokens),
        "total_tokens": total_tokens,
        "num_requests": len(prompts),
        "max_new_tokens": max_new_tokens,
    }


# =========================================================
# Paged Measurement
# =========================================================

def bind_paged_static_runtime(model, attn_layers, pool):
    # page_pool은 실행 중 거의 바뀌지 않으므로 한 번만 연결한다.
    for attn in attn_layers:
        attn.page_pool = pool
    model.model.page_pool = pool


def bind_paged_runtime(model, attn_layers, pool, batch_block_tables, batch_request_states):
    # block_table/request_state만 현재 batch에 맞춰 갱신한다.
    # page_pool은 bind_paged_static_runtime()에서 미리 연결하여 반복 비용을 줄인다.
    for attn in attn_layers:
        attn.block_table = batch_block_tables

    model.model.block_table = batch_block_tables
    model.model.active_request_state = None
    model.model.active_request_states = batch_request_states


@torch.no_grad()
def measure_paged_multi(
    model,
    tokenizer,
    prompts,
    scheduler,
    pool,
    max_new_tokens=80,
    enable_token_debug: bool = False,
):
    model.eval()
    process = psutil.Process(os.getpid())
    dev = model_device(model)
    attn_layers = [layer.self_attn for layer in model.model.layers]
    bind_paged_static_runtime(model, attn_layers, pool)

    request_ids = []
    runtimes = []

    # -------------------------------------------------
    # 0) Tokenize
    # -------------------------------------------------
    encoded_inputs = []
    max_prompt_len = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        prompt_len = input_ids.shape[1]
        encoded_inputs.append((input_ids, prompt_len))
        max_prompt_len = max(max_prompt_len, prompt_len)

    shared_total_tokens = max_prompt_len + max_new_tokens

    # -------------------------------------------------
    # 1) Runtime state 준비
    # -------------------------------------------------
    for i, (input_ids, prompt_len) in enumerate(encoded_inputs):
        rid = f"multi_req_{i}"
        request_ids.append(rid)

        block_table = scheduler.allocate_for_request(
            rid,
            shared_total_tokens,
        )
        scheduler.set_seq_len(rid, 0)
        request_state = scheduler.get_request_state(rid)

        runtimes.append({
            "request_id": rid,
            "block_table": block_table,
            "request_state": request_state,
            "generated": input_ids.clone(),
            "prompt_len": prompt_len,
            "finished": False,
        })

    allocated_blocks_initial = sum(
        get_block_table_len(rt["block_table"])
        for rt in runtimes
    )

    # -------------------------------------------------
    # 2) 성능 측정 시작
    # -------------------------------------------------
    reset_cuda_stats_for_measurement()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    t0 = time.time()

    # -------------------------------------------------
    # 3) PREFILL
    # -------------------------------------------------
    prefill_groups = {}
    for rt in runtimes:
        prefill_groups.setdefault(rt["prompt_len"], []).append(rt)

    for prompt_len in sorted(prefill_groups.keys()):
        group_rts = prefill_groups[prompt_len]

        batch_input_ids = torch.cat(
            [rt["generated"] for rt in group_rts],
            dim=0,
        )
        batch_block_tables = [rt["block_table"] for rt in group_rts]
        batch_request_states = [rt["request_state"] for rt in group_rts]

        bind_paged_runtime(
            model=model,
            attn_layers=attn_layers,
            pool=pool,
            batch_block_tables=batch_block_tables,
            batch_request_states=batch_request_states,
        )

        for rt in group_rts:
            scheduler.set_seq_len(rt["request_id"], 0)
            rt["request_state"]["seq_len"] = 0

        outputs = model(
            input_ids=batch_input_ids,
            use_cache=True,
            past_key_values=None,
        )

        if enable_token_debug:
            _print_topk_logits(outputs.logits, tokenizer, tag=f"prefill_len_{prompt_len}")

        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for i, rt in enumerate(group_rts):
            next_token = next_tokens[i:i + 1]
            rt["generated"] = torch.cat([rt["generated"], next_token], dim=1)
            rt["request_state"]["seq_len"] = rt["generated"].shape[1]

            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                rt["finished"] = True

    # -------------------------------------------------
    # 4) DECODE
    # -------------------------------------------------
    prev_active_key = None
    cached_batch_block_tables = None
    cached_batch_request_states = None

    for step in range(1, max_new_tokens):
        active_rts = [rt for rt in runtimes if not rt["finished"]]
        if not active_rts:
            break

        batch_last_tokens = torch.cat(
            [rt["generated"][:, -1:] for rt in active_rts],
            dim=0,
        )

        for rt in active_rts:
            rt["request_state"]["seq_len"] = rt["generated"].shape[1]

        batch_cache_position = torch.tensor(
            [int(rt["request_state"]["seq_len"]) - 1 for rt in active_rts],
            device=dev,
            dtype=torch.long,
        )
        batch_position_ids = batch_cache_position.unsqueeze(1)

        active_key = tuple(rt["request_id"] for rt in active_rts)

        # active request 구성이 바뀌었을 때만 block table을 다시 연결한다.
        # seq_len은 request_state dict 내부 값만 갱신되므로 매 step 재바인딩할 필요가 없다.
        if active_key != prev_active_key:
            cached_batch_block_tables = [rt["block_table"] for rt in active_rts]
            cached_batch_request_states = [rt["request_state"] for rt in active_rts]

            bind_paged_runtime(
                model=model,
                attn_layers=attn_layers,
                pool=pool,
                batch_block_tables=cached_batch_block_tables,
                batch_request_states=cached_batch_request_states,
            )
            prev_active_key = active_key

        outputs = model(
            input_ids=batch_last_tokens,
            use_cache=True,
            past_key_values=None,
            cache_position=batch_cache_position,
            position_ids=batch_position_ids,
        )

        if enable_token_debug:
            _print_topk_logits(outputs.logits, tokenizer, tag=f"decode_step_{step}")

        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for i, rt in enumerate(active_rts):
            next_token = next_tokens[i:i + 1]
            rt["generated"] = torch.cat([rt["generated"], next_token], dim=1)

            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                rt["finished"] = True

    # -------------------------------------------------
    # 5) 성능 측정 종료
    # -------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    prompt_tokens_total = sum(int(rt["prompt_len"]) for rt in runtimes)
    generated_tokens = sum(
        int(rt["generated"].shape[1] - rt["prompt_len"])
        for rt in runtimes
    )
    total_tokens = prompt_tokens_total + generated_tokens

    cuda_stats = read_cuda_stats_mb()
    peak_allocated = cuda_stats["peak_vram_mb"]

    # -------------------------------------------------
    # 6) Paged 전용 메모리 지표
    # -------------------------------------------------
    final_lengths = [int(rt["generated"].shape[1]) for rt in runtimes]
    needed_blocks_final = sum(math.ceil(length / pool.block_size) for length in final_lengths)
    allocated_blocks = allocated_blocks_initial

    token_block_capacity = allocated_blocks * pool.block_size
    token_block_utilization = (
        total_tokens / token_block_capacity
        if token_block_capacity > 0
        else 0.0
    )

    pool_block_utilization = (
        allocated_blocks / pool.num_blocks
        if pool.num_blocks > 0
        else 0.0
    )

    cfg = model.config
    kv_cache_capacity_mb = get_pool_cache_mb(pool)
    theoretical_kv_used_mb = estimate_kv_cache_mb(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        tokens=total_tokens,
        dtype=pool.k_cache.dtype,
    )
    kv_cache_waste_mb = max(kv_cache_capacity_mb - theoretical_kv_used_mb, 0.0)
    kv_cache_waste_ratio = (
        kv_cache_waste_mb / kv_cache_capacity_mb
        if kv_cache_capacity_mb > 0
        else 0.0
    )

    # -------------------------------------------------
    # 7) 디코딩 결과 정리
    # -------------------------------------------------
    decoded_texts = []
    decoded_responses = []
    generated_token_ids = []
    response_token_ids = []

    for rt in runtimes:
        full_ids = rt["generated"][0]
        response_ids = full_ids[rt["prompt_len"]:]

        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        decoded_texts.append(full_text)
        decoded_responses.append(response_text)
        generated_token_ids.append(full_ids.detach().cpu().tolist())
        response_token_ids.append(response_ids.detach().cpu().tolist())

    for rid in request_ids:
        scheduler.release_request(rid)

    latency = t1 - t0

    return {
        "latency": latency,
        "throughput": generated_tokens / latency if latency > 0 else 0.0,
        "ram_mb": ram_end - ram_start,
        "ctx_switches": (
            ctx_end.voluntary - ctx_start.voluntary
            + ctx_end.involuntary - ctx_start.involuntary
        ),
        "peak_vram_mb": cuda_stats["peak_vram_mb"],
        "alloc_vram_mb": cuda_stats["alloc_vram_mb"],
        "reserved_vram_mb": cuda_stats["reserved_vram_mb"],
        "max_reserved_vram_mb": cuda_stats["max_reserved_vram_mb"],
        "vram_per_token_kb": (
            peak_allocated * 1024 / generated_tokens
            if generated_tokens > 0
            else 0.0
        ),
        "used_blocks": needed_blocks_final,
        "allocated_blocks": allocated_blocks,
        "pool_blocks": pool.num_blocks,
        "block_utilization": token_block_utilization,
        "pool_block_utilization": pool_block_utilization,
        "kv_cache_capacity_mb": kv_cache_capacity_mb,
        "theoretical_kv_used_mb": theoretical_kv_used_mb,
        "kv_cache_waste_mb": kv_cache_waste_mb,
        "kv_cache_waste_ratio": kv_cache_waste_ratio,
        "texts": decoded_texts,
        "responses": decoded_responses,
        "generated_token_ids": generated_token_ids,
        "response_token_ids": response_token_ids,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens,
        "total_tokens": total_tokens,
        "num_requests": len(runtimes),
        "max_new_tokens": max_new_tokens,
    }


# =========================================================
# Print Result
# =========================================================

def print_multi_request_result_table(normal_stats, paged_stats):
    print("\n================ NORMAL RESPONSE ONLY ================\n")
    for i, text in enumerate(normal_stats.get("responses", [])):
        print(f"--- Normal Request {i + 1} response ---")
        print(repr(text))
        print()

    print("\n================ PAGED RESPONSE ONLY ================\n")
    for i, text in enumerate(paged_stats.get("responses", [])):
        print(f"--- Paged Request {i + 1} response ---")
        print(repr(text))
        print()

    print("\n================ GPU CHECK ================\n")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"device                    : {device}")
    if torch.cuda.is_available():
        print(f"GPU name                  : {torch.cuda.get_device_name(0)}")
        print(f"current device            : {torch.cuda.current_device()}")
        print(f"normal peak vram          : {normal_stats.get('peak_vram_mb', 0.0):.1f} MB")
        print(f"paged peak vram           : {paged_stats.get('peak_vram_mb', 0.0):.1f} MB")

    def fmt_float(value, digits=2):
        if value is None:
            return "-"
        return f"{value:.{digits}f}"

    def fmt_int(value):
        if value is None:
            return "-"
        return str(value)

    def fmt_percent(value):
        if value is None:
            return "-"
        return f"{value:.2%}"

    rows = [
        ("Latency (sec)", fmt_float(normal_stats.get("latency"), 4), fmt_float(paged_stats.get("latency"), 4)),
        ("Throughput (tok/s)", fmt_float(normal_stats.get("throughput"), 2), fmt_float(paged_stats.get("throughput"), 2)),
        ("RAM Increase (MB)", fmt_float(normal_stats.get("ram_mb"), 1), fmt_float(paged_stats.get("ram_mb"), 1)),
        ("Peak VRAM (MB)", fmt_float(normal_stats.get("peak_vram_mb"), 1), fmt_float(paged_stats.get("peak_vram_mb"), 1)),
        ("Alloc VRAM (MB)", fmt_float(normal_stats.get("alloc_vram_mb"), 1), fmt_float(paged_stats.get("alloc_vram_mb"), 1)),
        ("Reserved VRAM (MB)", fmt_float(normal_stats.get("reserved_vram_mb"), 1), fmt_float(paged_stats.get("reserved_vram_mb"), 1)),
        ("Max Reserved VRAM (MB)", fmt_float(normal_stats.get("max_reserved_vram_mb"), 1), fmt_float(paged_stats.get("max_reserved_vram_mb"), 1)),
        ("VRAM/token (KB)", fmt_float(normal_stats.get("vram_per_token_kb"), 2), fmt_float(paged_stats.get("vram_per_token_kb"), 2)),
        ("Context Switch", fmt_int(normal_stats.get("ctx_switches")), fmt_int(paged_stats.get("ctx_switches"))),
        ("Used Blocks", "-", fmt_int(paged_stats.get("used_blocks"))),
        ("Allocated Blocks", "-", fmt_int(paged_stats.get("allocated_blocks"))),
        ("Pool Blocks", "-", fmt_int(paged_stats.get("pool_blocks"))),
        ("Token Block Utilization", "-", fmt_percent(paged_stats.get("block_utilization"))),
        ("Pool Block Utilization", "-", fmt_percent(paged_stats.get("pool_block_utilization"))),
        ("KV Cache Capacity (MB)", "-", fmt_float(paged_stats.get("kv_cache_capacity_mb"), 2)),
        ("Theoretical KV Used (MB)", fmt_float(normal_stats.get("theoretical_kv_used_mb"), 2), fmt_float(paged_stats.get("theoretical_kv_used_mb"), 2)),
        ("KV Cache Waste (MB)", "-", fmt_float(paged_stats.get("kv_cache_waste_mb"), 2)),
        ("KV Cache Waste Ratio", "-", fmt_percent(paged_stats.get("kv_cache_waste_ratio"))),
        ("Prompt Tokens", fmt_int(normal_stats.get("prompt_tokens_total")), fmt_int(paged_stats.get("prompt_tokens_total"))),
        ("Generated Tokens", fmt_int(normal_stats.get("generated_tokens_total")), fmt_int(paged_stats.get("generated_tokens_total"))),
        ("Total Tokens", fmt_int(normal_stats.get("total_tokens")), fmt_int(paged_stats.get("total_tokens"))),
    ]

    metric_width = 30
    normal_width = 18
    paged_width = 18

    print("\n=== Multi Request Result ===")
    print(
        f"{'Metric':<{metric_width}} | "
        f"{'normal':<{normal_width}} | "
        f"{'Paged':<{paged_width}} |"
    )

    for metric, normal_value, paged_value in rows:
        print(
            f"{metric:<{metric_width}} | "
            f"{normal_value:<{normal_width}} | "
            f"{paged_value:<{paged_width}} |"
        )

    print("\n================ RESULT INTERPRETATION ================\n")
    normal_latency = normal_stats.get("latency") or 0.0
    paged_latency = paged_stats.get("latency") or 0.0
    normal_vram = normal_stats.get("peak_vram_mb") or 0.0
    paged_vram = paged_stats.get("peak_vram_mb") or 0.0

    if normal_latency > 0 and paged_latency > 0:
        print(f"Paged latency ratio        : {paged_latency / normal_latency:.2f}x of normal")
    if normal_vram > 0 and paged_vram > 0:
        print(f"Peak VRAM difference       : {normal_vram - paged_vram:.1f} MB")
    print(
        "Note: This implementation validates Paged Attention's block-based KV cache management. "
        "It is not expected to beat HuggingFace generate() in latency because this code uses Python-level "
        "block-wise attention rather than a CUDA/Triton fused PagedAttention kernel."
    )


# =========================================================
# Main
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_requests", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--token_debug", action="store_true")
    return parser.parse_args()


def multi_only_main():
    args = parse_args()

    print(">>> Multi Request 비교 실행")
    print(f">>> torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f">>> device = {device}")
    print(f">>> num_requests = {args.num_requests}")
    print(f">>> max_new_tokens = {args.max_new_tokens}")
    print(f">>> block_size = {args.block_size}")

    if not torch.cuda.is_available():
        print("[WARNING] CUDA가 잡히지 않아 VRAM 측정이 0으로 나올 수 있습니다.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    prompts = build_mixed_prompts(n_requests=args.num_requests, seed=args.seed)

    print("\n================ PROMPTS ================\n")
    for i, prompt in enumerate(prompts):
        print(f"--- Prompt {i + 1} ---")
        print(prompt)
        print()

    # -------------------------------------------------
    # 1) Normal baseline
    # -------------------------------------------------
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(">>> Normal baseline 실행 중...")
    model_normal = load_baseline_model()

    normal_stats = measure_baseline_multi(
        model=model_normal,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )

    del model_normal
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # 2) Paged model
    # -------------------------------------------------
    print(">>> Paged multi 로딩 중...")
    model_paged = load_paged_model()
    config = model_paged.config

    max_prompt_len = 0
    for prompt in prompts:
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        max_prompt_len = max(max_prompt_len, prompt_len)

    shared_total_tokens = max_prompt_len + args.max_new_tokens
    blocks_per_request = (shared_total_tokens + args.block_size - 1) // args.block_size
    all_needed_blocks = blocks_per_request * len(prompts)
    safety_margin = max(4, len(prompts))
    num_blocks = all_needed_blocks + safety_margin

    print("\n================ PAGED CACHE CONFIG ================\n")
    print(f"max_prompt_len       = {max_prompt_len}")
    print(f"shared_total_tokens  = {shared_total_tokens}")
    print(f"blocks_per_request   = {blocks_per_request}")
    print(f"all_needed_blocks    = {all_needed_blocks}")
    print(f"safety_margin        = {safety_margin}")
    print(f"num_blocks           = {num_blocks}")

    pool = PagePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=args.block_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=model_paged.dtype,
    )

    scheduler = SimpleScheduler(page_pool=pool, block_size=args.block_size)

    dummy_total_tokens = (
        tokenizer(prompts[0], return_tensors="pt")["input_ids"].shape[1]
        + args.max_new_tokens
    )
    dummy_block_table = scheduler.allocate_for_request("dummy_req", dummy_total_tokens)

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=dummy_block_table,
    )

    # patch helper가 debug 속성을 남겨도 최종 측정에서는 출력 제거
    for layer in model_paged.model.layers:
        layer.self_attn.debug = False
        layer.self_attn.debug_verbose = False

    scheduler.release_request("dummy_req")

    paged_stats = measure_paged_multi(
        model=model_paged,
        tokenizer=tokenizer,
        prompts=prompts,
        scheduler=scheduler,
        pool=pool,
        max_new_tokens=args.max_new_tokens,
        enable_token_debug=args.token_debug,
    )

    # -------------------------------------------------
    # 3) 답변 + 성능 표 출력
    # -------------------------------------------------
    print_multi_request_result_table(
        normal_stats=normal_stats,
        paged_stats=paged_stats,
    )

    del model_paged
    del pool
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    multi_only_main()
