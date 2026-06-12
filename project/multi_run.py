import os
import gc
import sys
import time
import math
import random

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
# CUDA / Metric Helpers
# =========================================================

def get_model_device(model):
    return next(model.parameters()).device


def cuda_reset_for_full_measurement():
    """
    모델 로딩 전부터 peak VRAM을 잡기 위한 reset 함수.
    반드시 model load / PagePool 생성 전에 호출한다.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def cuda_current_stats_mb():
    """현재 CUDA allocator 기준 메모리 상태를 MB 단위로 반환."""
    if device != "cuda" or not torch.cuda.is_available():
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


def attach_vram_stats(stats, generated_tokens_total):
    """
    main에서 model load 전부터 reset한 peak VRAM을 stats에 덮어쓴다.
    measure 함수 내부에서 reset하지 않는 것이 핵심이다.
    """
    cuda_stats = cuda_current_stats_mb()

    stats["peak_vram_mb"] = cuda_stats["peak_vram_mb"]
    stats["alloc_vram_mb"] = cuda_stats["alloc_vram_mb"]
    stats["reserved_vram_mb"] = cuda_stats["reserved_vram_mb"]
    stats["max_reserved_vram_mb"] = cuda_stats["max_reserved_vram_mb"]

    stats["vram_per_token_kb"] = (
        cuda_stats["peak_vram_mb"] * 1024 / generated_tokens_total
        if generated_tokens_total > 0
        else 0.0
    )

    return stats


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

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

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

def make_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        f"{question}\n"
        "Answer in clear plain English.\n"
        "Do not use markdown, bullet points, code blocks, or numbered lists.\n"
        "### Response:\n"
    )


def build_mixed_prompts(n_requests: int = 2, seed=None):
    rng = random.Random(seed)

    question_bank = [
        "Explain what a large language model is in one clear sentence.",
        "Explain transformer attention for a beginner in three simple sentences.",
        "Explain why KV cache is useful during LLM inference.",
        "Explain paged attention in simple terms.",
        "Explain why batching improves LLM inference throughput.",
        "Explain the difference between prefill and decode in LLM generation.",
        "Explain why memory usage matters when serving large language models.",
        "Explain how attention helps a transformer understand context.",
        "Explain what token generation means in an autoregressive language model.",
        "Explain why reducing unnecessary attention computation can improve latency.",
        "Explain what a cache miss means in a memory system.",
        "Explain why GPU memory access can become a bottleneck.",
        "Explain what sequence length means in transformer inference.",
        "Explain what request-level scheduling means in LLM serving.",
        "Explain why masking is used in causal language models.",
    ]

    if n_requests <= len(question_bank):
        selected_questions = rng.sample(question_bank, n_requests)
    else:
        selected_questions = [rng.choice(question_bank) for _ in range(n_requests)]

    prompts = [make_prompt(question) for question in selected_questions]

    print("\n================ GENERATED QUESTIONS ================\n")

    for i, question in enumerate(selected_questions):
        print(f"[Question {i + 1}] {question}")

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
def measure_baseline_multi(model, tokenizer, prompts, max_new_tokens=40):
    """
    HF generate() 기반 normal baseline 측정.
    주의: 여기서는 reset_peak_memory_stats()를 호출하지 않는다.
    VRAM peak는 multi_only_main()에서 model load 전 reset한 값 기준으로 측정한다.
    """
    model.eval()
    process = psutil.Process(os.getpid())

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    model_device = get_model_device(model)
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    prompt_tokens_total = int(attention_mask.sum().item())

    if device == "cuda":
        torch.cuda.synchronize()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    t0 = time.perf_counter()

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    ctx_end = process.num_ctx_switches()
    ram_end = process.memory_info().rss / 1024 / 1024

    latency = t1 - t0

    prompt_width = input_ids.shape[1]
    generated_tokens_total = int(
        outputs.shape[0] * max(outputs.shape[1] - prompt_width, 0)
    )

    total_tokens = prompt_tokens_total + generated_tokens_total

    throughput = (
        generated_tokens_total / latency
        if latency > 0
        else 0.0
    )

    context_switch = (
        ctx_end.voluntary - ctx_start.voluntary
        + ctx_end.involuntary - ctx_start.involuntary
    )

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

    return {
        "latency": latency,
        "throughput": throughput,
        "ram_increase_mb": ram_end - ram_start,

        "peak_vram_mb": 0.0,
        "alloc_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
        "max_reserved_vram_mb": 0.0,
        "vram_per_token_kb": 0.0,

        "context_switch": context_switch,
        "used_blocks": None,
        "block_utilization": None,

        "texts": decoded_texts,
        "responses": response_texts,

        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "total_tokens": total_tokens,

        "num_requests": len(prompts),
        "max_new_tokens": max_new_tokens,
    }


# =========================================================
# Paged Measurement
# =========================================================

@torch.no_grad()
def measure_paged_multi(
    model,
    tokenizer,
    prompts,
    scheduler,
    pool,
    max_new_tokens=40,
    enable_token_debug: bool = False,
):
    """
    Paged multi-request generation 실행 및 latency/token/block 측정.
    주의: 여기서는 reset_peak_memory_stats()를 호출하지 않는다.
    VRAM peak는 multi_only_main()에서 paged model load 전 reset한 값 기준으로 측정한다.
    """
    model.eval()
    process = psutil.Process(os.getpid())
    runtimes = []

    model_device = get_model_device(model)

    # -------------------------------------------------
    # 0) Tokenize
    # -------------------------------------------------
    encoded_inputs = []
    max_prompt_len = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model_device)
        prompt_len = input_ids.shape[1]

        encoded_inputs.append((input_ids, prompt_len))
        max_prompt_len = max(max_prompt_len, prompt_len)

    shared_total_tokens = max_prompt_len + max_new_tokens

    # -------------------------------------------------
    # 1) Runtime state 준비
    # -------------------------------------------------
    for i, (input_ids, prompt_len) in enumerate(encoded_inputs):
        rid = f"multi_req_{i}"

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

    # -------------------------------------------------
    # 2) 성능 측정 시작
    # -------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    t0 = time.perf_counter()

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

        for layer in model.model.layers:
            layer.self_attn.block_table = batch_block_tables
            layer.self_attn.page_pool = pool

        model.model.block_table = batch_block_tables
        model.model.page_pool = pool
        model.model.active_request_state = None
        model.model.active_request_states = batch_request_states

        for rt in group_rts:
            scheduler.set_seq_len(rt["request_id"], 0)
            rt["request_state"]["seq_len"] = 0

        outputs = model(
            input_ids=batch_input_ids,
            use_cache=True,
            past_key_values=None,
        )

        if enable_token_debug:
            _print_topk_logits(
                outputs.logits,
                tokenizer,
                tag=f"prefill_len_{prompt_len}",
            )

        next_tokens = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        for i, rt in enumerate(group_rts):
            next_token = next_tokens[i:i + 1]

            rt["generated"] = torch.cat(
                [rt["generated"], next_token],
                dim=1,
            )

            rt["request_state"]["seq_len"] = rt["generated"].shape[1]

            if (
                tokenizer.eos_token_id is not None
                and int(next_token.item()) == tokenizer.eos_token_id
            ):
                rt["finished"] = True

    # -------------------------------------------------
    # 4) DECODE
    # -------------------------------------------------
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
            device=model_device,
            dtype=torch.long,
        )

        batch_position_ids = batch_cache_position.unsqueeze(1)

        batch_block_tables = [rt["block_table"] for rt in active_rts]
        batch_request_states = [rt["request_state"] for rt in active_rts]

        for layer in model.model.layers:
            layer.self_attn.block_table = batch_block_tables
            layer.self_attn.page_pool = pool

        model.model.block_table = batch_block_tables
        model.model.page_pool = pool
        model.model.active_request_state = None
        model.model.active_request_states = batch_request_states

        outputs = model(
            input_ids=batch_last_tokens,
            use_cache=True,
            past_key_values=None,
            cache_position=batch_cache_position,
            position_ids=batch_position_ids,
        )

        if enable_token_debug:
            _print_topk_logits(
                outputs.logits,
                tokenizer,
                tag=f"decode_step_{step}",
            )

        next_tokens = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        if enable_token_debug:
            print(f"\n[STEP {step}] generated next tokens")

            for row in range(next_tokens.shape[0]):
                token_id = int(next_tokens[row, 0].item())
                decoded = tokenizer.decode([token_id])
                print(
                    f"row={row} "
                    f"next_token_id={token_id} "
                    f"decoded={repr(decoded)}"
                )

        for i, rt in enumerate(active_rts):
            next_token = next_tokens[i:i + 1]

            rt["generated"] = torch.cat(
                [rt["generated"], next_token],
                dim=1,
            )

            if (
                tokenizer.eos_token_id is not None
                and int(next_token.item()) == tokenizer.eos_token_id
            ):
                rt["finished"] = True

    # -------------------------------------------------
    # 5) 성능 측정 종료
    # -------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    latency = t1 - t0

    prompt_tokens_total = sum(int(rt["prompt_len"]) for rt in runtimes)

    generated_tokens_total = sum(
        int(rt["generated"].shape[1] - rt["prompt_len"])
        for rt in runtimes
    )

    total_tokens = prompt_tokens_total + generated_tokens_total

    throughput = (
        generated_tokens_total / latency
        if latency > 0
        else 0.0
    )

    context_switch = (
        ctx_end.voluntary - ctx_start.voluntary
        + ctx_end.involuntary - ctx_start.involuntary
    )

    final_lengths = [int(rt["generated"].shape[1]) for rt in runtimes]

    used_blocks = sum(
        math.ceil(length / pool.block_size)
        for length in final_lengths
    )

    total_block_capacity = used_blocks * pool.block_size

    block_utilization = (
        total_tokens / total_block_capacity * 100
        if total_block_capacity > 0
        else 0.0
    )

    decoded_texts = []
    decoded_responses = []
    generated_token_ids = []
    response_token_ids = []

    for rt in runtimes:
        full_ids = rt["generated"][0]
        response_ids = full_ids[rt["prompt_len"]:]

        full_text = tokenizer.decode(
            full_ids,
            skip_special_tokens=True,
        )

        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        decoded_texts.append(full_text)
        decoded_responses.append(response_text)
        generated_token_ids.append(full_ids.tolist())
        response_token_ids.append(response_ids.tolist())

    return {
        "latency": latency,
        "throughput": throughput,
        "ram_increase_mb": ram_end - ram_start,

        "peak_vram_mb": 0.0,
        "alloc_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
        "max_reserved_vram_mb": 0.0,
        "vram_per_token_kb": 0.0,

        "context_switch": context_switch,
        "used_blocks": used_blocks,
        "block_utilization": block_utilization,

        "texts": decoded_texts,
        "responses": decoded_responses,
        "generated_token_ids": generated_token_ids,
        "response_token_ids": response_token_ids,

        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
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
    else:
        print("GPU name                  : CPU mode / CUDA unavailable")

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
        return f"{value:.2f}%"

    rows = [
        (
            "Latency (sec)",
            fmt_float(normal_stats.get("latency"), 4),
            fmt_float(paged_stats.get("latency"), 4),
        ),
        (
            "Throughput (tok/s)",
            fmt_float(normal_stats.get("throughput"), 2),
            fmt_float(paged_stats.get("throughput"), 2),
        ),
        (
            "RAM Increase (MB)",
            fmt_float(normal_stats.get("ram_increase_mb"), 1),
            fmt_float(paged_stats.get("ram_increase_mb"), 1),
        ),
        (
            "Peak VRAM (MB)",
            fmt_float(normal_stats.get("peak_vram_mb"), 1),
            fmt_float(paged_stats.get("peak_vram_mb"), 1),
        ),
        (
            "Alloc VRAM (MB)",
            fmt_float(normal_stats.get("alloc_vram_mb"), 1),
            fmt_float(paged_stats.get("alloc_vram_mb"), 1),
        ),
        (
            "Reserved VRAM (MB)",
            fmt_float(normal_stats.get("reserved_vram_mb"), 1),
            fmt_float(paged_stats.get("reserved_vram_mb"), 1),
        ),
        (
            "Max Reserved VRAM (MB)",
            fmt_float(normal_stats.get("max_reserved_vram_mb"), 1),
            fmt_float(paged_stats.get("max_reserved_vram_mb"), 1),
        ),
        (
            "VRAM/token (KB)",
            fmt_float(normal_stats.get("vram_per_token_kb"), 2),
            fmt_float(paged_stats.get("vram_per_token_kb"), 2),
        ),
        (
            "Context Switch",
            fmt_int(normal_stats.get("context_switch")),
            fmt_int(paged_stats.get("context_switch")),
        ),
        (
            "Used Blocks",
            fmt_int(normal_stats.get("used_blocks")),
            fmt_int(paged_stats.get("used_blocks")),
        ),
        (
            "Block Utilization",
            fmt_percent(normal_stats.get("block_utilization")),
            fmt_percent(paged_stats.get("block_utilization")),
        ),
        (
            "Prompt Tokens",
            fmt_int(normal_stats.get("prompt_tokens_total")),
            fmt_int(paged_stats.get("prompt_tokens_total")),
        ),
        (
            "Generated Tokens",
            fmt_int(normal_stats.get("generated_tokens_total")),
            fmt_int(paged_stats.get("generated_tokens_total")),
        ),
        (
            "Total Tokens",
            fmt_int(normal_stats.get("total_tokens")),
            fmt_int(paged_stats.get("total_tokens")),
        ),
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


# =========================================================
# Main
# =========================================================

def multi_only_main():
    print("🔥 실제 사용 파일 = /LimYeonGyeong/project/paged_llama/llama/modeling/modeling_llama.py")
    print("🔥 LlamaModel.forward 위치 = /LimYeonGyeong/project/paged_llama/llama/utils/utils.py")
    print(">>> Multi Request 비교 실행")
    print(f">>> torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f">>> device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    num_requests = 2
    max_new_tokens = 40
    block_size = 16
    enable_token_debug = False

    prompts = build_mixed_prompts(
        n_requests=num_requests,
        seed=None,
    )

    print("\n================ PROMPTS ================\n")

    for i, prompt in enumerate(prompts):
        print(f"--- Prompt {i + 1} ---")
        print(prompt)
        print()

    # -------------------------------------------------
    # 1) Normal baseline
    # VRAM 측정 기준점: normal 모델 로딩 전
    # -------------------------------------------------
    gc.collect()
    cuda_reset_for_full_measurement()

    print(">>> Normal baseline 실행 중...")

    model_normal = load_baseline_model()

    normal_stats = measure_baseline_multi(
        model=model_normal,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
    )

    normal_stats = attach_vram_stats(
        normal_stats,
        normal_stats["generated_tokens_total"],
    )

    del model_normal
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # -------------------------------------------------
    # 2) Paged model
    # VRAM 측정 기준점: paged 모델 로딩 + PagePool 생성 전
    # -------------------------------------------------
    gc.collect()
    cuda_reset_for_full_measurement()

    print(">>> Paged multi 로딩 중...")

    model_paged = load_paged_model()
    config = model_paged.config

    max_prompt_len = 0

    for prompt in prompts:
        prompt_len = tokenizer(
            prompt,
            return_tensors="pt",
        )["input_ids"].shape[1]

        max_prompt_len = max(max_prompt_len, prompt_len)

    shared_total_tokens = max_prompt_len + max_new_tokens

    blocks_per_request = (
        shared_total_tokens + block_size - 1
    ) // block_size

    all_needed_blocks = blocks_per_request * len(prompts)
    safety_margin = 4
    num_blocks = all_needed_blocks + safety_margin

    pool = PagePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=model_paged.dtype,
    )

    scheduler = SimpleScheduler(
        page_pool=pool,
        block_size=block_size,
    )

    dummy_total_tokens = (
        tokenizer(
            prompts[0],
            return_tensors="pt",
        )["input_ids"].shape[1]
        + max_new_tokens
    )

    dummy_block_table = scheduler.allocate_for_request(
        "dummy_req",
        dummy_total_tokens,
    )

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=dummy_block_table,
        debug=False,
        debug_verbose=False,
    )

    scheduler.release_request("dummy_req")

    paged_stats = measure_paged_multi(
        model=model_paged,
        tokenizer=tokenizer,
        prompts=prompts,
        scheduler=scheduler,
        pool=pool,
        max_new_tokens=max_new_tokens,
        enable_token_debug=enable_token_debug,
    )

    paged_stats = attach_vram_stats(
        paged_stats,
        paged_stats["generated_tokens_total"],
    )

    # -------------------------------------------------
    # 3) 답변 + 성능 표 출력
    # -------------------------------------------------
    print_multi_request_result_table(
        normal_stats=normal_stats,
        paged_stats=paged_stats,
    )

    # cleanup
    del model_paged
    del pool
    del scheduler
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    multi_only_main()
