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

# main.py에 있는 검증된 patch helper 사용
from main import patch_model_with_paged_attention


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Utils
# =========================================================

def model_device(model):
    return next(model.parameters()).device


def reset_cuda_stats_for_measurement():
    """
    main.py의 측정 방식과 동일하게 generation 직전에 CUDA 통계를 초기화한다.
    주의: 이 함수는 모델이 GPU에 올라간 뒤 호출해도 현재 allocated 값은 사라지지 않고,
    peak 기준만 현재 상태로 리셋된다.
    """
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def read_cuda_stats_mb():
    """PyTorch CUDA allocator 기준 VRAM 통계를 MB 단위로 읽는다."""
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

def make_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        f"{question}\n"
        "Answer in clear plain English.\n"
        "Do not use markdown, bullet points, code blocks, or numbered lists.\n"
        "### Response:\n"
    )


def build_mixed_prompts(n_requests: int = 2, seed: int | None = None):
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
    main.py의 measure_baseline_multi 방식으로 측정하되,
    답변 출력용 responses/texts를 추가한다.
    """
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
        "used_blocks": 0,
        "block_utilization": 0.0,
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
    Custom Paged Attention multi-request generation + main.py 방식 성능 측정.

    핵심:
    - VRAM 통계는 함수 시작 직전에 reset하고, generation 직후 바로 읽는다.
    - return 이후에 del / empty_cache로 값을 덮어쓰지 않는다.
    - 답변 출력용 responses/texts도 함께 반환한다.
    """
    model.eval()
    process = psutil.Process(os.getpid())
    dev = model_device(model)

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

    # -------------------------------------------------
    # 2) 성능 측정 시작: main.py와 같은 위치
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
            device=dev,
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
    # 5) 성능 측정 종료: generation 직후 바로 읽기
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
    # 6) block 사용량 계산
    # main.py와 동일하게 used_blocks / pool.num_blocks 비율 사용
    # -------------------------------------------------
    used_blocks = 0
    for rt in runtimes:
        bt = rt["block_table"]
        if hasattr(bt, "to_tensor"):
            used_blocks += bt.to_tensor(device="cpu").shape[1]
        elif hasattr(bt, "block_table"):
            used_blocks += len(bt.block_table[0])
        elif hasattr(bt, "physical_blocks"):
            used_blocks += len(bt.physical_blocks)
        else:
            used_blocks += math.ceil(rt["generated"].shape[1] / pool.block_size)

    block_utilization = (
        used_blocks / pool.num_blocks
        if pool.num_blocks > 0
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
        generated_token_ids.append(full_ids.detach().cpu().tolist())
        response_token_ids.append(response_ids.detach().cpu().tolist())

    # request release는 통계 측정 이후에 수행
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
        "used_blocks": used_blocks,
        "block_utilization": block_utilization,
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
        ("Block Utilization", "-", fmt_percent(paged_stats.get("block_utilization"))),
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


# =========================================================
# Main
# =========================================================

def multi_only_main():
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

    prompts = build_mixed_prompts(n_requests=num_requests, seed=None)

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
        max_new_tokens=max_new_tokens,
    )

    # 통계는 이미 저장되었으므로 이후 cleanup 가능
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

    shared_total_tokens = max_prompt_len + max_new_tokens
    blocks_per_request = (shared_total_tokens + block_size - 1) // block_size
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

    scheduler = SimpleScheduler(page_pool=pool, block_size=block_size)

    dummy_total_tokens = (
        tokenizer(prompts[0], return_tensors="pt")["input_ids"].shape[1]
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
