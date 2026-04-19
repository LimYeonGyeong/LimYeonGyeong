import os
import gc
import sys
import time
import psutil
import torch
import builtins
from contextlib import contextmanager

sys.path.append("/LimYeonGyeong/project")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from paged_llama.llama.modeling.modeling_llama import (
    PagedLlamaAttention,
    LlamaForCausalLM as PagedLlamaForCausalLM,
)
from paged_llama.llama.config.configuration_llama import LlamaConfig
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

# 기존 main.py에서 검증된 helper 재사용
from main import (
    measure_baseline_multi,
    patch_model_with_paged_attention,
    print_stats_table,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


# 실행 중 너무 많이 찍히는 디버그 출력만 걸러낸다.
# - measure_paged_multi() 내부의 step-by-step 로그
# - modeling_llama.py 내부의 잔여 디버그 로그
# 최종 결과, 로딩 메시지, 성능 표는 그대로 둔다.
DEBUG_PREFIXES_TO_SUPPRESS = (
    "[MAIN-PREFILL]",
    "[MAIN-DECODE STEP",
    "[CACHE-ID]",
    "[CACHE-TYPE]",
    "[CACHE-SEQ]",
    "[CACHE-POS]",
    "[CACHE-UPDATE]",
    "[REQ-STATE]",
    "[REQ-STATE-ID]",
    "[POSITION-IDS]",
    "[CHECK]",
    "[MODEL-ENTRY]",
    "[MODEL-EXIT]",
    "[LM-HEAD-ENTRY]",
    "[LM-HEAD-EXIT]",
    "[PagedCacheShim",
    "[DBG]",
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

        text = " ".join(str(a) for a in args)
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
        num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
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


def load_paged_model():
    print(">>> HF config / state_dict 로딩 중...")

    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
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

    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    return model


def make_prompt(topic: str, detail_level: str) -> str:
    if detail_level == "short":
        return (
            "### Instruction:\n"
            f"Explain {topic} in one short sentence.\n"
            "### Response:\n"
        )

    if detail_level == "medium":
        return (
            "### Instruction:\n"
            f"Explain {topic} clearly for a beginner.\n"
            "Use 3 to 4 simple sentences and include one example.\n"
            "### Response:\n"
        )

    if detail_level == "long":
        return (
            "### Instruction:\n"
            f"Explain {topic} in detail for a student who is learning LLM systems.\n"
            "Your answer should include:\n"
            "1. a simple definition,\n"
            "2. why it matters,\n"
            "3. one technical detail,\n"
            "4. one practical example,\n"
            "5. one limitation.\n"
            "Write around 8 to 10 sentences.\n"
            "### Response:\n"
        )

    raise ValueError(f"Unknown detail_level: {detail_level}")


def build_mixed_prompts(n_requests: int = 20):
    topics = [
        "LLM",
        "transformer attention",
        "KV cache",
        "paged attention",
        "multi-head attention",
        "inference",
        "training",
        "prefill and decode",
        "block table",
        "page pool",
        "GPU memory fragmentation",
        "sequence length",
        "context window",
        "RoPE embeddings",
        "beam search",
        "greedy decoding",
        "tokenization",
        "causal mask",
        "batch inference",
        "dynamic cache",
        "static cache",
        "request scheduling",
        "latency",
        "throughput",
        "memory efficiency",
    ]

    detail_pattern = (
        ["short"] * (n_requests // 3)
        + ["medium"] * (n_requests // 3)
        + ["long"] * (n_requests - 2 * (n_requests // 3))
    )

    prompts = []
    for i in range(n_requests):
        topic = topics[i % len(topics)]
        detail = detail_pattern[i]
        prompts.append(make_prompt(topic, detail))

    return prompts


def _get_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _get_ctx_switches() -> int:
    cs = psutil.Process(os.getpid()).num_ctx_switches()
    return int(cs.voluntary + cs.involuntary)


def _build_request_entries(tokenizer, prompts, scheduler, max_new_tokens):
    entries = []
    for i, prompt in enumerate(prompts):
        req_id = f"req_{i}"
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        prompt_len = input_ids.shape[1]
        total_tokens = prompt_len + max_new_tokens
        block_table = scheduler.allocate_for_request(req_id, total_tokens)
        request_state = scheduler.get_request_state(req_id)

        entries.append(
            {
                "req_id": req_id,
                "prompt": prompt,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": prompt_len,
                "block_table": block_table,
                "request_state": request_state,
                "past_key_values": None,
                "generated_token_ids": [],
                "generated_tokens": 0,
                "finished": False,
            }
        )
    return entries


# -------------------------
# Step 1 구조 수정:
# - 기존: request-major loop
# - 변경: prefill-all -> decode step-major loop
# 아직 attention 내부는 bsz=1 전제라 진짜 병렬 GPU batch는 아님
# 하지만 실행 구조를 다음 단계(batch-aware attention)로 옮기기 쉽게 만든다.
# -------------------------
@torch.no_grad()
def measure_paged_multi(model, tokenizer, prompts, scheduler, pool, max_new_tokens=20):
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    ram_before = _get_ram_mb()
    ctx_before = _get_ctx_switches()
    start_time = time.perf_counter()

    entries = _build_request_entries(
        tokenizer=tokenizer,
        prompts=prompts,
        scheduler=scheduler,
        max_new_tokens=max_new_tokens,
    )

    # -------------------------
    # Prefill-all
    # -------------------------
    for entry in entries:
        model.active_request_state = entry["request_state"]
        model.active_block_table = entry["block_table"]

        outputs = model(
            input_ids=entry["input_ids"],
            attention_mask=entry["attention_mask"],
            use_cache=True,
            past_key_values=None,
            return_dict=True,
        )
        entry["past_key_values"] = outputs.past_key_values

    # -------------------------
    # Decode step-major
    # -------------------------
    active_entries = [e for e in entries if not e["finished"]]

    for _step in range(max_new_tokens):
        if not active_entries:
            break

        next_active = []
        for entry in active_entries:
            model.active_request_state = entry["request_state"]
            model.active_block_table = entry["block_table"]

            if entry["generated_tokens"] == 0:
                step_input_ids = entry["input_ids"][:, -1:]
            else:
                step_input_ids = torch.tensor(
                    [[entry["generated_token_ids"][-1]]],
                    device=device,
                    dtype=entry["input_ids"].dtype,
                )

            outputs = model(
                input_ids=step_input_ids,
                use_cache=True,
                past_key_values=entry["past_key_values"],
                return_dict=True,
            )
            entry["past_key_values"] = outputs.past_key_values

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            next_token = int(next_token_id.item())
            entry["generated_token_ids"].append(next_token)
            entry["generated_tokens"] += 1

            if next_token == tokenizer.eos_token_id:
                entry["finished"] = True
            else:
                next_active.append(entry)

        active_entries = next_active

    elapsed = time.perf_counter() - start_time
    ram_after = _get_ram_mb()
    ctx_after = _get_ctx_switches()

    texts = []
    total_generated_tokens = 0
    for entry in entries:
        total_generated_tokens += len(entry["generated_token_ids"])
        full_ids = torch.cat(
            [
                entry["input_ids"][0].detach().cpu(),
                torch.tensor(entry["generated_token_ids"], dtype=entry["input_ids"].dtype),
            ],
            dim=0,
        )
        texts.append(tokenizer.decode(full_ids, skip_special_tokens=True))

    used_blocks = pool.num_blocks - len(pool.free_blocks)
    block_util = (used_blocks / pool.num_blocks * 100.0) if pool.num_blocks > 0 else 0.0

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        alloc_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_vram = torch.cuda.memory_reserved() / (1024 * 1024)
        max_reserved_vram = torch.cuda.max_memory_reserved() / (1024 * 1024)
    else:
        peak_vram = alloc_vram = reserved_vram = max_reserved_vram = 0.0

    vram_per_token_kb = 0.0
    if total_generated_tokens > 0:
        vram_per_token_kb = (peak_vram * 1024.0) / total_generated_tokens

    return {
        "texts": texts,
        "latency": elapsed,
        "throughput": (total_generated_tokens / elapsed) if elapsed > 0 else 0.0,
        "ram_increase_mb": max(0.0, ram_after - ram_before),
        "peak_vram_mb": peak_vram,
        "alloc_vram_mb": alloc_vram,
        "reserved_vram_mb": reserved_vram,
        "max_reserved_vram_mb": max_reserved_vram,
        "vram_per_token_kb": vram_per_token_kb,
        "context_switch": max(0, ctx_after - ctx_before),
        "used_blocks": used_blocks,
        "block_utilization": block_util,
    }


def multi_only_main():
    print(">>> Multi Request 비교 실행")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # paged가 유리한 조건:
    # - 요청 수 많음
    # - 길이 제각각
    num_requests = 20
    prompts = build_mixed_prompts(n_requests=num_requests)

    max_new_tokens = 20
    block_size = 16

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Baseline multi
    # -------------------------
    print(">>> Baseline multi 로딩 중...")
    model_base_multi = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    stats_base_multi = measure_baseline_multi(
        model_base_multi, tokenizer, prompts, max_new_tokens=max_new_tokens
    )

    del model_base_multi
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Paged multi
    # -------------------------
    print(">>> Paged multi 로딩 중...")
    model_paged = load_paged_model()

    config = model_paged.config

    # 전체 multi prompt 기준으로 pool 크기 계산
    all_needed_blocks = 0
    for p in prompts:
        prompt_len = tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
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

    # patch 함수가 초기 block_table을 요구하므로 dummy 1개만 먼저 생성
    dummy_prompt_len = tokenizer(prompts[0], return_tensors="pt")["input_ids"].shape[1]
    dummy_total_tokens = dummy_prompt_len + max_new_tokens
    dummy_block_table = scheduler.allocate_for_request("dummy_req", dummy_total_tokens)

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=dummy_block_table,
        debug=False,
        debug_verbose=False,
    )

    # dummy는 바로 해제
    scheduler.release_request("dummy_req")

    pool.k_cache.zero_()
    pool.v_cache.zero_()

    with suppress_debug_prints(enabled=True):
        stats_paged_multi = measure_paged_multi(
            model_paged, tokenizer, prompts, scheduler, pool, max_new_tokens=max_new_tokens
        )

    print("\n[OUTPUT] Multi Request Generation Results (first 5 only)")
    for i, text in enumerate(stats_paged_multi["texts"][:5]):
        print(f"\n--- Request {i+1} ---")
        print(text[:500])

    print_stats_table("Multi Request Result", stats_base_multi, stats_paged_multi, include_blocks=True)

    del model_paged
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    multi_only_main()
