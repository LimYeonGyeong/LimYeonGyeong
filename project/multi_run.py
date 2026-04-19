import os
import gc
import sys
import time
import contextlib
import io
from typing import Dict, List

import torch
import psutil

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


class _StdoutFilter(io.TextIOBase):
    """과한 디버그 출력만 걸러내고, 나머지 정상 출력은 유지합니다."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self._buffer = ""
        self._drop_prefixes = (
            "[MAIN-PREFILL]",
            "[MAIN-DECODE STEP",
            "[CACHE-ID]",
            "[CACHE-TYPE]",
            "[CACHE-SEQ]",
            "[REQ-STATE]",
            "[REQ-STATE-ID]",
            "[CACHE-POS]",
            "[POSITION-IDS]",
            "[CHECK]",
            "[MODEL-ENTRY]",
            "[MODEL-EXIT]",
            "[LM-HEAD-ENTRY]",
            "[LM-HEAD-EXIT]",
            "[CACHE-UPDATE]",
            "[PagedCacheShim",
            "[VERIFY-WRITE]",
            "[WRITE][Layer",
            "[WRITE] k_diff=",
            "[READ][Layer",
            "[READ] check_pos=",
            "[READ] logic_block=",
            "[READ] physical_block=",
            "[READ] offset=",
            "[READ] k_diff=",
            "[READ] v_diff=",
            "[POS][Layer",
            "[POS] ",
            "[ATTN][Layer",
            "[DBG]",
            "[DBG][Layer",
        )

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.startswith(self._drop_prefixes):
                self.wrapped.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buffer and not self._buffer.startswith(self._drop_prefixes):
            self.wrapped.write(self._buffer)
        self._buffer = ""
        self.wrapped.flush()


@contextlib.contextmanager

def filtered_stdout():
    old_stdout = sys.stdout
    filt = _StdoutFilter(old_stdout)
    sys.stdout = filt
    try:
        yield
    finally:
        filt.flush()
        sys.stdout = old_stdout


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


def _ctx_switch_count() -> int:
    ctx = psutil.Process().num_ctx_switches()
    return int(ctx.voluntary + ctx.involuntary)


def _set_request_runtime_state(model_paged, block_table, request_state):
    # model 레벨
    model_paged.model.block_table = block_table
    model_paged.model.active_request_state = request_state

    # layer attention 레벨
    for layer in model_paged.model.layers:
        layer.self_attn.block_table = block_table
        layer.self_attn.page_pool = model_paged.model.page_pool
        layer.self_attn.debug = False
        layer.self_attn.debug_verbose = False
        layer.self_attn.debug_stop_on_nonfinite = True


def measure_paged_multi(
    model_paged,
    tokenizer,
    prompts: List[str],
    scheduler: SimpleScheduler,
    pool: PagePool,
    max_new_tokens: int = 20,
) -> Dict[str, object]:
    """
    correctness 우선 버전:
    - 요청별 순차 generate 유지
    - 각 요청마다 scheduler state만 정확히 바꿔가며 실행
    - main.print_stats_table과 호환되는 키를 모두 반환
    """
    model_paged.eval()
    model_paged.model.page_pool = pool

    texts: List[str] = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

    ram_before = psutil.Process().memory_info().rss / (1024 * 1024)
    ctx_before = _ctx_switch_count()
    t0 = time.perf_counter()

    with torch.inference_mode(), filtered_stdout():
        for req_idx, prompt in enumerate(prompts):
            request_id = f"req_{req_idx}"
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            prompt_len = input_ids.shape[1]
            total_tokens = prompt_len + max_new_tokens

            block_table = scheduler.allocate_for_request(request_id, total_tokens)
            request_state = scheduler.get_request_state(request_id)
            request_state["seq_len"] = 0

            _set_request_runtime_state(model_paged, block_table, request_state)

            try:
                outputs = model_paged.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                texts.append(text)
            finally:
                scheduler.release_request(request_id)

    latency = time.perf_counter() - t0
    ctx_after = _ctx_switch_count()
    ram_after = psutil.Process().memory_info().rss / (1024 * 1024)

    total_generated_tokens = len(prompts) * max_new_tokens
    throughput = total_generated_tokens / max(latency, 1e-9)

    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        alloc_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved_vram = torch.cuda.memory_reserved() / (1024 * 1024)
        max_reserved_vram = torch.cuda.max_memory_reserved() / (1024 * 1024)
    else:
        peak_vram = alloc_vram = reserved_vram = max_reserved_vram = 0.0

    used_blocks = int(pool.num_blocks - len(pool.free_blocks))
    block_utilization = (used_blocks / max(pool.num_blocks, 1)) * 100.0
    vram_per_token_kb = (peak_vram * 1024.0) / max(total_generated_tokens, 1)

    stats = {
        # main.print_stats_table 호환 키
        "latency": float(latency),
        "throughput": float(throughput),
        "ram_mb": float(ram_after - ram_before),
        "peak_vram": float(peak_vram),
        "alloc_vram": float(alloc_vram),
        "reserved_vram": float(reserved_vram),
        "max_reserved_vram": float(max_reserved_vram),
        "vram_per_token_kb": float(vram_per_token_kb),
        "ctx_switches": int(ctx_after - ctx_before),
        "used_blocks": int(used_blocks),
        "block_utilization": float(block_utilization),
        # 추가 정보
        "texts": texts,
        "total_generated_tokens": int(total_generated_tokens),
        "num_requests": int(len(prompts)),
        # 보조 호환 키
        "context_switch": int(ctx_after - ctx_before),
        "ram_increase_mb": float(ram_after - ram_before),
        "peak_vram_mb": float(peak_vram),
        "alloc_vram_mb": float(alloc_vram),
        "reserved_vram_mb": float(reserved_vram),
        "max_reserved_vram_mb": float(max_reserved_vram),
    }
    return stats


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
