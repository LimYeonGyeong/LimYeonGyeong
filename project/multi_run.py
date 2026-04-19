import os
import gc
import sys
import torch
import builtins
import psutil
import time
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

def build_decode_positions(model):
    request_state = getattr(model.model, "active_request_state", None)
    if request_state is None:
        raise RuntimeError("[POSITION ERROR] active_request_state is None")

    current_pos = int(request_state["seq_len"])
    cache_position = torch.tensor([current_pos], device=model.device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)

    return cache_position, position_ids

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



def _bind_request_runtime(model, block_table, request_state, pool):
    for layer in model.model.layers:
        layer.self_attn.block_table = block_table
        layer.self_attn.page_pool = pool
    model.model.block_table = block_table
    model.model.page_pool = pool
    model.model.active_request_state = request_state


@torch.no_grad()
def measure_paged_multi(model, tokenizer, prompts, scheduler, pool, max_new_tokens=20):
    process = psutil.Process(os.getpid())

    request_ids = []
    block_tables = []
    runtimes = []

    # ---------------------------------------------
    # 0) 토크나이징 먼저 하고, batch 내 최대 prompt 길이 계산
    # ---------------------------------------------
    encoded_inputs = []
    max_prompt_len = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]
        encoded_inputs.append((input_ids, prompt_len))
        max_prompt_len = max(max_prompt_len, prompt_len)

    # batch decode에서는 모든 요청이 step을 함께 가므로
    # block table도 batch 내 최대 길이에 맞춰 동일하게 잡아준다.
    shared_total_tokens = max_prompt_len + max_new_tokens

    # ---------------------------------------------
    # 1) request runtime/state 준비
    # ---------------------------------------------
    for i, (input_ids, prompt_len) in enumerate(encoded_inputs):
        rid = f"multi_req_{i}"
        bt = scheduler.allocate_for_request(rid, shared_total_tokens)
        scheduler.set_seq_len(rid, 0)
        request_state = scheduler.get_request_state(rid)

        request_ids.append(rid)
        block_tables.append(bt)
        runtimes.append({
            "request_id": rid,
            "block_table": bt,
            "request_state": request_state,
            "generated": input_ids.clone(),
            "prompt_len": prompt_len,
            "finished": False,
        })

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.perf_counter()

    # -------------------------------------------------
    # 2) PREFILL
    # correctness를 위해 요청별로 그대로 진행
    # -------------------------------------------------
    for rt in runtimes:
        for layer in model.model.layers:
            layer.self_attn.block_table = rt["block_table"]
            layer.self_attn.page_pool = pool

        model.model.block_table = rt["block_table"]
        model.model.page_pool = pool
        model.model.active_request_state = rt["request_state"]

        if hasattr(model.model, "active_request_states"):
            model.model.active_request_states = None

        scheduler.set_seq_len(rt["request_id"], 0)

        outputs = model(
            input_ids=rt["generated"],
            use_cache=True,
            past_key_values=None,
        )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        rt["generated"] = torch.cat([rt["generated"], next_token], dim=1)

        # prefill 직후 seq_len 동기화
        rt["request_state"]["seq_len"] = rt["generated"].shape[1]

        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            rt["finished"] = True

    # -------------------------------------------------
    # 3) DECODE
    # step마다 active requests를 batch로 묶어 한 번에 forward
    # -------------------------------------------------
    for step in range(1, max_new_tokens):
        active_rts = [rt for rt in runtimes if not rt["finished"]]
        if not active_rts:
            break

        batch_last_tokens = torch.cat(
            [rt["generated"][:, -1:] for rt in active_rts],
            dim=0,
        )  # [batch, 1]
        for rt in active_rts:
            rt["request_state"]["seq_len"] = rt["generated"].shape[1] - 1
        
        batch_cache_position = torch.tensor(
            [int(rt["request_state"]["seq_len"]) for rt in active_rts],
            device=model.device,
            dtype=torch.long,
        )  # [batch]

        batch_position_ids = batch_cache_position.unsqueeze(1)  # [batch, 1]

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

        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # [batch, 1]

        for i, rt in enumerate(active_rts):
            next_token = next_tokens[i:i+1]  # [1, 1]
            rt["generated"] = torch.cat([rt["generated"], next_token], dim=1)

            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                rt["finished"] = True

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    generated_tokens = sum(
        rt["generated"].shape[1] - rt["prompt_len"]
        for rt in runtimes
    )

    if device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        current_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    else:
        peak_allocated = 0.0
        current_allocated = 0.0
        current_reserved = 0.0
        max_reserved = 0.0

    used_blocks = 0
    for bt in block_tables:
        if hasattr(bt, "to_tensor"):
            used_blocks += bt.to_tensor(device="cpu").shape[1]
        else:
            used_blocks += len(bt)

    block_utilization = used_blocks / pool.num_blocks if pool.num_blocks > 0 else 0.0

    for rid in request_ids:
        scheduler.release_request(rid)

    decoded_texts = [
        tokenizer.decode(rt["generated"][0], skip_special_tokens=True)
        for rt in runtimes
    ]

    return {
        "latency": t1 - t0,
        "throughput": generated_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0,
        "ram_mb": ram_end - ram_start,
        "ctx_switches": (ctx_end.voluntary - ctx_start.voluntary)
        + (ctx_end.involuntary - ctx_start.involuntary),
        "peak_vram_mb": peak_allocated,
        "alloc_vram_mb": current_allocated,
        "reserved_vram_mb": current_reserved,
        "max_reserved_vram_mb": max_reserved,
        "vram_per_token_kb": (peak_allocated * 1024 / generated_tokens) if generated_tokens > 0 else 0.0,
        "used_blocks": used_blocks,
        "block_utilization": block_utilization,
        "texts": decoded_texts,
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

    num_blocks = int(all_needed_blocks * 2.5) + 32

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

    # measure_paged_multi 내부의 step-by-step print는 성능 측정에 큰 방해가 되므로 막는다.
    with suppress_debug_prints(enabled=True):
        stats_paged_multi = measure_paged_multi(
            model_paged, tokenizer, prompts, scheduler, pool, max_new_tokens=max_new_tokens
        )

    print("[OUTPUT] Multi Request Generation Results (first 5 only)")
    for i, text in enumerate(stats_paged_multi["texts"][:5]):
        print(f"--- Request {i+1} ---")
        print(text[:500])

    print_stats_table("Multi Request Result", stats_base_multi, stats_paged_multi, include_blocks=True)

    del model_paged
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    multi_only_main()
