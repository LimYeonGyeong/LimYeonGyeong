import os
import gc
import sys
import torch

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
    measure_paged_multi,
    patch_model_with_paged_attention,
    print_stats_table,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def multi_only_main():
    print(">>> Multi Request 비교 실행")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 메모리 안정성을 위해 우선 4개 질문만 사용
    prompts = [
        "### Instruction:\nExplain what LLM is in one sentence.\n### Response:",
        "### Instruction:\nExplain Medusa LLM in one simple sentence.\n### Response:",
        "### Instruction:\nDescribe how attention works in transformers in one sentence.\n### Response:",
        "### Instruction:\nExplain KV cache in simple terms in one sentence.\n### Response:",
    ]

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

    print("\n[OUTPUT] Multi Request Generation Results")
    for i, text in enumerate(stats_paged_multi["texts"]):
        print(f"\n--- Request {i+1} ---")
        print(text)

    print_stats_table("Multi Request Result", stats_base_multi, stats_paged_multi, include_blocks=True)

    del model_paged
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    multi_only_main()
