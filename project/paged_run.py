import os
import gc
import sys
import torch

sys.path.append("/LimYeonGyeong/project")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from paged_llama.llama.modeling.modeling_llama import (
    LlamaForCausalLM as PagedLlamaForCausalLM,
)
from paged_llama.llama.config.configuration_llama import LlamaConfig
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

from main import patch_model_with_paged_attention, measure_paged_only

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


def paged_main():
    print(">>> Paged ONLY 실행")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt = (
        "### Instruction:\n"
        "Explain the difference between training and inference in LLMs.\n"
        "Explain it in one simple sentence.\n"
        "### Response:\n"
    )

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(">>> HF config / state_dict 로딩 중...")

    # HF의 실제 TinyLlama config 가져오기
    hf_config = AutoConfig.from_pretrained(MODEL_ID)

    # HF 원본 모델은 CPU에서만 로드해서 state_dict 추출
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

    print(">>> PagePool 생성")

    max_new_tokens = 10
    block_size = 16
    request_id = "req_1"

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    total_tokens = prompt_len + max_new_tokens
    num_blocks = ((total_tokens + block_size - 1) // block_size) + 2

    config = model.config

    pool = PagePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=model.dtype,
    )

    scheduler = SimpleScheduler(page_pool=pool, block_size=block_size)
    block_table = scheduler.allocate_for_request(request_id, total_tokens)

    print(">>> PagedAttention 패치")

    model = patch_model_with_paged_attention(
        model=model,
        page_pool=pool,
        block_table=block_table,
        debug=True,
        debug_verbose=False,
    )

    pool.k_cache.zero_()
    pool.v_cache.zero_()
    scheduler.set_seq_len(request_id, 0)

    print(">>> 생성 시작")

    stats = measure_paged_only(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        block_table=block_table,
        pool=pool,
        scheduler=scheduler,
        request_id=request_id,
        max_new_tokens=max_new_tokens,
    )

    print("\n" + "=" * 60)
    print("[OUTPUT]")
    print(stats["text"])
    print("=" * 60)

    print("\n[METRICS]")
    for k, v in stats.items():
        if k != "text":
            print(f"{k}: {v}")

    scheduler.release_request(request_id)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    paged_main()