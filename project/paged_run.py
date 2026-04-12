import os
import gc
import torch

from transformers import AutoTokenizer

from paged_llama.llama.modeling.modeling_llama import (
    LlamaForCausalLM as PagedLlamaForCausalLM,
)
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.block_table import BlockTable
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

# 너 기존 코드에서 쓰던 함수
from main import patch_model_with_paged_attention, measure_paged_only

# 메모리 fragmentation 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


def paged_main():
    print(">>> Paged ONLY 실행")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt = (
        '### Instruction:\n'
        'Explain the difference between training and inference in LLMs.\n'
        'Explain it in one simple sentence.\n'
        '### Response:\n'
    )

    # 메모리 초기화
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(">>> 모델 로딩 중...")

    model = PagedLlamaForCausalLM.from_pretrained(MODEL_ID)

    if device == "cuda":
        model = model.half()

    model = model.to(device)

    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    print(">>> PagePool 생성")

    max_new_tokens = 3
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
        debug=False,
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