import gc
import os
import time
import psutil
import torch

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from paged_llama.llama.modeling.modeling_llama import PagedLlamaAttention
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.block_table import BlockTable


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Hugging Face 로그인
# -----------------------------
TOKEN = os.getenv("HF_TOKEN")
if TOKEN:
    login(token=TOKEN)


# -----------------------------
# Baseline 성능 측정
# -----------------------------
@torch.no_grad()
def measure_performance(model, tokenizer, prompt_text, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    if device == "cuda":
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
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    stats = {
        "text": generated_text,
        "latency": t1 - t0,
        "throughput": generated_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0,
        "ram_mb": ram_end,
        "ctx_switches": (ctx_end.voluntary - ctx_start.voluntary)
        + (ctx_end.involuntary - ctx_start.involuntary),
    }

    if device == "cuda":
        stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats["peak_vram_mb"] = 0.0

    return stats


# -----------------------------
# PagedAttention 정확성 테스트
# HF cache OFF + PagePool only
# -----------------------------
@torch.no_grad()
def test_paged_generation_step_by_step(model, tokenizer, prompt_text, max_new_tokens=20):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generated = inputs["input_ids"].clone()

    # 1) prefill
    seq_len = generated.shape[1]
    attention_mask = torch.ones_like(generated, device=generated.device)
    position_ids = torch.arange(seq_len, device=generated.device).unsqueeze(0)

    outputs = model(
        input_ids=generated,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        past_key_values=None,
    )

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    print(
        f"[STEP 0] token_id = {next_token.item()} | "
        f"token = {repr(tokenizer.decode(next_token[0], skip_special_tokens=False))}"
    )

    # 2) decode
    for step in range(1, max_new_tokens):
        cur_len = generated.shape[1]

        last_token = generated[:, -1:]
        attention_mask = torch.ones((1, cur_len), dtype=torch.long, device=generated.device)
        position_ids = torch.tensor([[cur_len - 1]], device=generated.device)

        outputs = model(
            input_ids=last_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            past_key_values=None,
        )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        token_id = next_token.item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"[STEP {step}] token_id = {token_id} | token = {repr(token_text)}")

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    final_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("[TEST OUTPUT] HF cache OFF + PagePool only")
    print(final_text)
    print("=" * 60 + "\n")

    return final_text


# -----------------------------
# PagedAttention 성능 측정
# HF cache OFF + PagePool only
# -----------------------------
@torch.no_grad()
def measure_paged_only(model, tokenizer, prompt_text, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    generated = inputs["input_ids"].clone()

    t0 = time.time()

    # 1) prefill
    seq_len = generated.shape[1]
    attention_mask = torch.ones_like(generated, device=generated.device)
    position_ids = torch.arange(seq_len, device=generated.device).unsqueeze(0)

    outputs = model(
        input_ids=generated,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        past_key_values=None,
    )

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    # 2) decode
    for _ in range(max_new_tokens - 1):
        cur_len = generated.shape[1]

        last_token = generated[:, -1:]
        attention_mask = torch.ones((1, cur_len), dtype=torch.long, device=generated.device)
        position_ids = torch.tensor([[cur_len - 1]], device=generated.device)

        outputs = model(
            input_ids=last_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            past_key_values=None,
        )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    stats = {
        "text": generated_text,
        "latency": t1 - t0,
        "throughput": generated_tokens / (t1 - t0) if (t1 - t0) > 0 else 0.0,
        "ram_mb": ram_end,
        "ctx_switches": (ctx_end.voluntary - ctx_start.voluntary)
        + (ctx_end.involuntary - ctx_start.involuntary),
    }

    if device == "cuda":
        stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats["peak_vram_mb"] = 0.0

    return stats


# -----------------------------
# Attention patch
# -----------------------------
def patch_model_with_paged_attention(model, page_pool, block_table):
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        new_attn = PagedLlamaAttention(
            config=model.config,
            layer_idx=layer_idx,
            page_pool=page_pool,
        )

        # 기존 weight 복사
        new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
        new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

        # block table 연결
        new_attn.block_table = block_table

        # 디버그 출력 비활성화
        new_attn.debug = False

        ref_param = next(old_attn.parameters())
        new_attn = new_attn.to(device=ref_param.device, dtype=ref_param.dtype)

        layer.self_attn = new_attn

    return model


# -----------------------------
# 메인 실행
# -----------------------------
def main():
    print(">>> 측정 시작 (Baseline 로딩 중...)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = (
        '### Instruction:\n'
        'In this context, "LLM" means "Large Language Model". '
        'Explain it in one simple sentence.\n'
        '### Response:\n'
    )

    # -------------------------
    # Baseline
    # -------------------------
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    stats_base = measure_performance(model_base, tokenizer, prompt, max_new_tokens=20)

    del model_base
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Paged
    # -------------------------
    print(">>> PagedAttention 로딩 및 패치 중...")

    model_paged = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # HF 기본 cache 끄기
    model_paged.config.use_cache = False
    if hasattr(model_paged, "generation_config"):
        model_paged.generation_config.use_cache = False

    config = model_paged.config

    pool = PagePool(
        num_blocks=2500,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=16,
        head_dim=config.hidden_size // config.num_attention_heads,
        device=device,
        dtype=model_paged.dtype,
    )

    shared_block_table = BlockTable(block_size=16)
    for _ in range(100):
        shared_block_table.add_block(pool.allocate())

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=shared_block_table,
    )

    # PagePool 초기화
    pool.k_cache.zero_()
    pool.v_cache.zero_()

    # -------------------------
    # 1단계 정확성 테스트
    # -------------------------
    print(">>> HF cache OFF + PagePool only 테스트 중...")

    test_text = test_paged_generation_step_by_step(
        model=model_paged,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=20,
    )

    # -------------------------
    # 2단계 성능 측정
    # -------------------------
    stats_paged = measure_paged_only(
        model=model_paged,
        tokenizer=tokenizer,
        prompt_text=prompt,
        max_new_tokens=20,
    )

    # -------------------------
    # 디버그 정보
    # -------------------------
    print("[DEBUG] PagePool / BlockTable 확인")
    print("pool.k_cache.shape =", pool.k_cache.shape)
    print("pool.v_cache.shape =", pool.v_cache.shape)

    if hasattr(shared_block_table, "to_tensor"):
        bt = shared_block_table.to_tensor(device=device)
        print("block_table.shape =", bt.shape)
        print("block_table[0, :10] =", bt[0, :10])
    else:
        print("shared_block_table has no to_tensor()")

    print("\n" + "=" * 60)
    print("[OUTPUT] PagedAttention Generation Result")
    print(stats_paged["text"])
    print("=" * 60 + "\n")

    # -------------------------
    # 결과 출력
    # -------------------------
    print(f"{'Metric':<25} | {'normal':<15} | {'Paged':<15} |")
    print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
    print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")
    print(f"{'Total RAM (MB)':<25} | {stats_base['ram_mb']:<15.1f} | {stats_paged['ram_mb']:<15.1f} |")
    print(f"{'Peak VRAM (MB)':<25} | {stats_base['peak_vram_mb']:<15.1f} | {stats_paged['peak_vram_mb']:<15.1f} |")
    print(f"{'Context Switch':<25} | {stats_base['ctx_switches']:<15} | {stats_paged['ctx_switches']:<15} |")


if __name__ == "__main__":
    main()