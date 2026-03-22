import torch
import time
import os
import psutil
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TOKEN = os.getenv("HF_TOKEN")
if TOKEN:
    login(token=TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"


def patch_model_with_paged_attention(model, page_pool, block_table):
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        new_attn = PagedLlamaAttention(
            config=model.config,
            layer_idx=layer_idx,
            page_pool=page_pool,
        )

        new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
        new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

        new_attn.block_table = block_table

        ref_param = next(old_attn.parameters())
        new_attn = new_attn.to(device=ref_param.device, dtype=ref_param.dtype)

        layer.self_attn = new_attn

    return model


def measure_performance(model, tokenizer, prompt_text):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
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
        "ctx_switches": (ctx_end.voluntary - ctx_start.voluntary) + (ctx_end.involuntary - ctx_start.involuntary),
    }

    if device == "cuda":
        stats["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats["peak_vram_mb"] = 0.0

    return stats


print(">>> 측정 시작 (Baseline 로딩 중...)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = """### Instruction:\n what is paged attention?.\n### Response:\n"""

model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

stats_base = measure_performance(model_base, tokenizer, prompt)

del model_base
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

print(">>> PagedAttention 로딩 및 패치 중...")
from paged_llama.llama.modeling.modeling_llama import PagedLlamaAttention
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.block_table import BlockTable

model_paged = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

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
    block_table=shared_block_table
)

pool.k_cache.zero_()
pool.v_cache.zero_()

stats_paged = measure_performance(model_paged, tokenizer, prompt)

print("\n[DEBUG] PagePool / BlockTable 확인")
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

print(f"{'Metric':<25} | {'normal':<15} | {'Paged':<15} |")
print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")
print(f"{'Total RAM (MB)':<25} | {stats_base['ram_mb']:<15.1f} | {stats_paged['ram_mb']:<15.1f} |")
print(f"{'Peak VRAM (MB)':<25} | {stats_base['peak_vram_mb']:<15.1f} | {stats_paged['peak_vram_mb']:<15.1f} |")
print(f"{'Context Switch':<25} | {stats_base['ctx_switches']:<15} | {stats_paged['ctx_switches']:<15} |")