import torch
import time
import os
import psutil
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# ==========================================
# 1. 설정 (모델 및 토큰)
# ==========================================
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 토큰 설정 (환경변수 또는 직접 입력)
TOKEN = os.getenv("HF_TOKEN")
if TOKEN:
    login(token=TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 성능 측정 엔진
# ==========================================
def measure_performance(model, tokenizer, prompt_text):
    # 워밍업
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 측정 시작 전 상태 기록
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss
    ctx_start = process.num_ctx_switches()
    vram_start = torch.cuda.memory_allocated() if device == "cuda" else 0
    
    t0 = time.time()
    
    # 텍스트 생성 (120토큰)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=120,
            do_sample=False,
            use_cache=True
        )
    
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    
    # 결과 계산
    latency = t1 - t0
    generated_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
    throughput = generated_tokens / latency
    
    ram_usage = (process.memory_info().rss - ram_start) / 1024 / 1024
    vram_usage = (torch.cuda.memory_allocated() - vram_start) / 1024 / 1024 if device == "cuda" else 0
    
    ctx_end = process.num_ctx_switches()
    ctx_sw_total_start = ctx_start.voluntary + ctx_start.involuntary
    ctx_sw_total_end = ctx_end.voluntary + ctx_end.involuntary
    ctx_switch = ctx_sw_total_end - ctx_sw_total_start
    
    return {
        "text": tokenizer.decode(outputs[0], skip_special_tokens=True),
        "latency": latency,
        "throughput": throughput,
        "ram": ram_usage,
        "vram": vram_usage,
        "ctx_switch": ctx_switch
    }

# ==========================================
# 3. 실험 수행 (Baseline & Paged)
# ==========================================

print(">>> 측정 시작 (Baseline 로딩 중...)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = """### Instruction:\nIn this context, "LLM" means "Large Language Model". Explain it in one simple sentence.\n### Response:\n"""

# --- Baseline 측정 ---
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if device=="cuda" else torch.float32, 
    device_map=device
)
stats_base = measure_performance(model_base, tokenizer, prompt)

# 메모리 정리
del model_base
gc.collect()
if device == "cuda": torch.cuda.empty_cache()

# --- PagedAttention 측정 ---
print(">>> PagedAttention 로딩 및 패치 중...")
from paged_llama.llama.modeling.modeling_llama import PagedLlamaAttention
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.block_table import BlockTable

model_paged = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if device=="cuda" else torch.float32, 
    device_map=device
)
config = model_paged.config
pool = PagePool(
    num_blocks=2500, 
    num_heads=config.num_key_value_heads, 
    block_size=16, 
    head_dim=config.hidden_size // config.num_attention_heads, 
    device=device
)

table = BlockTable(block_size=16)
first_block = pool.allocate() 
table.add_block(first_block)

for layer in model_paged.model.layers:
    new_attn = PagedLlamaAttention(config=config, layer_idx=layer.self_attn.layer_idx)
    new_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
    new_attn.to(device)
    new_attn.pool = pool
    new_attn.block_table = table
    layer.self_attn = new_attn

stats_paged = measure_performance(model_paged, tokenizer, prompt)

# ==========================================
# 4. 최종 결과 출력
# ==========================================

print("\n" + "="*60)
print("[OUTPUT] PagedAttention Generation Result")
print(stats_paged['text'])
print("="*60 + "\n")

print(f"{'Metric':<25} | {'Baseline':<15} | {'Paged (Ours)':<15} |")
print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")
print(f"{'RAM Usage (MB)':<25} | {stats_base['ram']:<15.1f} | {stats_paged['ram']:<15.1f} |")
print(f"{'VRAM Usage (MB)':<25} | {stats_base['vram']:<15.1f} | {stats_paged['vram']:<15.1f} |")
print(f"{'Context Switch':<25} | {stats_base['ctx_switch']:<15} | {stats_paged['ctx_switch']:<15} |")
