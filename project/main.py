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
# ==========================================
# 2. 성능 측정 엔진 (개선 버전)
# ==========================================
def measure_performance(model, tokenizer, prompt_text):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # [변경] 측정 전 캐시 및 통계 초기화
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # [변경] 시작 시점의 절대 메모리 기록 (RSS 사용)
    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, # 테스트용으로 짧게 설정
            do_sample=True,    # 답변 활성화를 위해 샘플링 켬
            use_cache=True
        )
    if device == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    
    # [변경] 피크 VRAM 및 최종 RAM 점유량 계산
    latency = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0
    ram_end = process.memory_info().rss / 1024 / 1024
    
    ctx_end = process.num_ctx_switches()
    ctx_switch = (ctx_end.voluntary + ctx_end.involuntary) - (ctx_start.voluntary + ctx_start.involuntary)
    
    return {
        "text": tokenizer.decode(outputs[0], skip_special_tokens=True),
        "latency": latency,
        "throughput": 50 / latency,
        "peak_vram": peak_vram,    # 순간 최대 GPU 사용량
        "total_ram": ram_end,      # 현재 프로세스가 먹고 있는 전체 RAM
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

# [★핵심 수정] 모델의 실제 dtype을 가져옵니다. (보통 float16)
model_dtype = model_paged.dtype 
config = model_paged.config

# 캐시 풀 생성 시 모델과 동일한 dtype 주입
# PagePool 생성 시 모델의 dtype을 명시적으로 전달
pool = PagePool(
    num_blocks=2500, 
    num_heads=config.num_key_value_heads, 
    block_size=16, 
    head_dim=config.hidden_size // config.num_attention_heads, 
    device=device,
    dtype=model_paged.dtype # <--- 추가됨
)

shared_block_table = BlockTable(block_size=16)
# ... (이후 블록 할당 및 레이어 패치 로직은 동일)

# 테스트 문장에 필요한 충분한 블록 할당 (예: 100개)
for _ in range(100):
    shared_block_table.add_block(pool.allocate())

# 3. 레이어 패치 루프
for i, layer in enumerate(model_paged.model.layers):
    new_attn = PagedLlamaAttention(config=config, layer_idx=i)
    new_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
    
    new_attn.to(device)
    new_attn.page_pool = pool
    
    # 4. [★핵심] 모든 레이어에 '동일한' BlockTable 객체를 주입
    # modeling_llama.py에서 hasattr(..., "to_tensor")로 처리하도록 설계했습니다.
    new_attn.block_table = shared_block_table 
    
    layer.self_attn = new_attn

# 패치 완료 후 성능 측정
stats_paged = measure_performance(model_paged, tokenizer, prompt)

# ==========================================
# 4. 최종 결과 출력 (수정된 Key 반영)
# ==========================================

print("\n" + "="*60)
print("[OUTPUT] PagedAttention Generation Result")
print(stats_paged['text'])
print("="*60 + "\n")

print(f"{'Metric':<25} | {'nomal':<15} | {'Paged':<15} |")
print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")

print(f"{'Total RAM (MB)':<25} | {stats_base['total_ram']:<15.1f} | {stats_paged['total_ram']:<15.1f} |")

print(f"{'Peak VRAM (MB)':<25} | {stats_base['peak_vram']:<15.1f} | {stats_paged['peak_vram']:<15.1f} |")

print(f"{'Context Switch':<25} | {stats_base['ctx_switch']:<15} | {stats_paged['ctx_switch']:<15} |")