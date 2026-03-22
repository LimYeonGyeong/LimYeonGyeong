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
def patch_model_with_paged_attention(model, page_pool, block_table):
    """
    model.model.layers[i].self_attn 를 PagedLlamaAttention으로 교체
    """
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        new_attn = PagedLlamaAttention(
            config=model.config,
            layer_idx=layer_idx,
            page_pool=page_pool,
        ).to(next(old_attn.parameters()).device)

        # 기존 projection weight 복사
        new_attn.q_proj.weight.data.copy_(old_attn.q_proj.weight.data)
        new_attn.k_proj.weight.data.copy_(old_attn.k_proj.weight.data)
        new_attn.v_proj.weight.data.copy_(old_attn.v_proj.weight.data)
        new_attn.o_proj.weight.data.copy_(old_attn.o_proj.weight.data)

        # block_table 연결
        new_attn.block_table = block_table

        layer.self_attn = new_attn

    return model

@torch.no_grad()
def greedy_decode_fullseq(model, tokenizer, input_ids, max_new_tokens=20, device="cuda"):
    model.eval()
    generated = input_ids
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    # 1) 공백/개행 계열 토큰 미리 수집
    banned_token_ids = set()
    bad_texts = {"", " ", "\n", "\t", "\r", " ", " ", " "}  # 마지막은 non-breaking space

    for tid in range(tokenizer.vocab_size):
        txt = tokenizer.decode([tid], skip_special_tokens=False)
        if txt in bad_texts:
            banned_token_ids.add(tid)

    banned_token_ids = list(banned_token_ids)
    print(f"[DEBUG] banned_token_ids count = {len(banned_token_ids)}")

    for step in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=False
        )

        logits = outputs.logits[:, -1, :].clone()

        # 2) 공백 토큰 막기
        if len(banned_token_ids) > 0:
            logits[:, banned_token_ids] = torch.finfo(logits.dtype).min

        # 3) greedy decode
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        print(f"[DECODE-STEP {step}] next_token_id = {token_id} | token_text = {repr(token_text)}")

        generated = torch.cat([generated, next_token], dim=1)

        # attention mask도 같이 늘려주기
        new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        # eos 만나면 종료
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break

    return generated

def measure_performance(model, tokenizer, prompt_text):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = 20

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.time()
    with torch.no_grad():
        outputs = greedy_decode_fullseq(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            device=model.device
        )
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    latency = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0
    ram_end = process.memory_info().rss / 1024 / 1024

    ctx_end = process.num_ctx_switches()
    ctx_switch = (ctx_end.voluntary + ctx_end.involuntary) - (ctx_start.voluntary + ctx_start.involuntary)

    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[DEBUG] decoded text preview")
    print(decoded_text)

    return {
        "text": decoded_text,
        "latency": latency,
        "throughput": max_new_tokens / latency,
        "peak_vram": peak_vram,
        "total_ram": ram_end,
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
# main.py의 PagePool 생성 부분
config = model_paged.config
pool = PagePool(
    num_blocks=2500, 
    num_layers=config.num_hidden_layers, # TinyLlama의 경우 22 전달
    num_heads=config.num_key_value_heads, 
    block_size=16, 
    head_dim=config.hidden_size // config.num_attention_heads, 
    device=device,
    dtype=model_paged.dtype
)

shared_block_table = BlockTable(block_size=16)
# ... (이후 블록 할당 및 레이어 패치 로직은 동일)

# 테스트 문장에 필요한 충분한 블록 할당 (예: 100개)
for _ in range(100):
    shared_block_table.add_block(pool.allocate())

# 3. 레이어 패치 루프
for i, layer in enumerate(model_paged.model.layers):
    new_attn = PagedLlamaAttention(
        config=config,
        layer_idx=i,
        page_pool=pool 
    )
    new_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
    
    new_attn.to(device)
    new_attn.page_pool = pool
    
    # 4. [★핵심] 모든 레이어에 '동일한' BlockTable 객체를 주입
    # modeling_llama.py에서 hasattr(..., "to_tensor")로 처리하도록 설계했습니다.
    new_attn.block_table = shared_block_table 
    
    layer.self_attn = new_attn

pool.k_cache.zero_()
pool.v_cache.zero_()
# 패치 완료 후 성능 측정
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