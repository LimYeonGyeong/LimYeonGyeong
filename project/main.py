import gc
import os
import time
import psutil
import torch

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

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
def measure_performance(model, tokenizer, prompt_text, max_new_tokens=3):
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

    stats = {
        "text": generated_text,
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
        "used_blocks": 0,
        "block_utilization": 0.0,
    }

    return stats

@torch.no_grad()
def measure_baseline_multi(model, tokenizer, prompts, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

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

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    generated_tokens = (outputs.shape[1] - inputs["input_ids"].shape[1]) * inputs["input_ids"].shape[0]

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
        "used_blocks": 0,
        "block_utilization": 0.0,
    }

@torch.no_grad()
def measure_paged_multi(model, tokenizer, prompts, scheduler, pool, max_new_tokens=20):
    process = psutil.Process(os.getpid())

    request_ids = []
    block_tables = []
    encoded_prompts = []

    # 토크나이징을 루프 밖에서 1회만 수행
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        encoded_prompts.append({
            "input_ids": enc["input_ids"].to(model.device),
            "prompt_len": enc["input_ids"].shape[1],
        })

    for i, item in enumerate(encoded_prompts):
        rid = f"multi_req_{i}"
        request_ids.append(rid)

        prompt_len = item["prompt_len"]
        total_tokens = prompt_len + max_new_tokens

        bt = scheduler.allocate_for_request(rid, total_tokens)
        scheduler.set_seq_len(rid, 0)
        block_tables.append(bt)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.time()
    results = []

    for i, item in enumerate(encoded_prompts):
        request_id = request_ids[i]
        request_state = scheduler.get_request_state(request_id)

        for layer in model.model.layers:
            layer.self_attn.block_table = block_tables[i]
        model.model.block_table = block_tables[i]
        model.model.active_request_state = request_state

        generated = item["input_ids"].clone()
        prompt_len = item["prompt_len"]
        past_key_values = None

        # prefill
        scheduler.set_seq_len(request_id, 0)

        # 1) prefill
        print("\n[MAIN-PREFILL] before call")
        print(f"[CACHE-ID] before prefill id={id(past_key_values) if past_key_values is not None else None}")
        print(f"[REQ-STATE] before prefill = {model_paged.model.active_request_state}")
        print(f"[REQ-STATE-ID] before prefill = {id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}")

        outputs = model_paged(
            input_ids=generated,
            use_cache=True,
            past_key_values=None,
        )

        print(f"[CACHE-ID] after prefill call id={id(outputs.past_key_values) if outputs.past_key_values is not None else None}")

        past_key_values = outputs.past_key_values

        print(f"[CACHE-TYPE] after prefill = {type(past_key_values)}")
        print(f"[CACHE-SEQ] after prefill = {past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}")
        print(f"[REQ-STATE] after prefill = {model_paged.model.active_request_state}")
        print(f"[REQ-STATE-ID] after prefill = {id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}")

        scheduler.set_seq_len(request_id, prompt_len)

        print(f"[REQ-STATE] after scheduler.set_seq_len(prompt_len) = {model_paged.model.active_request_state}")

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # 2) decode
        for step in range(1, divergence_step + 1):
            last_token = generated[:, -1:]

            print(f"\n[MAIN-DECODE STEP {step}] before call")
            print(f"[CACHE-ID] before call id={id(past_key_values) if past_key_values is not None else None}")
            print(f"[CACHE-TYPE] before call = {type(past_key_values)}")
            print(f"[CACHE-SEQ] before call = {past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}")
            print(f"[REQ-STATE] before call = {model.model.active_request_state}")
            print(f"[REQ-STATE-ID] before call = {id(model.model.active_request_state) if model.model.active_request_state is not None else None}")

            outputs = model(
                input_ids=last_token,
                use_cache=True,
                past_key_values=past_key_values,
)

            print(f"[CACHE-ID] after call  id={id(outputs.past_key_values) if outputs.past_key_values is not None else None}")

            past_key_values = outputs.past_key_values

            print(f"[CACHE-TYPE] after call = {type(past_key_values)}")
            print(f"[CACHE-SEQ] after call = {past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}")
            print(f"[REQ-STATE] after call = {model_paged.model.active_request_state}")
            print(f"[REQ-STATE-ID] after call = {id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}")

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

        results.append((generated, prompt_len))

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    generated_tokens = sum(
        gen.shape[1] - prompt_len
        for gen, prompt_len in results
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
            used_blocks += len(bt.block_table[0])

    block_utilization = used_blocks / pool.num_blocks if pool.num_blocks > 0 else 0.0

    for rid in request_ids:
        scheduler.release_request(rid)

    decoded_texts = [
        tokenizer.decode(gen[0], skip_special_tokens=True)
        for gen, _ in results
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

def print_stats_table(title, stats_base, stats_paged, include_blocks=False):
    print(f"\n=== {title} ===")
    print(f"{'Metric':<25} | {'nomal':<15} | {'Paged':<15} |")
    print(f"{'Latency (sec)':<25} | {stats_base['latency']:<15.4f} | {stats_paged['latency']:<15.4f} |")
    print(f"{'Throughput (tok/s)':<25} | {stats_base['throughput']:<15.2f} | {stats_paged['throughput']:<15.2f} |")
    print(f"{'RAM Increase (MB)':<25} | {stats_base['ram_mb']:<15.1f} | {stats_paged['ram_mb']:<15.1f} |")
    print(f"{'Peak VRAM (MB)':<25} | {stats_base['peak_vram_mb']:<15.1f} | {stats_paged['peak_vram_mb']:<15.1f} |")
    print(f"{'Alloc VRAM (MB)':<25} | {stats_base['alloc_vram_mb']:<15.1f} | {stats_paged['alloc_vram_mb']:<15.1f} |")
    print(f"{'Reserved VRAM (MB)':<25} | {stats_base['reserved_vram_mb']:<15.1f} | {stats_paged['reserved_vram_mb']:<15.1f} |")
    print(f"{'Max Reserved VRAM (MB)':<25} | {stats_base['max_reserved_vram_mb']:<15.1f} | {stats_paged['max_reserved_vram_mb']:<15.1f} |")
    print(f"{'VRAM/token (KB)':<25} | {stats_base['vram_per_token_kb']:<15.2f} | {stats_paged['vram_per_token_kb']:<15.2f} |")
    print(f"{'Context Switch':<25} | {stats_base['ctx_switches']:<15} | {stats_paged['ctx_switches']:<15} |")

    if include_blocks:
        print(f"{'Used Blocks':<25} | {'-':<15} | {stats_paged['used_blocks']:<15} |")
        print(f"{'Block Utilization':<25} | {'-':<15} | {stats_paged['block_utilization']:<15.2%} |")


def set_layer0_debug(model, debug=False, debug_verbose=False):
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.debug = bool(debug and layer_idx == 0)
            layer.self_attn.debug_verbose = bool(debug_verbose and layer_idx == 0)


@torch.no_grad()
def collect_greedy_trace(model, tokenizer, prompt_text, max_new_tokens=20, scheduler=None, request_id=None):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generated = inputs["input_ids"].clone()
    prompt_len = generated.shape[1]
    past_key_values = None

    if scheduler is not None and request_id is not None:
        request_state = scheduler.get_request_state(request_id)
        model.model.active_request_state = request_state
        scheduler.set_seq_len(request_id, 0)

    trace = {
        "prompt_len": prompt_len,
        "steps": [],
        "final_text": "",
    }

    # 1) prefill
    outputs = model(
        input_ids=generated,
        use_cache=True,
        past_key_values=None,
    )
    past_key_values = outputs.past_key_values

    if scheduler is not None and request_id is not None:
        scheduler.set_seq_len(request_id, prompt_len)

    logits = outputs.logits[:, -1, :]
    topk = torch.topk(logits[0], k=5)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)

    trace["steps"].append({
        "step": 0,
        "input_ids": generated[0].detach().cpu().tolist(),
        "next_token_id": int(next_token.item()),
        "next_token_text": tokenizer.decode([int(next_token.item())], skip_special_tokens=False),
        "topk_ids": topk.indices.detach().cpu().tolist(),
        "topk_vals": topk.values.detach().cpu().float().tolist(),
        "past_seq_len_after": int(past_key_values.get_seq_length()) if past_key_values is not None and hasattr(past_key_values, "get_seq_length") else None,
    })

    generated = torch.cat([generated, next_token], dim=1)

    # 2) decode
    for step in range(1, max_new_tokens):
        last_token = generated[:, -1:]

        outputs = model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        topk = torch.topk(logits[0], k=5)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        trace["steps"].append({
            "step": step,
            "input_ids": last_token[0].detach().cpu().tolist(),
            "next_token_id": int(next_token.item()),
            "next_token_text": tokenizer.decode([int(next_token.item())], skip_special_tokens=False),
            "topk_ids": topk.indices.detach().cpu().tolist(),
            "topk_vals": topk.values.detach().cpu().float().tolist(),
            "past_seq_len_after": int(past_key_values.get_seq_length()) if past_key_values is not None and hasattr(past_key_values, "get_seq_length") else None,
        })

        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

    trace["final_text"] = tokenizer.decode(generated[0], skip_special_tokens=True)
    return trace


@torch.no_grad()
def debug_first_divergence(model_base, model_paged, tokenizer, prompt_text, scheduler, request_id, max_new_tokens=20):
    print("\n>>> Baseline vs Paged token-by-token 비교 시작")

    set_layer0_debug(model_paged, debug=False, debug_verbose=False)

    paged_trace = collect_greedy_trace(
        model=model_paged,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        scheduler=scheduler,
        request_id=request_id,
    )

    base_trace = collect_greedy_trace(
        model=model_base,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        scheduler=None,
        request_id=None,
    )

    divergence_step = None
    num_steps = min(len(base_trace["steps"]), len(paged_trace["steps"]))

    print("\n=== Token Compare ===")
    for step in range(num_steps):
        b = base_trace["steps"][step]
        p = paged_trace["steps"][step]

        same = (b["next_token_id"] == p["next_token_id"])
        print(
            f"[STEP {step}] "
            f"baseline={b['next_token_id']} ({repr(b['next_token_text'])}) | "
            f"paged={p['next_token_id']} ({repr(p['next_token_text'])}) | "
            f"{'MATCH' if same else 'DIFF'}"
        )

        if not same:
            divergence_step = step
            print("\n[FOUND] 첫 divergence step =", divergence_step)
            print("baseline topk ids  =", b["topk_ids"])
            print("baseline topk vals =", [round(v, 6) for v in b["topk_vals"]])
            print("paged topk ids     =", p["topk_ids"])
            print("paged topk vals    =", [round(v, 6) for v in p["topk_vals"]])
            print("baseline past_seq_len_after =", b["past_seq_len_after"])
            print("paged past_seq_len_after    =", p["past_seq_len_after"])
            break

    if divergence_step is None:
        print("\n[RESULT] max_new_tokens 범위 내에서는 divergence가 발견되지 않음")
        print("[BASELINE FINAL]")
        print(base_trace["final_text"])
        print("\n[PAGED FINAL]")
        print(paged_trace["final_text"])
        return None

    print("\n>>> divergence step에서 layer 0 debug 재실행")
    set_layer0_debug(model_paged, debug=True, debug_verbose=False)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model_paged.device)
    generated = inputs["input_ids"].clone()
    prompt_len = generated.shape[1]
    past_key_values = None

    request_state = scheduler.get_request_state(request_id)
    model_paged.model.active_request_state = request_state
    scheduler.set_seq_len(request_id, 0)

    # 1) prefill
    print("\n[MAIN-PREFILL] before call")
    print(f"[CACHE-ID] before prefill id={id(past_key_values) if past_key_values is not None else None}")
    print(f"[REQ-STATE] before prefill = {model_paged.model.active_request_state}")
    print(
        f"[REQ-STATE-ID] before prefill = "
        f"{id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}"
    )

    outputs = model_paged(
        input_ids=generated,
        use_cache=True,
        past_key_values=None,
    )

    print(
        f"[CACHE-ID] after prefill call id="
        f"{id(outputs.past_key_values) if outputs.past_key_values is not None else None}"
    )

    past_key_values = outputs.past_key_values

    print(f"[CACHE-TYPE] after prefill = {type(past_key_values)}")
    print(
        f"[CACHE-SEQ] after prefill = "
        f"{past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}"
    )
    print(f"[REQ-STATE] after prefill = {model_paged.model.active_request_state}")
    print(
        f"[REQ-STATE-ID] after prefill = "
        f"{id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}"
    )

    scheduler.set_seq_len(request_id, prompt_len)
    print(f"[REQ-STATE] after scheduler.set_seq_len(prompt_len) = {model_paged.model.active_request_state}")

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    # 2) decode
    for step in range(1, divergence_step + 1):
        last_token = generated[:, -1:]

        print(f"\n[MAIN-DECODE STEP {step}] before call")
        print(f"[CACHE-ID] before call id={id(past_key_values) if past_key_values is not None else None}")
        print(f"[CACHE-TYPE] before call = {type(past_key_values)}")
        print(
            f"[CACHE-SEQ] before call = "
            f"{past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}"
        )
        print(f"[REQ-STATE] before call = {model_paged.model.active_request_state}")
        print(
            f"[REQ-STATE-ID] before call = "
            f"{id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}"
        )

        outputs = model_paged(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past_key_values,
        )

        print(
            f"[CACHE-ID] after call  id="
            f"{id(outputs.past_key_values) if outputs.past_key_values is not None else None}"
        )

        past_key_values = outputs.past_key_values

        print(f"[CACHE-TYPE] after call = {type(past_key_values)}")
        print(
            f"[CACHE-SEQ] after call = "
            f"{past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 'N/A'}"
        )
        print(f"[REQ-STATE] after call = {model_paged.model.active_request_state}")
        print(
            f"[REQ-STATE-ID] after call = "
            f"{id(model_paged.model.active_request_state) if model_paged.model.active_request_state is not None else None}"
        )

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    set_layer0_debug(model_paged, debug=False, debug_verbose=False)

    return {
        "divergence_step": divergence_step,
        "baseline_trace": base_trace,
        "paged_trace": paged_trace,
    }

# -----------------------------
# PagedAttention 정확성 테스트
# HF cache OFF + PagePool only
# -----------------------------
@torch.no_grad()
def test_paged_generation_step_by_step(model, tokenizer, prompt_text, scheduler, request_id, max_new_tokens=20):
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generated = inputs["input_ids"].clone()
    prompt_len = generated.shape[1]
    past_key_values = None
    request_state = scheduler.get_request_state(request_id)
    model.model.active_request_state = request_state
    scheduler.set_seq_len(request_id, 0)

    # 1) prefill
    outputs = model(
        input_ids=generated,
        use_cache=True,
        past_key_values=None,
    )
    past_key_values = outputs.past_key_values
    scheduler.set_seq_len(request_id, prompt_len)

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    print(
        f"[STEP 0] token_id = {next_token.item()} | "
        f"token = {repr(tokenizer.decode(next_token[0], skip_special_tokens=False))}"
    )

    # 2) decode
    for step in range(1, max_new_tokens):
        last_token = generated[:, -1:]

        outputs = model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

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
def measure_paged_only(model, tokenizer, prompt_text, block_table, pool, scheduler, request_id, max_new_tokens=20):
    process = psutil.Process(os.getpid())
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    generated = inputs["input_ids"].clone()
    prompt_len = generated.shape[1]
    past_key_values = None
    request_state = scheduler.get_request_state(request_id)
    model.model.active_request_state = request_state
    scheduler.set_seq_len(request_id, 0)

    t0 = time.time()

    # 1) prefill
    outputs = model(
        input_ids=generated,
        use_cache=True,
        past_key_values=None,
    )
    past_key_values = outputs.past_key_values
    scheduler.set_seq_len(request_id, prompt_len)

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    # 2) decode
    for _ in range(max_new_tokens - 1):
        last_token = generated[:, -1:]

        outputs = model(
            input_ids=last_token,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

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

    if hasattr(block_table, "to_tensor"):
        bt = block_table.to_tensor(device="cpu")
        used_blocks = bt.shape[1]
    else:
        used_blocks = len(block_table.block_table[0])

    total_blocks = pool.num_blocks
    block_utilization = used_blocks / total_blocks if total_blocks > 0 else 0.0

    stats = {
        "text": generated_text,
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
    }

    return stats

# -----------------------------
# Attention patch
# -----------------------------
def patch_model_with_paged_attention(model, page_pool, block_table, debug=False, debug_verbose=False):
    # model-level cache info 연결
    model.model.page_pool = page_pool
    model.model.block_table = block_table

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
        new_attn.debug = debug
        new_attn.debug_verbose = debug_verbose
        new_attn.debug_stop_on_nonfinite = False

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
    tokenizer.padding_side = "left"

    prompt = (
        '### Instruction:\n'
        'Explain the difference between training and inference in LLMs.'
        'Explain it in one simple sentence.\n'
        '### Response:\n'
    )

    prompts = [
        "### Instruction:\nExplain what LLM is in one sentence.\n### Response:",
        "### Instruction:\nExplain Medusa LLM in one simple sentence.\n### Response:",
        "### Instruction:\nDescribe how attention works in transformers in one sentence.\n### Response:",
        "### Instruction:\nExplain KV cache in simple terms in one sentence.\n### Response:",
        "### Instruction:\nWhat is a transformer model? Answer in one sentence.\n### Response:",
        "### Instruction:\nExplain the difference between training and inference in LLMs.\n### Response:",
        "### Instruction:\nWhat is self-attention and why is it important?\n### Response:",
        "### Instruction:\nExplain what tokenization is in natural language processing.\n### Response:",
        "### Instruction:\nWhat is the purpose of positional encoding in transformers?\n### Response:",
        "### Instruction:\nExplain what fine-tuning means for language models.\n### Response:",
        "### Instruction:\nWhat is the role of embeddings in LLMs?\n### Response:",
        "### Instruction:\nExplain what beam search is in text generation.\n### Response:",
        "### Instruction:\nWhat is greedy decoding and how does it work?\n### Response:",
        "### Instruction:\nExplain why KV cache improves inference speed.\n### Response:",
        "### Instruction:\nWhat is the difference between encoder-only and decoder-only models?\n### Response:",
    ]
    # -------------------------
    # Baseline
    # -------------------------
    model_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    stats_base = measure_performance(model_base, tokenizer, prompt, max_new_tokens=10)

    # divergence 확인용 baseline trace
    baseline_debug_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

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
    model_paged.config.use_cache = True
    if hasattr(model_paged, "generation_config"):
        model_paged.generation_config.use_cache = True

    config = model_paged.config

    max_new_tokens = 20
    block_size = 16

    all_needed_blocks = 0
    for p in prompts:
        prompt_len = tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
        needed_tokens = prompt_len + max_new_tokens
        needed_blocks = (needed_tokens + block_size - 1) // block_size
        all_needed_blocks += needed_blocks

    # 여유분 조금 추가
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

    scheduler = SimpleScheduler(page_pool=pool, block_size=16)

    request_id = "req_1"

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    max_new_tokens = 20
    total_tokens = prompt_len + max_new_tokens

    block_table = scheduler.allocate_for_request(request_id, total_tokens)

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=block_table,
        debug=False,
        debug_verbose=False,
    )

    # PagePool 초기화
    pool.k_cache.zero_()
    pool.v_cache.zero_()

    # -------------------------
    # 1단계 정확성 테스트
    # -------------------------
    print(">>> HF cache OFF + PagePool only 테스트 중...")

    debug_result = debug_first_divergence(
        model_base=baseline_debug_model,
        model_paged=model_paged,
        tokenizer=tokenizer,
        prompt_text=prompt,
        scheduler=scheduler,
        request_id=request_id,
        max_new_tokens=10,
    )

    # -------------------------
    # 2단계 성능 측정
    # -------------------------
    pool.k_cache.zero_()
    pool.v_cache.zero_()
    scheduler.set_seq_len(request_id, 0)

    stats_paged = measure_paged_only(
        model=model_paged,
        tokenizer=tokenizer,
        prompt_text=prompt,
        block_table=block_table,
        pool=pool,
        scheduler=scheduler,
        request_id=request_id,
        max_new_tokens=max_new_tokens,
    )

    # -------------------------
    # 디버그 정보
    # -------------------------
    print("[DEBUG] PagePool / BlockTable 확인")
    print("pool.k_cache.shape =", pool.k_cache.shape)
    print("pool.v_cache.shape =", pool.v_cache.shape)

    if hasattr(block_table, "to_tensor"):
        bt = block_table.to_tensor(device=device)
        print("block_table.shape =", bt.shape)
        print("block_table[0, :10] =", bt[0, :10])
    else:
        print("block_table has no to_tensor()")

    if debug_result is not None:
        print("\n[DEBUG] divergence_step =", debug_result["divergence_step"])

    print("\n" + "=" * 60)
    print("[OUTPUT] PagedAttention Generation Result")
    print(stats_paged["text"])
    print("=" * 60 + "\n")

    # -------------------------
    # 결과 출력
    # -------------------------
    print_stats_table("Single Request Result", stats_base, stats_paged, include_blocks=True)

    scheduler.release_request(request_id)

    print("\n>>> Multi-request 실험 시작")

    # -------------------------
    # Baseline multi
    # -------------------------
    model_base_multi = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    stats_base_multi = measure_baseline_multi(
        model_base_multi, tokenizer, prompts, max_new_tokens=20
    )

    del model_base_multi
    del baseline_debug_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Paged multi
    # -------------------------
    stats_paged_multi = measure_paged_multi(
        model_paged, tokenizer, prompts, scheduler, pool, max_new_tokens=20
    )
    print("\n[OUTPUT] Multi Request Generation Results")
    for i, text in enumerate(stats_paged_multi["texts"]):
        print(f"\n--- Request {i+1} ---")
        print(text)

    print_stats_table("Multi Request Result", stats_base_multi, stats_paged_multi, include_blocks=True)


if __name__ == "__main__":
    main()