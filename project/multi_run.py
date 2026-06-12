import os
import gc
import sys
import time
import math
import random
import builtins
from contextlib import contextmanager

import torch
import psutil



sys.path.append("/LimYeonGyeong/project")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from paged_llama.llama.modeling.modeling_llama import (
    PagedLlamaAttention,
    LlamaForCausalLM as PagedLlamaForCausalLM,
)

from paged_llama.llama.config.configuration_llama import LlamaConfig
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.simple_scheduler import SimpleScheduler

# 기존 main.py에서 검증된 patch helper 재사용
from main import patch_model_with_paged_attention

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
        num_key_value_heads=getattr(
            hf_config,
            "num_key_value_heads",
            hf_config.num_attention_heads,
        ),
        max_position_embeddings=hf_config.max_position_embeddings,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        hidden_act=hf_config.hidden_act,
        pad_token_id=hf_config.pad_token_id,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        attention_bias=getattr(
            hf_config,
            "attention_bias",
            False,
        ),
        mlp_bias=getattr(
            hf_config,
            "mlp_bias",
            False,
        ),
        attention_dropout=getattr(
            hf_config,
            "attention_dropout",
            0.0,
        ),
    )


def load_paged_model():

    print(">>> HF config / state_dict 로딩 중...")

    hf_config = AutoConfig.from_pretrained(MODEL_ID)

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
        if device == "cuda"
        else torch.float32,
    )

    hf_model = hf_model.to("cpu")

    state_dict = hf_model.state_dict()

    print(">>> 로컬 paged 모델 생성 중...")

    local_config = build_local_config_from_hf(hf_config)

    model = PagedLlamaForCausalLM(local_config)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

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

def load_baseline_model():
    """
    HuggingFace 기본 TinyLlama 모델을 로드한다.
    Paged Attention과 비교하기 위한 normal baseline이다.
    """

    print(">>> Baseline HF 모델 로딩 중...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
        if device == "cuda"
        else torch.float32,
    )

    model = model.to(device)
    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    return model

def make_prompt(question: str) -> str:
    """
    TinyLlama가 안정적으로 답변하도록 prompt 형식을 고정한다.

    핵심:
    - 질문은 매 실행마다 build_mixed_prompts()에서 자동 생성
    - 답변은 plain English sentence로 제한
    - markdown, code, bullet point를 금지해서 이상한 ###, 괄호, 기호 생성을 줄임
    """

    return (
        "### Instruction:\n"
        f"{question}\n"
        "Answer in clear plain English.\n"
        "Do not use markdown, bullet points, code blocks, or numbered lists.\n"
        "### Response:\n"
    )


def build_mixed_prompts(
    n_requests: int = 2,
    seed: int | None = None,
):
    """
    multi-request 실험용 prompt를 자동 생성한다.

    기존 문제:
    - 항상 같은 질문만 사용해서 실험 다양성이 낮았음.
    - "LLM"만 쓰면 Long-Lived Memory처럼 잘못 해석될 수 있었음.

    수정:
    - 질문 후보를 여러 개 두고 매 실행마다 랜덤 선택
    - LLM은 large language model이라고 풀어서 질문
    - request마다 서로 다른 질문을 넣음
    """

    rng = random.Random(seed)

    question_bank = [
        "Explain what a large language model is in one clear sentence.",
        "Explain transformer attention for a beginner in three simple sentences.",
        "Explain why KV cache is useful during LLM inference.",
        "Explain paged attention in simple terms.",
        "Explain why batching improves LLM inference throughput.",
        "Explain the difference between prefill and decode in LLM generation.",
        "Explain why memory usage matters when serving large language models.",
        "Explain how attention helps a transformer understand context.",
        "Explain what token generation means in an autoregressive language model.",
        "Explain why reducing unnecessary attention computation can improve latency.",
        "Explain what a cache miss means in a memory system.",
        "Explain why GPU memory access can become a bottleneck.",
        "Explain what sequence length means in transformer inference.",
        "Explain what request-level scheduling means in LLM serving.",
        "Explain why masking is used in causal language models.",
    ]

    if n_requests <= len(question_bank):
        selected_questions = rng.sample(
            question_bank,
            n_requests,
        )
    else:
        selected_questions = [
            rng.choice(question_bank)
            for _ in range(n_requests)
        ]

    prompts = [
        make_prompt(question)
        for question in selected_questions
    ]

    print("\n================ GENERATED QUESTIONS ================\n")

    for i, question in enumerate(selected_questions):
        print(f"[Question {i + 1}] {question}")

    print()

    return prompts


def print_topk_logits(tokenizer, logits, step_label: str, k: int = 5):

    topk = torch.topk(
        logits[:, -1, :],
        k=k,
        dim=-1,
    )

    for row in range(topk.indices.shape[0]):

        print(f"\n[TOPK][{step_label}] row={row}")

        for rank in range(k):

            token_id = int(topk.indices[row, rank].item())
            token_text = tokenizer.decode([token_id])
            score = float(topk.values[row, rank].item())

            print(
                f"rank={rank + 1} "
                f"token_id={token_id} "
                f"decoded={repr(token_text)} "
                f"logit={score:.4f}"
            )


def decode_request_outputs(tokenizer, runtimes):

    full_texts = []
    response_texts = []

    print("\n===== GENERATED TOKEN IDS =====")

    for i, rt in enumerate(runtimes):

        token_ids = rt["generated"][0].detach().cpu().tolist()
        prompt_len = int(rt["prompt_len"])
        response_ids = token_ids[prompt_len:]

        print(f"\n[REQUEST {i + 1}] prompt_len={prompt_len}")
        print(f"full_token_ids={token_ids}")
        print(f"response_token_ids={response_ids}")

        full_text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        full_texts.append(full_text)
        response_texts.append(response_text)

    return full_texts, response_texts

def _print_topk_logits(
    logits,
    tokenizer,
    tag: str,
    k: int = 5,
):
    """
    correctness 디버깅용 top-k 출력 함수.

    enable_token_debug=True일 때만 호출한다.
    평소 성능 측정에서는 이 함수를 호출하지 않아야 latency가 덜 왜곡된다.
    """

    topk = torch.topk(
        logits[:, -1, :],
        k=k,
        dim=-1,
    )

    for row in range(topk.indices.shape[0]):

        print(f"\n[TOPK][{tag}] row={row}")

        for rank in range(k):

            token_id = int(
                topk.indices[row, rank].item()
            )

            decoded = tokenizer.decode([token_id])

            value = float(
                topk.values[row, rank].item()
            )

            print(
                f"rank={rank + 1} "
                f"token_id={token_id} "
                f"decoded={repr(decoded)} "
                f"logit={value:.4f}"
            )

@torch.no_grad()
def measure_baseline_multi(
    model,
    tokenizer,
    prompts,
    max_new_tokens=40,
):
    """
    HuggingFace 기본 generate() 기반 multi-request baseline 측정 함수.

    반환 지표:
    - latency
    - throughput
    - RAM increase
    - peak / allocated / reserved VRAM
    - VRAM per generated token
    - context switch
    """

    model.eval()

    process = psutil.Process(os.getpid())

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_tokens_total = int(
        attention_mask.sum().item()
    )

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.perf_counter()

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    ctx_end = process.num_ctx_switches()
    ram_end = process.memory_info().rss / 1024 / 1024

    latency = t1 - t0

    generated_tokens_total = int(
        outputs.shape[0] * max_new_tokens
    )

    total_tokens = (
        prompt_tokens_total
        + generated_tokens_total
    )

    throughput = (
        generated_tokens_total / latency
        if latency > 0
        else 0.0
    )

    if device == "cuda":
        peak_vram_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

        alloc_vram_mb = (
            torch.cuda.memory_allocated()
            / 1024
            / 1024
        )

        reserved_vram_mb = (
            torch.cuda.memory_reserved()
            / 1024
            / 1024
        )

        max_reserved_vram_mb = (
            torch.cuda.max_memory_reserved()
            / 1024
            / 1024
        )
    else:
        peak_vram_mb = 0.0
        alloc_vram_mb = 0.0
        reserved_vram_mb = 0.0
        max_reserved_vram_mb = 0.0

    vram_per_token_kb = (
        peak_vram_mb * 1024 / generated_tokens_total
        if generated_tokens_total > 0
        else 0.0
    )

    context_switch = (
        ctx_end.voluntary
        - ctx_start.voluntary
        + ctx_end.involuntary
        - ctx_start.involuntary
    )

    decoded_texts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    response_texts = []

    prompt_width = input_ids.shape[1]

    for row in range(outputs.shape[0]):
        response_ids = outputs[
            row,
            prompt_width:,
        ]

        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        response_texts.append(response_text)

    return {
        "latency": latency,
        "throughput": throughput,
        "ram_increase_mb": ram_end - ram_start,
        "peak_vram_mb": peak_vram_mb,
        "alloc_vram_mb": alloc_vram_mb,
        "reserved_vram_mb": reserved_vram_mb,
        "max_reserved_vram_mb": max_reserved_vram_mb,
        "vram_per_token_kb": vram_per_token_kb,
        "context_switch": context_switch,
        "used_blocks": None,
        "block_utilization": None,
        "texts": decoded_texts,
        "responses": response_texts,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "total_tokens": total_tokens,
    }

@torch.no_grad()
def measure_paged_multi(
    model,
    tokenizer,
    prompts,
    scheduler,
    pool,
    max_new_tokens=40,
    enable_token_debug: bool = False,
):
    """
    Paged multi-request generation 실행 및 성능 측정 함수.

    기능:
    1. multi-request prefill / decode 수행
    2. request별 block_table / request_state 연결
    3. response only / full text 저장
    4. latency, throughput, RAM, VRAM, context switch 측정
    5. used blocks, block utilization 계산
    """

    model.eval()

    process = psutil.Process(os.getpid())

    runtimes = []

    # -------------------------------------------------
    # 0) 토크나이징
    # -------------------------------------------------
    encoded_inputs = []
    max_prompt_len = 0

    for prompt in prompts:

        enc = tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(model.device)

        prompt_len = input_ids.shape[1]

        encoded_inputs.append(
            (
                input_ids,
                prompt_len,
            )
        )

        max_prompt_len = max(
            max_prompt_len,
            prompt_len,
        )

    # batch decode에서는 모든 request가 같은 step 수만큼 진행되므로
    # block table은 batch 내 최대 길이에 맞춰 할당
    shared_total_tokens = (
        max_prompt_len
        + max_new_tokens
    )

    # -------------------------------------------------
    # 1) request runtime/state 준비
    # -------------------------------------------------
    for i, (input_ids, prompt_len) in enumerate(encoded_inputs):

        rid = f"multi_req_{i}"

        block_table = scheduler.allocate_for_request(
            rid,
            shared_total_tokens,
        )

        scheduler.set_seq_len(
            rid,
            0,
        )

        request_state = scheduler.get_request_state(
            rid
        )

        runtimes.append({
            "request_id": rid,
            "block_table": block_table,
            "request_state": request_state,
            "generated": input_ids.clone(),
            "prompt_len": prompt_len,
            "finished": False,
        })

    # -------------------------------------------------
    # 2) 성능 측정 시작
    # -------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    ram_start = process.memory_info().rss / 1024 / 1024
    ctx_start = process.num_ctx_switches()

    t0 = time.perf_counter()

    # -------------------------------------------------
    # 3) PREFILL
    # prompt 길이가 같은 request끼리 묶어서 prefill
    # -------------------------------------------------
    prefill_groups = {}

    for rt in runtimes:

        prefill_groups.setdefault(
            rt["prompt_len"],
            [],
        ).append(rt)

    for prompt_len in sorted(prefill_groups.keys()):

        group_rts = prefill_groups[prompt_len]

        batch_input_ids = torch.cat(
            [
                rt["generated"]
                for rt in group_rts
            ],
            dim=0,
        )

        batch_block_tables = [
            rt["block_table"]
            for rt in group_rts
        ]

        batch_request_states = [
            rt["request_state"]
            for rt in group_rts
        ]

        # runtime 연결
        for layer in model.model.layers:

            layer.self_attn.block_table = (
                batch_block_tables
            )

            layer.self_attn.page_pool = pool

        model.model.block_table = batch_block_tables
        model.model.page_pool = pool
        model.model.active_request_state = None
        model.model.active_request_states = (
            batch_request_states
        )

        # prefill 시작 전 seq_len은 0
        for rt in group_rts:

            scheduler.set_seq_len(
                rt["request_id"],
                0,
            )

            rt["request_state"]["seq_len"] = 0

        outputs = model(
            input_ids=batch_input_ids,
            use_cache=True,
            past_key_values=None,
        )

        if enable_token_debug:
            _print_topk_logits(
                outputs.logits,
                tokenizer,
                tag=f"prefill_len_{prompt_len}",
            )

        next_tokens = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        for i, rt in enumerate(group_rts):

            next_token = next_tokens[i:i + 1]

            rt["generated"] = torch.cat(
                [
                    rt["generated"],
                    next_token,
                ],
                dim=1,
            )

            # prefill 후에는 prompt + 첫 생성 token까지 들어간 상태
            rt["request_state"]["seq_len"] = (
                rt["generated"].shape[1]
            )

            if (
                tokenizer.eos_token_id is not None
                and int(next_token.item())
                == tokenizer.eos_token_id
            ):
                rt["finished"] = True

    # -------------------------------------------------
    # 4) DECODE
    # 이미 prefill에서 1개 token을 생성했으므로
    # range는 1부터 시작
    # -------------------------------------------------
    for step in range(1, max_new_tokens):

        active_rts = [
            rt
            for rt in runtimes
            if not rt["finished"]
        ]

        if not active_rts:
            break

        batch_last_tokens = torch.cat(
            [
                rt["generated"][:, -1:]
                for rt in active_rts
            ],
            dim=0,
        )

        # request별 seq_len 동기화
        for rt in active_rts:

            rt["request_state"]["seq_len"] = (
                rt["generated"].shape[1]
            )

        # 현재 입력 token은 generated의 마지막 token
        # 따라서 cache_position은 현재 token이 들어갈 위치 = seq_len - 1
        batch_cache_position = torch.tensor(
            [
                int(rt["request_state"]["seq_len"]) - 1
                for rt in active_rts
            ],
            device=model.device,
            dtype=torch.long,
        )

        batch_position_ids = (
            batch_cache_position.unsqueeze(1)
        )

        batch_block_tables = [
            rt["block_table"]
            for rt in active_rts
        ]

        batch_request_states = [
            rt["request_state"]
            for rt in active_rts
        ]

        # runtime 연결
        for layer in model.model.layers:

            layer.self_attn.block_table = (
                batch_block_tables
            )

            layer.self_attn.page_pool = pool

        model.model.block_table = batch_block_tables
        model.model.page_pool = pool
        model.model.active_request_state = None
        model.model.active_request_states = (
            batch_request_states
        )

        outputs = model(
            input_ids=batch_last_tokens,
            use_cache=True,
            past_key_values=None,
            cache_position=batch_cache_position,
            position_ids=batch_position_ids,
        )

        if enable_token_debug:
            _print_topk_logits(
                outputs.logits,
                tokenizer,
                tag=f"decode_step_{step}",
            )

        next_tokens = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        if enable_token_debug:

            print(f"\n[STEP {step}] generated next tokens")

            for row in range(next_tokens.shape[0]):

                token_id = int(
                    next_tokens[row, 0].item()
                )

                decoded = tokenizer.decode([token_id])

                print(
                    f"row={row} "
                    f"next_token_id={token_id} "
                    f"decoded={repr(decoded)}"
                )

        for i, rt in enumerate(active_rts):

            next_token = next_tokens[i:i + 1]

            rt["generated"] = torch.cat(
                [
                    rt["generated"],
                    next_token,
                ],
                dim=1,
            )

            if (
                tokenizer.eos_token_id is not None
                and int(next_token.item())
                == tokenizer.eos_token_id
            ):
                rt["finished"] = True

    # -------------------------------------------------
    # 5) 성능 측정 종료
    # -------------------------------------------------
    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    ram_end = process.memory_info().rss / 1024 / 1024
    ctx_end = process.num_ctx_switches()

    latency = t1 - t0

    # -------------------------------------------------
    # 6) token 수 계산
    # -------------------------------------------------
    prompt_tokens_total = sum(
        int(rt["prompt_len"])
        for rt in runtimes
    )

    generated_tokens_total = sum(
        int(rt["generated"].shape[1] - rt["prompt_len"])
        for rt in runtimes
    )

    total_tokens = (
        prompt_tokens_total
        + generated_tokens_total
    )

    throughput = (
        generated_tokens_total / latency
        if latency > 0
        else 0.0
    )

    total_throughput = (
        total_tokens / latency
        if latency > 0
        else 0.0
    )

    # -------------------------------------------------
    # 7) VRAM 측정
    # -------------------------------------------------
    if device == "cuda":

        peak_vram_mb = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
        )

        alloc_vram_mb = (
            torch.cuda.memory_allocated()
            / 1024
            / 1024
        )

        reserved_vram_mb = (
            torch.cuda.memory_reserved()
            / 1024
            / 1024
        )

        max_reserved_vram_mb = (
            torch.cuda.max_memory_reserved()
            / 1024
            / 1024
        )

    else:

        peak_vram_mb = 0.0
        alloc_vram_mb = 0.0
        reserved_vram_mb = 0.0
        max_reserved_vram_mb = 0.0

    vram_per_token_kb = (
        peak_vram_mb * 1024 / generated_tokens_total
        if generated_tokens_total > 0
        else 0.0
    )

    # -------------------------------------------------
    # 8) context switch
    # -------------------------------------------------
    context_switch = (
        ctx_end.voluntary
        - ctx_start.voluntary
        + ctx_end.involuntary
        - ctx_start.involuntary
    )

    voluntary_ctx_switches = (
        ctx_end.voluntary
        - ctx_start.voluntary
    )

    involuntary_ctx_switches = (
        ctx_end.involuntary
        - ctx_start.involuntary
    )

    # -------------------------------------------------
    # 9) Paged block 사용량 계산
    # -------------------------------------------------
    final_lengths = [
        int(rt["generated"].shape[1])
        for rt in runtimes
    ]

    used_blocks = sum(
        math.ceil(length / pool.block_size)
        for length in final_lengths
    )

    total_block_capacity = (
        used_blocks
        * pool.block_size
    )

    block_utilization = (
        total_tokens / total_block_capacity * 100
        if total_block_capacity > 0
        else 0.0
    )

    # -------------------------------------------------
    # 10) 디코딩 결과 정리
    # -------------------------------------------------
    decoded_texts = []
    decoded_responses = []

    generated_token_ids = []
    response_token_ids = []

    for rt in runtimes:

        full_ids = rt["generated"][0]
        response_ids = full_ids[rt["prompt_len"]:]

        full_text = tokenizer.decode(
            full_ids,
            skip_special_tokens=True,
        )

        response_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )

        decoded_texts.append(full_text)
        decoded_responses.append(response_text)

        generated_token_ids.append(
            full_ids.tolist()
        )

        response_token_ids.append(
            response_ids.tolist()
        )

    # -------------------------------------------------
    # 11) 결과 반환
    # -------------------------------------------------
    return {
        "latency": latency,
        "throughput": throughput,
        "total_throughput": total_throughput,

        "ram_increase_mb": ram_end - ram_start,
        "ram_start_mb": ram_start,
        "ram_end_mb": ram_end,

        "peak_vram_mb": peak_vram_mb,
        "alloc_vram_mb": alloc_vram_mb,
        "reserved_vram_mb": reserved_vram_mb,
        "max_reserved_vram_mb": max_reserved_vram_mb,
        "vram_per_token_kb": vram_per_token_kb,

        "context_switch": context_switch,
        "voluntary_ctx_switches": voluntary_ctx_switches,
        "involuntary_ctx_switches": involuntary_ctx_switches,

        "used_blocks": used_blocks,
        "block_utilization": block_utilization,

        "texts": decoded_texts,
        "responses": decoded_responses,
        "generated_token_ids": generated_token_ids,
        "response_token_ids": response_token_ids,

        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "total_tokens": total_tokens,

        "num_requests": len(runtimes),
        "max_new_tokens": max_new_tokens,
    }

def print_multi_request_result_table(
    normal_stats,
    paged_stats,
):
    """
    normal baseline과 paged attention 결과를 함께 출력한다.

    출력 순서:
    1. Normal 답변
    2. Paged 답변
    3. Multi Request 성능 비교 표
    """

    # -------------------------------------------------
    # 1) 답변 출력
    # -------------------------------------------------
    print("\n================ NORMAL RESPONSE ONLY ================\n")

    for i, text in enumerate(
        normal_stats.get("responses", [])
    ):

        print(f"--- Normal Request {i + 1} response ---")
        print(repr(text))
        print()

    print("\n================ PAGED RESPONSE ONLY ================\n")

    for i, text in enumerate(
        paged_stats.get("responses", [])
    ):

        print(f"--- Paged Request {i + 1} response ---")
        print(repr(text))
        print()

    # -------------------------------------------------
    # 2) 성능 표 출력
    # -------------------------------------------------
    def fmt_float(value, digits=2):

        if value is None:
            return "-"

        return f"{value:.{digits}f}"

    def fmt_int(value):

        if value is None:
            return "-"

        return str(value)

    def fmt_percent(value):

        if value is None:
            return "-"

        return f"{value:.2f}%"

    rows = [
        (
            "Latency (sec)",
            fmt_float(
                normal_stats.get("latency"),
                4,
            ),
            fmt_float(
                paged_stats.get("latency"),
                4,
            ),
        ),
        (
            "Throughput (tok/s)",
            fmt_float(
                normal_stats.get("throughput"),
                2,
            ),
            fmt_float(
                paged_stats.get("throughput"),
                2,
            ),
        ),
        (
            "RAM Increase (MB)",
            fmt_float(
                normal_stats.get("ram_increase_mb"),
                1,
            ),
            fmt_float(
                paged_stats.get("ram_increase_mb"),
                1,
            ),
        ),
        (
            "Peak VRAM (MB)",
            fmt_float(
                normal_stats.get("peak_vram_mb"),
                1,
            ),
            fmt_float(
                paged_stats.get("peak_vram_mb"),
                1,
            ),
        ),
        (
            "Alloc VRAM (MB)",
            fmt_float(
                normal_stats.get("alloc_vram_mb"),
                1,
            ),
            fmt_float(
                paged_stats.get("alloc_vram_mb"),
                1,
            ),
        ),
        (
            "Reserved VRAM (MB)",
            fmt_float(
                normal_stats.get("reserved_vram_mb"),
                1,
            ),
            fmt_float(
                paged_stats.get("reserved_vram_mb"),
                1,
            ),
        ),
        (
            "Max Reserved VRAM (MB)",
            fmt_float(
                normal_stats.get("max_reserved_vram_mb"),
                1,
            ),
            fmt_float(
                paged_stats.get("max_reserved_vram_mb"),
                1,
            ),
        ),
        (
            "VRAM/token (KB)",
            fmt_float(
                normal_stats.get("vram_per_token_kb"),
                2,
            ),
            fmt_float(
                paged_stats.get("vram_per_token_kb"),
                2,
            ),
        ),
        (
            "Context Switch",
            fmt_int(
                normal_stats.get("context_switch"),
            ),
            fmt_int(
                paged_stats.get("context_switch"),
            ),
        ),
        (
            "Used Blocks",
            fmt_int(
                normal_stats.get("used_blocks"),
            ),
            fmt_int(
                paged_stats.get("used_blocks"),
            ),
        ),
        (
            "Block Utilization",
            fmt_percent(
                normal_stats.get("block_utilization"),
            ),
            fmt_percent(
                paged_stats.get("block_utilization"),
            ),
        ),
    ]

    metric_width = 32
    normal_width = 18
    paged_width = 18

    print("\n=== Multi Request Result ===")

    print(
        f"{'Metric':<{metric_width}} | "
        f"{'normal':<{normal_width}} | "
        f"{'Paged':<{paged_width}} |"
    )

    for metric, normal_value, paged_value in rows:

        print(
            f"{metric:<{metric_width}} | "
            f"{normal_value:<{normal_width}} | "
            f"{paged_value:<{paged_width}} |"
        )

def multi_only_main():

    print(">>> Multi Request 비교 실행")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    num_requests = 2

    prompts = build_mixed_prompts(
        n_requests=num_requests
    )

    max_new_tokens = 20


    block_size = 16

    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()

    print(">>> Paged multi 로딩 중...")

    model_paged = load_paged_model()

    config = model_paged.config

    max_prompt_len = 0

    for p in prompts:

        prompt_len = tokenizer(
            p,
            return_tensors="pt",
        )["input_ids"].shape[1]

        max_prompt_len = max(
            max_prompt_len,
            prompt_len,
        )

    shared_total_tokens = (
        max_prompt_len + max_new_tokens
    )

    blocks_per_request = (
        shared_total_tokens + block_size - 1
    ) // block_size

    all_needed_blocks = (
        blocks_per_request * len(prompts)
    )

    safety_margin = 4

    num_blocks = (
        all_needed_blocks + safety_margin
    )

    pool = PagePool(
        num_blocks=num_blocks,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        block_size=block_size,
        head_dim=(
            config.hidden_size
            // config.num_attention_heads
        ),
        device=device,
        dtype=model_paged.dtype,
    )

    scheduler = SimpleScheduler(
        page_pool=pool,
        block_size=block_size,
    )

    dummy_prompt_len = tokenizer(
        prompts[0],
        return_tensors="pt",
    )["input_ids"].shape[1]

    dummy_total_tokens = (
        dummy_prompt_len + max_new_tokens
    )

    dummy_block_table = (
        scheduler.allocate_for_request(
            "dummy_req",
            dummy_total_tokens,
        )
    )

    model_paged = patch_model_with_paged_attention(
        model=model_paged,
        page_pool=pool,
        block_table=dummy_block_table,
        debug=True,
        debug_verbose=False,
    )

    scheduler.release_request("dummy_req")

    stats_paged_multi = measure_paged_multi(
        model=model_paged,
        tokenizer=tokenizer,
        prompts=prompts,
        scheduler=scheduler,
        pool=pool,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":

    multi_only_main()