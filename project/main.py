# /Users/yeongyeong/Desktop/3-2/종설/main.py

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 현재 경로를 추가하여 paged_llama를 인식하게 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paged_llama.llama.modeling.modeling_llama import PagedLlamaAttention
from paged_llama.llama.memory.page_pool import PagePool
from paged_llama.llama.memory.block_table import BlockTable

def patch_llama():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f">>> [1/3] 허깅페이스에서 Llama-3 로드 중... (쿠키 사용)")
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # PagePool 설정
    pool = PagePool(
        num_blocks=1024, 
        num_heads=model.config.num_attention_heads, 
        block_size=16, 
        head_dim=128, 
        device=model.device
    )

    print(f">>> [2/3] PagedAttention 부품 교체 시작...")
    for i, layer in enumerate(model.model.layers):
        new_attn = PagedLlamaAttention(config=model.config, layer_idx=i, page_pool=pool)
        # 기존 가중치 그대로 이식
        new_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
        layer.self_attn = new_attn
        
    print(f">>> [3/3] 엔진 교체 완료! 이제 Paged Attention으로 동작합니다.")
    return model, pool

if __name__ == "__main__":
    model, pool = patch_llama()
    # 이후 테스트 추론 코드 추가 가능
