# paged_llama/llama/utils/type_helpers.py

from typing import Union, Any

# 외부 의존성 제거
class TensorType: pass
class PaddingStrategy: pass
class TruncationStrategy: pass

def positive_any_number(value):
    if value is not None and value < 0:
        raise ValueError(f"Value must be positive, got {value}")

def positive_int(value):
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError(f"Value must be a positive integer, got {value}")

# 검증 로직 단순화 (Pass 처리)
def padding_validator(value): pass
def truncation_validator(value): pass
def tensor_type_validator(value): pass