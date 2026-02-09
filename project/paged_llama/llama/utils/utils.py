# paged_llama/llama/utils/utils.py

from . import logging

# 1. 타입 힌트용 더미
class TransformersKwargs:
    pass

# 2. 문서화 데코레이터 (기능은 없지만 에러 방지용)
def auto_docstring(func):
    return func

# 3. 튜플 반환 데코레이터
def can_return_tuple(func):
    return func

# 4. 커널 사용 관련 데코레이터
def use_kernel_forward_from_hub(kernel_name):
    def decorator(cls):
        return cls
    return decorator

def use_kernelized_func(func):
    def decorator(cls):
        return cls
    return decorator

def use_kernel_func_from_hub(kernel_name):
    def decorator(func):
        return func
    return decorator

def check_model_inputs(func):
    """
    입력값을 검증하는 데코레이터 (현재는 더미 구현)
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper