# paged_llama/llama/utils/imports.py

import importlib.util
import operator
import os
import sys
from collections import OrderedDict
from typing import Any

# 최소한의 패키지 확인 로직만 남김
def _is_package_available(pkg_name: str, return_version: bool = False) -> bool:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if return_version:
        return package_exists, "0.0.0" # 버전 체크 무력화
    return package_exists

def is_torch_available():
    return _is_package_available("torch")

def is_vision_available():
    return False # 이미지 처리 안 함

def is_scipy_available():
    return False

def is_accelerate_available():
    return _is_package_available("accelerate")
    
# 기타 불필요한 함수들은 빈 껍데기(Dummy)로 처리하거나 삭제
def direct_transformers_import(path: str, file="__init__.py"):
    return sys.modules[__name__]