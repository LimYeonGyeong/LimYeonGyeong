# paged_llama/llama/utils/__init__.py

import logging

# 1. PyTorch가 설치되어 있는지 확인하는 함수
def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

# 2. 로깅 설정 (기본 logging 모듈 사용)
# 필요하다면 기본 로거를 반환하도록 설정
def get_logger(name):
    return logging.getLogger(name)

# __version__ 관련 오류 방지 (상위 폴더에서 가져오거나 직접 정의)
try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.1"