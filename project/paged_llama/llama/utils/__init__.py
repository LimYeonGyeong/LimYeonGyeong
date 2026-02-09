import logging as _native_logging

# 1. PyTorch가 설치되어 있는지 확인하는 함수
def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

# 2. 로깅 문제 해결 (핵심 수정 부분)
# 코드는 'logging.get_logger'를 찾지만, 표준 모듈은 'logging.getLogger'입니다.
# 이를 연결해주는 래퍼(Wrapper) 클래스를 만듭니다.
class LoggingWrapper:
    # get_logger를 호출하면 표준의 getLogger를 실행
    def get_logger(self, name):
        return _native_logging.getLogger(name)
    
    # 그 외(INFO, WARNING 등)는 표준 logging 모듈의 기능을 그대로 사용
    def __getattr__(self, name):
        return getattr(_native_logging, name)

# 위에서 만든 클래스를 'logging'이라는 이름으로 할당
logging = LoggingWrapper()

# 3. 버전 처리
try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.1"