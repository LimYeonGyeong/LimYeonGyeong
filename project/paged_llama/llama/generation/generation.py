# paged_llama/llama/generation/generation.py

from transformers.generation import GenerationMixin

class GenerationConfig:
    """
    텍스트 생성을 위한 설정 클래스
    """
    def __init__(self, **kwargs):
        self.max_length = kwargs.pop("max_length", 20)
        self.do_sample = kwargs.pop("do_sample", False)
    
    @classmethod
    def from_model_config(cls, config):
        return cls()