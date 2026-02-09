# paged_llama/llama/modeling/modeling_utils.py

import torch
import torch.nn as nn
from typing import Any, Optional, Union, Dict, List
import copy
import os

from paged_llama.llama import initialization as init
from paged_llama.llama.configuration_utils import PreTrainedConfig
from paged_llama.llama.conversion_mapping import get_model_conversion_mapping
from paged_llama.llama.utils import logging

logger = logging.get_logger(__name__)

class ModuleUtilsMixin:
    """
    파라미터 수 계산, 디바이스 확인 등 모듈 유틸리티
    """
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        total_params = 0
        for name, param in self.named_parameters():
            if exclude_embeddings and "embedding" in name:
                continue
            if not only_trainable or param.requires_grad:
                total_params += param.numel()
        return total_params

class PreTrainedModel(nn.Module, ModuleUtilsMixin):
    r"""
    모든 모델의 기본이 되는 클래스입니다.
    설정(Config) 관리, 가중치 저장/로드 등의 기능을 담당합니다.
    """
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _no_split_modules = None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Config must be instance of PreTrainedConfig, got {type(config)}"
            )
        self.config = config
        self.name_or_path = config.name_or_path if hasattr(config, "name_or_path") else ""
        
    def get_memory_footprint(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        간소화된 from_pretrained 메서드.
        실제 가중치 로딩 로직은 복잡하므로, 여기서는 Config를 로드하고
        랜덤 초기화된 모델을 반환하는 형태로 구현합니다.
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        
        # 1. Config 로드
        if config is None:
            if cls.config_class is None:
                raise ValueError("This model class does not have a config_class defined.")
            # 실제로는 파일에서 읽어야 하지만, 편의상 기본값 생성
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. 모델 초기화
        model = cls(config, *model_args, **kwargs)

        # 3. State Dict 로드 (있다면)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model

    def save_pretrained(self, save_directory):
        pass # 더미 구현
    