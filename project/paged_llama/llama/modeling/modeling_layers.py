import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Union

# 상위 폴더(llama)의 memory 폴더에서 Cache 가져오기
from ..memory.cache_utils import Cache

# Transformers 공식 모듈 사용
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(super().__call__),
                *args,
                **kwargs
            )
        return super().__call__(*args, **kwargs)

class GenericForQuestionAnswering(nn.Module): pass
class GenericForSequenceClassification(nn.Module): pass
class GenericForTokenClassification(nn.Module): pass