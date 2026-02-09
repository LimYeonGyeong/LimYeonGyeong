# paged_llama/llama/utils/generic.py

import inspect
from collections import OrderedDict
from typing import Any, MutableMapping
from enum import Enum

# 로컬 utils 참조
from . import logging

logger = logging.get_logger(__name__)

class ModelOutput(OrderedDict):
    """
    Base class for model's outputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_tuple(self) -> tuple:
        return tuple(self[k] for k in self.keys())
        
# 필요한 유틸리티 클래스만 남김
class TensorType(str, Enum):
    PYTORCH = "pt"
    NUMPY = "np"