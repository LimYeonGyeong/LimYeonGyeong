# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integrations with other Python libraries.
"""

import copy
import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import re
import shutil
import sys
import tempfile
import warnings
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import packaging.version

from transformers.utils.import_utils import _is_package_available


if os.getenv("WANDB_MODE") == "offline":
    print("[INFO] Running in WANDB offline mode")

from llama import __version__ as version
from llama.utils import (
    PushToHubMixin,
    flatten_dict,
    is_datasets_available,
    is_pandas_available,
    is_torch_available,
    logging,
)


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    Environment:
        - **TENSORBOARD_LOGGING_DIR** (`str`, *optional*, defaults to `None`):
            The logging dir to log the results. Default value is os.path.join(args.output_dir, default_logdir())
    """

    def __init__(self, tb_writer=None):
        if not is_tensorboard_available():
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter

        self._SummaryWriter = SummaryWriter
        self.tb_writer = tb_writer
        self.logging_dir = os.getenv("TENSORBOARD_LOGGING_DIR", None)
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

    def _init_summary_writer(self, args):
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=self.logging_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                # overwrite logging dir for trials
                self.logging_dir = os.path.join(args.output_dir, default_logdir(), trial_name)

        if self.logging_dir is None:
            self.logging_dir = os.path.join(args.output_dir, default_logdir())

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, str):
                    self.tb_writer.add_text(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


def save_model_architecture_to_file(model: Any, output_dir: str):
    with open(f"{output_dir}/model_architecture.txt", "w+") as f:
        if isinstance(model, PreTrainedModel):
            print(model, file=f)
        elif is_torch_available() and (
            isinstance(model, (torch.nn.Module, PushToHubMixin)) and hasattr(model, "base_model")
        ):
            print(model, file=f)

