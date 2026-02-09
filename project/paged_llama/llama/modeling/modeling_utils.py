# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from paged_llama.llama import initialization as init
from paged_llama.llama.configuration_utils import PreTrainedConfig
from paged_llama.llama.conversion_mapping import get_model_conversion_mapping
from paged_llama.llama.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    convert_and_load_state_dict_in_model,
    revert_weight_conversion,
)
from paged_llama.llama.distributed import DistributedConfig
from paged_llama.llama.dynamic_module_utils import custom_object_save
from paged_llama.llama.generation import CompileConfig, GenerationConfig
from paged_llama.llama.integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled, is_fsdp_enabled
from paged_llama.llama.integrations.accelerate import (
    _get_device_map,
    accelerate_disk_offload,
    accelerate_dispatch,
    check_and_set_device_map,
    expand_device_map,
    get_device,
    load_offloaded_parameter,
)
from paged_llama.llama.integrations.deepspeed import _load_state_dict_into_zero3_model
from paged_llama.llama.integrations.eager_paged import eager_paged_attention_forward
from paged_llama.llama.integrations.flash_attention import flash_attention_forward
from paged_llama.llama.integrations.flash_paged import paged_attention_forward
from paged_llama.llama.integrations.flex_attention import flex_attention_forward
from paged_llama.llama.integrations.hub_kernels import is_kernel
from paged_llama.llama.integrations.peft import maybe_load_adapters
from paged_llama.llama.integrations.sdpa_attention import sdpa_attention_forward
from paged_llama.llama.integrations.sdpa_paged import sdpa_attention_paged_forward
from paged_llama.llama.integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    _get_parameter_tp_plan,
    distribute_model,
    initialize_tensor_parallelism,
    repack_weights,
    replace_state_dict_local_with_dtensor,
    shard_and_distribute_module,
    verify_tp_plan,
)
from paged_llama.llama.loss.loss_utils import LOSS_MAPPING
from paged_llama.llama.modeling_flash_attention_utils import lazy_import_flash_attention, lazy_import_paged_flash_attention
from paged_llama.llama.modeling.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from paged_llama.llama.pytorch_utils import id_tensor_storage
from paged_llama.llama.quantizers import HfQuantizer
from paged_llama.llama.quantizers.auto import get_hf_quantizer
from paged_llama.llama.quantizers.quantizers_utils import get_module_from_name
from paged_llama.llama.safetensors_conversion import auto_conversion
from paged_llama.llama.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    DUMMY_INPUTS,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    KernelConfig,
    PushToHubMixin,
    cached_file,
    check_torch_load_is_safe,
    copy_func,
    has_file,
    is_accelerate_available,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_grouped_mm_available,
    is_kernels_available,
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from paged_llama.llama.utils.generic import _CAN_RECORD_REGISTRY, GeneralInterface, OutputRecorder
from paged_llama.llama.utils.hub import DownloadKwargs, create_and_tag_model_card, get_checkpoint_shard_files
from paged_llama.llama.utils.import_utils import (
    is_huggingface_hub_greater_or_equal,
    is_sagemaker_mp_enabled,
    is_tracing,
)
from paged_llama.llama.utils.loading_report import log_state_dict_report
from paged_llama.llama.utils.quantization_config import QuantizationMethod



if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import extract_model_from_parallel


_torch_distributed_available = torch.distributed.is_available()
_is_dtensor_available = _torch_distributed_available and is_torch_greater_or_equal("2.5")
if _is_dtensor_available:
    from torch.distributed.tensor import DTensor

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.get_logger(__name__)

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
_is_quantized = False
_is_ds_init_called = False

# Mapping from flash attention implementations to their kernel fallback repositories
FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": "kernels-community/vllm-flash-attn3",
}


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    # `get_default_device` was only introduced in torch>=2.3 - use cpu otherwise to align the behavior
    default_device = torch.get_default_device() if is_torch_greater_or_equal("2.3") else torch.device("cpu")
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    if len(state_dict) == 0:
        return torch.float32
    return next(iter(state_dict.values())).dtype


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


if is_torch_greater_or_equal("2.3.0"):
    str_to_torch_dtype["U16"] = torch.uint16
    str_to_torch_dtype["U32"] = torch.uint32
    str_to_torch_dtype["U64"] = torch.uint64

class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(param.device for param in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(param.dtype for param in self.parameters() if param.is_floating_point())

    

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]

        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)
        if is_loaded_in_4bit:
            import bitsandbytes as bnb

        total_params = 0
        for name, param in self.named_parameters():
            if exclude_embeddings and name in embedding_param_names:
                continue
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_params += param.numel() * 2 * num_bytes
                else:
                    total_params += param.numel()

        return total_params


class EmbeddingAccessMixin:
    """
    Base utilities to regroup getters and setters for embeddings.
    Introduces the `input_layer_embed` attribute, which indicates
    where the input embeddings come from and where they
    should be set.
    """

    _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")

        # 1) Direct attribute (most NLP models).
        if (default_embedding := getattr(self, name, None)) is not None:
            return default_embedding
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision/audio models).
        if hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            return getattr(self.embeddings, name)
        # 3) Encoder/decoder wrappers (e.g., `self.model.embed_tokens` or similar overrides).
        if hasattr(self, "model") and hasattr(self.model, name):
            return getattr(self.model, name)

        if hasattr(self, "base_model"):
            base_model = self.base_model
            if base_model is not None and base_model is not self:
                return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"`get_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
        )

    def set_input_embeddings(self, value: nn.Module):
        """Fallback setter that handles **~70%** of models in the code-base.

        Order of attempts:
        1. `self.<_input_embed_layer>` (direct attribute)
        2. `self.embeddings.<_input_embed_layer>` (nested embeddings for vision/audio models)
        3. `self.model.<_input_embed_layer>` (encoder/decoder models)
        4. delegate to the *base model* if one exists
        5. otherwise raise `NotImplementedError` so subclasses still can (and
            should) override for exotic layouts.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")
        # 1) Direct attribute (most NLP models)
        if hasattr(self, name):
            setattr(self, name, value)
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision models)
        elif hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            setattr(self.embeddings, name, value)
        # 3) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`
        elif hasattr(self, "model") and hasattr(self.model, name):
            setattr(self.model, name, value)
        # 4) recurse once into the registered *base* model (e.g. for encoder/decoder)
        elif hasattr(self, "base_model") and self.base_model is not self:
            self.base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError(
                f"`set_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
            )

    


class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings

    Class attributes (overridden by derived classes):

        - **config_class** ([`PreTrainedConfig`]) -- A subclass of [`PreTrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):
    """

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _checkpoint_conversion_mapping = {}  # used for BC support in VLMs, not meant to be used by new models

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None

    _keep_in_fp32_modules = None
    # the _keep_in_fp32_modules will avoid casting to anything other than float32, except bfloat16
    # to also prevent bfloat16 casting, use the _keep_in_fp32_modules_strict flag
    _keep_in_fp32_modules_strict = None

    dtype_plan: dict[str, torch.dtype] | None = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    supports_gradient_checkpointing = False
    _is_stateful = False

    # Flash Attention support
    _supports_flash_attn = False

    # SDPA support
    _supports_sdpa = False

    # Flex Attention support
    _supports_flex_attn = False

    _can_compile_fullgraph = False

    # A tensor parallel plan to be applied to the model when TP is enabled. For
    # top-level models, this attribute is currently defined in respective model
    # code. For base models, this attribute comes from
    # `config.base_model_tp_plan` during `__init__`.
    # It should identify the layers exactly: if you want to TP model.language_model.layers.fc1
    # by passing `tp_plan` to the init, it should be {"model.language_model.layers.fc1":"colwise"}
    # for example.
    _tp_plan = None

    # tensor parallel degree to which model is sharded to.
    _tp_size = None

    # A pipeline parallel plan specifying the layers which may not be present
    # on all ranks when PP is enabled. For top-level models, this attribute is
    # currently defined in respective model code. For base models, this
    # attribute comes from `config.base_model_pp_plan` during `post_init`.
    #
    # The variable names for the inputs and outputs of the specified layers can
    # be indexed using the `PipelineParallel` enum as follows:
    # - `_pp_plan["layers"][PipelineParallel.inputs]`
    # - `_pp_plan["layers"][PipelineParallel.outputs]`
    _pp_plan = None

    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend = False
    _can_record_outputs = None

    # Attributes used mainly in multimodal LLMs, though all models contain a valid field for these
    # Possible values are: text, image, video, audio and time
    input_modalities: str | list[str] = "text"  # most models are text


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation, is_init_check=True
        )
        # Check the experts implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._experts_implementation_internal = self._check_and_adjust_experts_implementation(
            self.config._experts_implementation
        )
        if self.can_generate():
            self.generation_config = GenerationConfig.from_model_config(config)

        # for initialization of the loss
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
        self._keep_in_fp32_modules_strict = copy.copy(self.__class__._keep_in_fp32_modules_strict)
        self.dtype_plan = {}

        if isinstance(self._keep_in_fp32_modules, list):
            self.dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules, torch.float32))
        if isinstance(self._keep_in_fp32_modules_strict, list):
            self.dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules_strict, torch.float32))

        self._no_split_modules = self._no_split_modules or []
        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs  # added for executorch support only


    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum(param.nelement() * param.element_size() for param in self.parameters())
        if return_buffers:
            mem_bufs = sum(buf.nelement() * buf.element_size() for buf in self.buffers())
            mem = mem + mem_bufs
        return mem

    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute,
            # it's necessary to manually call the `cuda` method on HQQLinear layers.
            super().cuda(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if len(args) > 0:
                        device = args[0]
                    else:
                        device = kwargs.get("device", "cuda")
                    module.cuda(device)
            return self

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
        return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute, we must
            # explicitly move the parameters to the target device for each HQQLinear layer after `to`.
            super().to(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if "device" in kwargs:
                        device = kwargs["device"]
                    else:
                        device = args[0]
                    if "dtype" in kwargs:
                        dtype = kwargs["dtype"]
                    elif dtype_present_in_args:
                        dtype = arg
                    else:
                        dtype = None
                    # Due to the current messy implementation of HQQLinear, updating `compute_dtype`
                    # followed by calling the `cuda` method achieves the intended behavior of `to`,
                    # even when the target device is CPU.
                    if dtype is not None:
                        module.compute_dtype = dtype
                    module.cuda(device)
            return self

        if dtype_present_in_args and getattr(self, "quantization_method", None) == QuantizationMethod.QUARK:
            raise ValueError("Casting a Quark quantized model to a new `dtype` is not supported.")

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    
    def set_use_kernels(self, use_kernels, kernel_config):
        if use_kernels:
            if not is_kernels_available():
                raise ValueError(
                    "`use_kernels=True` requires kernels>=0.9.0. Please install the latest version with `pip install -U kernels`"
                )
            from kernels import use_kernel_mapping

            from .integrations.hub_kernels import register_kernel_mapping_transformers

            register_kernel_mapping_transformers()

            if kernel_config is not None and isinstance(kernel_config, KernelConfig):
                # This will make sure the mapping is valid, and the layers are registered in the model
                kernel_config.sanitize_kernel_mapping(self)

                # This will create a compatible mapping for the model with the kernels library
                kernel_config.create_compatible_mapping(self)

                # This is a context manager to override the default kernel mapping
                # We are calling kernelize inside this context manager using the use_kernels setter
                # Param inherit_mapping should be False to avoid still loading kernel from remote
                inherit_mapping = not kernel_config.use_local_kernel
                with use_kernel_mapping(kernel_config.kernel_mapping, inherit_mapping=inherit_mapping):
                    self.use_kernels = True
            # We use the default kernel mapping in .integrations.hub_kernels
            else:
                self.use_kernels = True
        else:
            self.use_kernels = False

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PreTrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = True,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PreTrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PreTrainedConfig`],
                    - a string or path valid as input to [`~PreTrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

                Accept HF kernel references in the form:
                  <namespace>/<repo_name>[@<revision>][:<kernel_name>]

                - <namespace> and <repo_name> are any non-"/" and non-":" sequences.
                - "@<revision>" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123".
                - ":<kernel_name>" is optional and selects a function inside the kernel repo.
                - Both options can appear together and in this order only: @revision first, then :kernel_name.
                - We intentionally allow a leading "<wrapper>|" prefix (e.g., "flash|...") because the code
                  strips it before loading; '|' is not excluded in the character classes here.

                Examples that match:
                  "org/model"
                  "org/model@main"
                  "org/model:custom_kernel"
                  "org/model@v1.2.3:custom_kernel"
            experts_implementation (`str`, *optional*):
                The experts implementation to use in the model (if relevant). Can be any of:

                - `"eager"` (sequential implementation of the experts matrix multiplications).
                - `"batched_mm"` (using [`torch.bmm`](https://pytorch.org/docs/stable/generated/torch.bmm.html)).
                - `"grouped_mm"` (using [`torch._grouped_mm`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html)).

                By default, if available, `grouped_mm` will be used for torch>=2.9.0. The default is otherwise the sequential `"eager"` implementation.

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*, defaults to `"auto"`):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `dtype` or `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            tp_plan (`Optional[Union[dict, str]]`, *optional*):
                A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Use `tp_plan="auto"` to
                use the predefined plan based on the model. If it's a dict, then it should match between module names and desired layout.
                Note that if you use it, you should launch your script accordingly with `torchrun [args] script.py`. This will be much
                faster than using a `device_map`, but has limitations.
            tp_size (`str`, *optional*):
                A torch tensor parallel degree. If not provided would default to world size.
            device_mesh (`torch.distributed.DeviceMesh`, *optional*):
                A torch device mesh. If not provided would default to world size. Used only for tensor parallel for now.
                If provided, it has to contain dimension named `"tp"` in case it's > 1 dimensional, this dimension will be used for tensor parallelism
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq).
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            key_mapping (`dict[str, str], *optional*):
                A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
                architecture, but was not converted accordingly.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PreTrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        ```
        """
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        distributed_config: DistributedConfig = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        key_mapping = kwargs.pop("key_mapping", None)

        if distributed_config is not None and tp_plan is None:
            tp_plan = "auto"

        # Not used anymore -- remove them from the kwargs
        for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
            _ = kwargs.pop(name, None)

        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }
        download_kwargs_with_commit = {**download_kwargs, "commit_hash": commit_hash}

        if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
            raise ValueError(
                "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
            )

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        if tp_plan is not None or tp_size is not None:  # TP warnings, and setup
            device_map, device_mesh, tp_size = initialize_tensor_parallelism(
                tp_plan, tp_size=tp_size, device_mesh=device_mesh, device_map=device_map
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if adapter_kwargs is None:
            adapter_kwargs = {}

        _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs = maybe_load_adapters(
            pretrained_model_name_or_path,
            download_kwargs_with_commit,
            **adapter_kwargs,
        )
        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **download_kwargs,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
            commit_hash = model_kwargs.pop("_commit_hash", commit_hash)
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs
            commit_hash = getattr(config, "_commit_hash", commit_hash)

        download_kwargs_with_commit["commit_hash"] = commit_hash

        # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
        # to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, device_map, weights_only, user_agent
        )

        if gguf_file:
            if hf_quantizer is not None:
                raise ValueError(
                    "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
                )
            if device_map is not None and (
                (isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map
            ):
                raise RuntimeError(
                    "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                    "loaded from GGUF files."
                )

        if kernel_config is not None and not use_kernels:
            logger.warning_once(
                "A kernel_config was provided but use_kernels is False; setting use_kernels=True automatically. To suppress this warning, explicitly set use_kernels to True."
            )
            use_kernels = True

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=variant,
            gguf_file=gguf_file,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs_with_commit,
            user_agent=user_agent,
            is_remote_code=cls._auto_class is not None,
            transformers_explicit_filename=getattr(config, "transformers_weights", None),
        )

        is_quantized = hf_quantizer is not None

        if gguf_file:
            from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
            # passed directly as a kwarg from now on
            with torch.device("meta"):
                dummy_model = cls(config)
            state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
                "tensors"
            ]

        # Find the correct dtype based on current state
        config, dtype = _get_dtype(
            dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only, hf_quantizer
        )

        config.name_or_path = pretrained_model_name_or_path
        model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called)
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

            if hf_quantizer is not None:  # replace module with quantized modules (does not touch weights)
                hf_quantizer.preprocess_model(
                    model=model,
                    dtype=dtype,
                    device_map=device_map,
                    checkpoint_files=checkpoint_files,
                    use_kernels=use_kernels,
                )

        # Obtain the weight conversion mapping for this model if any are registered
        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)

        if _torch_distributed_available and device_mesh is not None:  # add hooks to nn.Modules: no weights
            model = distribute_model(model, tp_plan, distributed_config, device_mesh, tp_size)

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)

        # Finalize model weight initialization
        model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            checkpoint_files,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            device_map=device_map,
            disk_offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=dtype,
            hf_quantizer=hf_quantizer,
            device_mesh=device_mesh,
            weights_only=weights_only,
            weight_mapping=weight_conversions,
        )

        model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        model.set_use_kernels(use_kernels, kernel_config)

        # If it is a model with generation capabilities, attempt to load generation files (generation config,
        # custom generate function)
        if model.can_generate() and hasattr(model, "adjust_generation_fn"):
            model.adjust_generation_fn(
                generation_config,
                from_auto_class,
                from_pipeline,
                pretrained_model_name_or_path,
                **download_kwargs,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # If the device_map has more than 1 device: dispatch model with hooks on all devices
        if device_map is not None and len(set(device_map.values())) > 1:
            accelerate_dispatch(model, hf_quantizer, device_map, offload_folder, offload_index, offload_buffers)

        if hf_quantizer is not None:
            model.hf_quantizer = hf_quantizer
            hf_quantizer.postprocess_model(
                model
            )  # usually a no-op but sometimes needed, e.g to remove the quant config when dequantizing

        if _adapter_model_path is not None:
            adapter_kwargs["key_mapping"] = key_mapping
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info
        return model

    @property
    def use_kernels(self) -> bool:
        return getattr(self, "_use_kernels", False)

    @use_kernels.setter
    def use_kernels(self, value: bool) -> None:
        # Avoid re-kernelizing if already enabled
        if bool(value) and getattr(self, "_use_kernels", False):
            return

        if value:
            self.kernelize()
        else:
            if getattr(self, "_use_kernels", False):
                logger.warning_once(
                    "Disabling kernels at runtime is a no-op as there is no 'unkernelize' routine; keeping current kernels active."
                )
            self._use_kernels = False



PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
if PreTrainedModel.push_to_hub.__doc__ is not None:
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )


def unwrap_model(model: nn.Module, recursive: bool = False) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.
    """
    # Use accelerate implementation if available (should always be the case when using torch)
    # This is for pytorch, as we also have to handle things like dynamo
    if is_accelerate_available():
        kwargs = {}
        if recursive:
            kwargs["recursive"] = recursive
        return extract_model_from_parallel(model, **kwargs)
    else:
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model
