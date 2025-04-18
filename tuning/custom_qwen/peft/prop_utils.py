from typing import List, Optional

from tuning.custom_qwen.peft.prop_lora import LORA_ATTN_MODULES
from torchtune.modules.peft import (
    get_lora_module_names,
)


def validate_missing_and_unexpected_for_propulsion(
    prop_attn_modules: List[LORA_ATTN_MODULES],
    apply_prop_to_mlp: bool,
    apply_prop_to_output: bool,
    base_missing: Optional[List[str]] = None,
    base_unexpected: Optional[List[str]] = None,
    prop_missing: Optional[List[str]] = None,
    prop_unexpected: Optional[List[str]] = None,
) -> None:
    """
    Validate that Propulsion state dict loading was done properly.
    """
    prop_modules = get_lora_module_names(
        prop_attn_modules, apply_prop_to_mlp, apply_prop_to_output
    )

    is_prop_param = lambda x: any(
        [
            ".".join([k, "propulsion"]) in x
            or ".".join([k, "prop_linear.weight"]) in x
            or ".".join([k, "prop_linear.bias"]) in x
            for k in prop_modules
        ]
    )

    if base_missing:
        for k in base_missing:
            if not is_prop_param(k):
                raise AssertionError(
                    f"Missing non-propulsion key {k} from base model dict"
                )
    if base_unexpected:
        raise AssertionError(f"Unexpected keys {base_unexpected} loading base model")
    if prop_missing:
        for k in prop_missing:
            if not is_prop_param(k):
                raise AssertionError(
                    f"Missing propulsion key {k} from adapter state dict"
                )
    if prop_unexpected:
        raise AssertionError(f"Unexpected keys {prop_unexpected} loading adapter")


def validate_missing_and_unexpected_for_lora_and_propulsion(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
    prop_attn_modules: List[LORA_ATTN_MODULES],
    apply_prop_to_mlp: bool,
    apply_prop_to_output: bool,
    base_missing: Optional[List[str]] = None,
    base_unexpected: Optional[List[str]] = None,
    adapter_missing: Optional[List[str]] = None,
    adapter_unexpected: Optional[List[str]] = None,
) -> None:
    """
    A more memory-efficient way to validate that LoRA state dict loading was done properly.

    This function uses a model's LoRA config to check that LoRA and/or base model weights
    are loaded into the full model correctly. This function relies only on the values of missing and
    unexpected as returned by the load_state_dict API with strict=False. This allows us to do the
    validation without any additional calls to .state_dict(), which use additional memory.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether LoRA is applied to each MLP linear.
        apply_lora_to_output (bool): whether LoRA is applied to the final output projection.
        prop_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            Propulsion should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_prop_to_mlp (bool): whether Propulsion is applied to each MLP linear.
        apply_prop_to_output (bool): whether Propulsion is applied to the final output projection.
        base_missing (Optional[List[str]]): List of missing keys when loading base model weights.
            Default: None
        base_unexpected (Optional[List[str]]): List of unexpected keys when loading base model weights.
            Default: None
        adapter_missing (Optional[List[str]]): List of missing keys when loading adapter weights.
            Default: None
        adapter_unexpected (Optional[List[str]]): List of unexpected keys when loading adapter weights.
            Default: None

    Returns:
        None

    Raises:
        AssertionError:
            If base_missing contains any base model keys, **or**
            if base_unexpected is nonempty, **or**
            if adapter_missing contains any adapter keys, **or**
            if adapter_unexpected is nonempty.
    """
    lora_modules = get_lora_module_names(
        lora_attn_modules, apply_lora_to_mlp, apply_lora_to_output
    )
    prop_modules = get_lora_module_names(
        prop_attn_modules, apply_prop_to_mlp, apply_prop_to_output
    )

    is_lora_param = lambda x: any(
        [
            ".".join([k, "lora"]) in x or ".".join([k, "magnitude"]) in x
            for k in lora_modules
        ]
    )

    is_prop_param = lambda x: any(
        [
            ".".join([k, "propulsion"]) in x
            or ".".join([k, "prop_linear.weight"]) in x
            or ".".join([k, "prop_linear.bias"]) in x
            for k in prop_modules
        ]
    )

    if base_missing:
        for k in base_missing:
            if not is_lora_param(k) and not is_prop_param(k):
                raise AssertionError(
                    f"Missing non-LoRA and non-propulsion key {k} from base model dict"
                )
    if base_unexpected:
        raise AssertionError(f"Unexpected keys {base_unexpected} loading base model")
    if adapter_missing:
        for k in adapter_missing:
            if is_lora_param(k):
                raise AssertionError(f"Missing LoRA key {k} from adapter state dict")
            if is_prop_param(k):
                raise AssertionError(
                    f"Missing propulsion key {k} from adapter state dict"
                )
    if adapter_unexpected:
        raise AssertionError(f"Unexpected keys {adapter_unexpected} loading adapter")
