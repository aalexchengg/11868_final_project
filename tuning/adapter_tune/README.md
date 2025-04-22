# Tuning w/ Adapters Explanation

Below descibes different configs for use in [fim_adapter_recipe.py](../custom_recipes/fim/fim_adapter_recipe.py)

## LoRA

See [debug_lora_config.yaml](debug_lora_config.yaml) for an example of setting up LoRA. Some key sections:

- `do_lora: True`
  - Actually enables LoRA during training, for prop+lora you need to keep the lora params so the config is parsed correctly but set do_lora to False
- `_component_: torchtune.models.qwen2_5.lora_qwen2_5_0_5b`
  - This changes whether the base model architecture supports just LoRA, just Propulsion, or both - if doing just LoRA we can use the model directly form `torchtune`
- `lora_attn_modules`, `apply_lora_to_mlp`, `apply_lora_to_output`
  - Which layers get lora projection matrices added

Everything else is pretty self-explanatory and doesn't differ from the default config

## Propulsion

See [prop_config.yaml](prop_config.yaml) for an example of setting up just Propulsion. Some key sections:

- `do_prop: True`
- `_component_: tuning.custom_qwen.qwen2_5.propulsion_qwen2_5_0_5b`
  - Note! We're using our custom model code here to integrate the propulsion layers.
- `_component_: tuning.custom_qwen.qwen2_5.CustomFullModelHFCheckpointer`

While it's easier to just take the existing model and do a traversal through and add the layers, this fucks _all_ of the torchtune code for things like checkpointing and setup. As such, there's a lot of custom files in [custom_qwen/](../custom_qwen/) that are copies of torchtune files except with added support for Propulsion layers

- `prop_attn_modules`, `apply_prop_to_mlp`, `apply_prop_to_output`
- Which layers get propulsion vectors

## Propulsion + LoRA

See [debug_lora_prop_config](debug_lora_prop_config.yaml) - same fields as LoRA + Propulsion sections just with a different custom_qwen for doing both and enabling both `do_lora` and `do_prop`

# Fast-Forward Explanation

Right now, Fast-Forward (FF) is supported in [fim_adapter_recipe.py](../custom_recipes/fim/fim_adapter_recipe.py) - if we want to run FF while doing vanilla fine-tuning (which the paper says doesn't work) we'd need to integrate the logic into [fim_vanilla_recipe.py](../custom_recipes/fim/fim_vanilla_recipe.py). This should be easy, it just hasn't been done because we don't plan on doing it

All of the configs in [adapter_tune/](../adapter_tune/) support fast-forward through its own config section, e.g.

```
fast_forward:
  do_ff: False # whether to do FF
  evaluate_every: 6 # How many GD steps between FF runs
  num_stabilization_steps: 50 # how many initial GD steps to take in epoch 0 before starting any FF runs
  verbose: True # whether to include debug output about the FF process

  # TODO - CHANGE THIS TO BE A 'REAL' VALIDATION SET
  # ff_collate_fn: None
  ff_dataset:
    _component_: tuning.custom_datasets.codexglue.text_completion_dataset
    source: google/code_x_glue_cc_code_completion_token
    column: code
    split: test[:64]
    verbose: False # Enable verbose logging in the dataset
    packed: True # VERY IMPORTANT - Uncommented to ensure proper packing is used
```
