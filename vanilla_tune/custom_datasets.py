# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified by @jacksontromero

import random  # Add import
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune import utils  # Import torchtune utils

# Use the torchtune logger with DEBUG level
log = utils.get_logger("DEBUG")


class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Includes specific handling for 'google/code_x_glue_cc_code_completion_token'
    to load and concatenate Java and Python subsets and handle list-based code columns.

    Also includes probabilistic Fill-in-the-Middle (FIM) formatting using random
    span splitting, compatible with Qwen2.5 Models

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
            Must include <|fim_prefix|>, <|fim_suffix|>, <|fim_middle|> tokens for FIM.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        column (str): name of column in the sample that contains the text data. For CodeXGlue, this should be "code".
            Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence in standard (non-FIM) completion.
            For FIM, EOS is added after the middle section. Default is True.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        fim_prob (float): Probability (0.0 to 1.0) of applying FIM formatting to a sample. Default is 0.0 (disabled).
        min_fim_middle_percent (float): Minimum percentage of the original sequence length for the FIM middle span. Default 0.1.
        max_fim_middle_percent (float): Maximum percentage of the original sequence length for the FIM middle span. Default 0.6.
        verbose (bool): If True, print debugging info during sample preparation. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details. 'split' is handled specially for CodeXGlue.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        column: str = "text",
        add_eos: bool = True,
        filter_fn: Optional[Callable] = None,
        fim_prob: float = 0.0,  # Default to 0 (disabled)
        min_fim_middle_percent: float = 0.1,
        max_fim_middle_percent: float = 0.6,
        verbose: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._column = column
        self.add_eos = add_eos
        self._source = source  # Store source for checking
        self.fim_prob = fim_prob
        self.verbose = verbose

        # Add tracking for FIM statistics
        self.fim_applied_count = 0
        self.fim_skipped_count = 0
        self.total_samples_processed = 0
        self.samples_too_short_for_fim = 0

        if (
            min_fim_middle_percent <= 0
            or min_fim_middle_percent > max_fim_middle_percent
            or max_fim_middle_percent > 1
        ):
            raise ValueError(
                "Invalid FIM middle span percentage range. Ensure 0 < min <= max <= 1."
            )

        self.min_fim_middle_percent = min_fim_middle_percent
        self.max_fim_middle_percent = max_fim_middle_percent

        # --- FIM Token ID Initialization ---
        self.fim_prefix_id = None
        self.fim_suffix_id = None
        self.fim_middle_id = None
        self.eos_id = self._tokenizer.eos_id
        self.bos_id = self._tokenizer.bos_id

        if self.fim_prob > 0:  # Only attempt if FIM is requested
            try:
                prefix_tokens = self._tokenizer.encode(
                    "<|fim_prefix|>", add_bos=False, add_eos=False
                )
                suffix_tokens = self._tokenizer.encode(
                    "<|fim_suffix|>", add_bos=False, add_eos=False
                )
                middle_tokens = self._tokenizer.encode(
                    "<|fim_middle|>", add_bos=False, add_eos=False
                )

                # Ensure tokens exist and are single tokens
                if len(prefix_tokens) == 1:
                    self.fim_prefix_id = prefix_tokens[0]
                if len(suffix_tokens) == 1:
                    self.fim_suffix_id = suffix_tokens[0]
                if len(middle_tokens) == 1:
                    self.fim_middle_id = middle_tokens[0]

                if None in [
                    self.fim_prefix_id,
                    self.fim_suffix_id,
                    self.fim_middle_id,
                    self.eos_id,
                ]:
                    raise ValueError(
                        "One or more required FIM/EOS tokens not found or map to multiple tokens."
                    )

                # Validate token decoding
                try:
                    test_prefix_decode = self._tokenizer.decode(
                        [self.fim_prefix_id], skip_special_tokens=False
                    )
                    test_suffix_decode = self._tokenizer.decode(
                        [self.fim_suffix_id], skip_special_tokens=False
                    )
                    test_middle_decode = self._tokenizer.decode(
                        [self.fim_middle_id], skip_special_tokens=False
                    )

                    if not (
                        "<|fim_prefix|>" in test_prefix_decode
                        and "<|fim_suffix|>" in test_suffix_decode
                        and "<|fim_middle|>" in test_middle_decode
                    ):
                        log.warning(
                            f"FIM tokens don't decode correctly: {test_prefix_decode}, {test_suffix_decode}, {test_middle_decode}",
                        )
                        self.fim_prob = 0.0  # Disable FIM
                    else:
                        log.info(
                            f"FIM token decoding validated: {test_prefix_decode}, {test_suffix_decode}, {test_middle_decode}"
                        )
                except Exception as e:
                    log.warning(f"Error validating FIM token decoding: {e}")
                    self.fim_prob = 0.0  # Disable FIM

                log.info(
                    f"FIM tokens successfully initialized: Prefix={self.fim_prefix_id}, Suffix={self.fim_suffix_id}, Middle={self.fim_middle_id}, EOS={self.eos_id}"
                )

            except Exception as e:
                log.warning(
                    f"Could not initialize all FIM tokens ({e}), FIM formatting will be disabled.",
                )
                self.fim_prob = 0.0  # Disable FIM

        # Specific handling for CodeXGlue
        if source == "google/code_x_glue_cc_code_completion_token":
            split = load_dataset_kwargs.pop(
                "split", "train"
            )  # Get split, remove from kwargs passed down
            log.info(
                f"Loading and concatenating CodeXGlue Java & Python for split: {split}"
            )
            if column != "code":
                log.warning(
                    f"Using CodeXGlue source but column is '{column}'. Expected 'code'.",
                )

            try:
                # Load Java
                ds_java = load_dataset(
                    source, name="java", split=split, **load_dataset_kwargs
                )
                # Load Python (handle potential split issues like 'validation' vs 'test')
                try:
                    ds_python = load_dataset(
                        source, name="python", split=split, **load_dataset_kwargs
                    )
                except Exception as e_py:
                    if "invalid split" in str(e_py).lower() and split == "validation":
                        log.warning(
                            "Python 'validation' split not found for CodeXGlue, attempting 'test' instead.",
                        )
                        ds_python = load_dataset(
                            source, name="python", split="test", **load_dataset_kwargs
                        )
                    else:
                        log.error(
                            f"Failed to load Python subset for CodeXGlue (split={split}): {e_py}"
                        )
                        raise e_py  # Re-raise other python load errors
                self._data = concatenate_datasets([ds_java, ds_python])
                log.info(
                    f"Successfully loaded and concatenated CodeXGlue Java/Python for split '{split}'. Total size: {len(self._data)}"
                )
            except Exception as e:
                log.error(f"Failed to load CodeXGlue dataset: {e}")
                raise e
        else:
            # Original behavior for other datasets
            log.info(f"Loading dataset from source: {source}")
            self._data = load_dataset(source, **load_dataset_kwargs)

        if filter_fn is not None:
            log.info("Applying filter function to the dataset.")
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def print_fim_stats(self):
        """Print FIM application statistics"""
        if self.total_samples_processed == 0:
            log.info("No samples processed yet.")
            return

        fim_attempted_count = self.fim_applied_count + self.fim_skipped_count

        log.info("\n=== FIM Statistics ===")
        log.info(f"Total samples processed: {self.total_samples_processed}")
        if self.fim_prob > 0:
            log.info(f"FIM probability setting: {self.fim_prob:.2f}")
            log.info(
                f"FIM application rate: {self.fim_applied_count}/{self.total_samples_processed} ({self.fim_applied_count/self.total_samples_processed:.2%})"
            )
            log.info(
                f"FIM successfully applied: {self.fim_applied_count}/{fim_attempted_count} ({self.fim_applied_count/max(1,fim_attempted_count):.2%} of attempts)"
            )
            log.info(
                f"FIM skipped: {self.fim_skipped_count}/{fim_attempted_count} ({self.fim_skipped_count/max(1,fim_attempted_count):.2%} of attempts)"
            )
            log.info(f"Samples too short for FIM: {self.samples_too_short_for_fim}")
        else:
            log.info("FIM disabled (fim_prob=0)")
        log.info("=====================")

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        try:
            self.total_samples_processed += 1
            return self._prepare_sample(sample, index)  # Pass index for logging
        except Exception as e:
            log.error(f"Error processing sample {index}: {e}")
            raise e

    def _transform_to_fim(
        self, tokens: List[int], index: int
    ) -> Optional[Dict[str, List[int]]]:
        """Attempts to transform a token sequence into FIM format."""
        # Check if FIM is possible (tokens initialized)
        if None in [
            self.fim_prefix_id,
            self.fim_suffix_id,
            self.fim_middle_id,
            self.eos_id,
        ]:
            return None  # FIM tokens not available

        try:
            has_bos = self.bos_id is not None and tokens and tokens[0] == self.bos_id
            # Treat original EOS as part of the suffix if present
            has_eos = self.eos_id is not None and tokens and tokens[-1] == self.eos_id

            # Content length excludes BOS/EOS for span calculation
            content_start_offset = 1 if has_bos else 0
            content_end_offset = 1 if has_eos else 0
            content_len = len(tokens) - content_start_offset - content_end_offset

            # Need at least 1 token for prefix, 1 for middle, 1 for suffix
            if content_len < 3:
                self.samples_too_short_for_fim += 1
                log.warning(
                    f"Sample {index}: Content length too short for FIM. Skipping FIM transformation.",
                )
                return None  # Too short to split meaningfully

            # Calculate middle span length based on percentages of content_len
            min_middle = max(1, int(content_len * self.min_fim_middle_percent))
            max_middle = max(min_middle, int(content_len * self.max_fim_middle_percent))
            # Ensure max_middle leaves at least 1 token for prefix and 1 for suffix
            max_middle = min(max_middle, content_len - 2)

            if max_middle < min_middle:
                return None  # Invalid span range

            middle_len = random.randint(min_middle, max_middle)

            # Choose start index for middle span within the content part
            # Max start index ensures middle_len fits and leaves >=1 for suffix
            max_content_start_index = content_len - middle_len - 1
            content_start_index = random.randint(
                0, max_content_start_index
            )  # Relative to content start

            # Convert content indices back to original token list indices
            start_index = content_start_offset + content_start_index
            end_index = start_index + middle_len  # end_index is exclusive

            # Extract parts from original tokens list
            prefix_tokens = tokens[content_start_offset:start_index]
            middle_tokens = tokens[start_index:end_index]
            suffix_tokens = tokens[
                end_index : len(tokens) - content_end_offset
            ]  # Exclude original EOS

            # Construct FIM input and target sequences
            # Input: [<bos>] <prefix> PREFIX <suffix> SUFFIX <middle>
            fim_input_no_bos = (
                [self.fim_prefix_id]
                + prefix_tokens
                + [self.fim_suffix_id]
                + suffix_tokens
                + [self.fim_middle_id]
            )
            fim_input = (
                ([self.bos_id] + fim_input_no_bos) if has_bos else fim_input_no_bos
            )

            # Target: MIDDLE <eos>
            fim_target = middle_tokens + [self.eos_id]

            # Final sequence for model input (PSM format for generation)
            final_tokens = fim_input + fim_target
            # Labels for loss calculation (Mask input, keep target)
            final_labels = [-100] * len(fim_input) + fim_target

            return {"tokens": final_tokens, "labels": final_labels}

        except Exception as e:
            log.warning(f"Error during FIM transformation logic: {e}")
            return None  # Fallback on any error within transformation

    def _prepare_sample(
        self, sample: Mapping[str, Any], index: int
    ) -> Dict[str, List[int]]:
        # Handle list vs string in the target column (specifically for CodeXGlue 'code')
        text_input = sample[self._column]
        if isinstance(text_input, list):
            prompt = " ".join(text_input)
        elif isinstance(text_input, str):
            prompt = text_input
        else:
            log.warning(
                f"Unexpected type {type(text_input)} for column '{self._column}' at index {index}. Attempting str conversion.",
            )
            prompt = str(text_input)

        # <<< Add verbose log #1 >>>
        if self.verbose:
            log.info(f"\n--- Sample {index} ---")
            log.info(f"Original Prompt (joined):\n{prompt}...")

        # --- Initial Tokenization (add BOS/EOS based on config/tokenizer defaults) ---
        # Let tokenizer handle BOS based on its settings. Add EOS based on self.add_eos for standard completion.
        # FIM transform will handle EOS specifically if applied.
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        # --- Attempt FIM Transformation ---
        final_tokens = tokens
        labels_or_none = None  # Use None to signal if FIM labels were generated
        is_fim_sample = False

        if self.fim_prob > 0 and random.random() < self.fim_prob:
            fim_result = self._transform_to_fim(tokens, index)  # Pass original tokens
            if fim_result is not None:
                final_tokens = fim_result["tokens"]
                labels_or_none = fim_result["labels"]  # Store FIM labels
                is_fim_sample = True
                self.fim_applied_count += 1
            else:
                log.debug(
                    f"Sample {index}: FIM skipped (condition not met or transform failed)."
                )
                self.fim_skipped_count += 1

        # <<< Add verbose log #2 >>>
        if self.verbose:
            log.info(f"Sample {index}: FIM Applied: {is_fim_sample}")

        # --- Truncation ---
        final_labels = None  # Initialize final_labels
        if self._tokenizer.max_seq_len is not None:
            max_len = self._tokenizer.max_seq_len
            # Truncate the potentially modified tokens
            truncated_tokens = truncate(final_tokens, max_len)

            # Handle labels based on whether FIM was applied
            if is_fim_sample:
                # Truncate FIM labels to match truncated tokens
                final_labels = truncate(labels_or_none, max_len)
            else:
                # Standard case: Create labels AFTER truncation by copying
                final_labels = truncated_tokens.copy()

            final_tokens = truncated_tokens  # Use the truncated tokens
        else:
            # No truncation needed
            if is_fim_sample:
                final_labels = labels_or_none
            else:
                final_labels = final_tokens.copy()  # Standard case, labels are copy

        # --- Final Checks ---
        if (
            final_labels is None
        ):  # Should only happen if truncation disabled and FIM failed unexpectedly
            log.error(
                f"Sample {index}: final_labels is None unexpectedly. Defaulting to copy."
            )
            final_labels = final_tokens.copy()

        if len(final_tokens) != len(final_labels):
            log.error(
                f"CRITICAL: Sample {index}: Token and label length mismatch! T={len(final_tokens)}, L={len(final_labels)}. Truncating labels."
            )
            final_labels = final_labels[: len(final_tokens)]
            if len(final_labels) < len(final_tokens):
                final_labels += [-100] * (len(final_tokens) - len(final_labels))

        if not final_tokens:
            error_msg = f"Sample {index} resulted in empty tokens after processing."
            log.error(error_msg)
            raise ValueError(error_msg)  # Explicitly raise an exception

        # <<< Add verbose log #3 >>>
        if self.verbose:
            try:
                decoded_final_tokens = self._tokenizer.decode(
                    final_tokens, skip_special_tokens=False
                )
                log.info(
                    f"Sample {index}: Final Tokens ({len(final_tokens)}):\n{decoded_final_tokens}"
                )

                target_tokens_from_labels = [
                    t for t, l in zip(final_tokens, final_labels) if l != -100
                ]
                if target_tokens_from_labels:
                    decoded_target = self._tokenizer.decode(
                        target_tokens_from_labels, skip_special_tokens=False
                    )
                    log.info(
                        f"Sample {index}: Decoded Target ({len(target_tokens_from_labels)}):\n{decoded_target}"
                    )
                else:
                    log.info(f"Sample {index}: No target tokens found in labels.")
            except Exception as e:
                log.error(f"Sample {index}: Error during verbose decoding: {e}")
            log.info(f"--- End Sample {index} ---")

        # Print FIM stats for the last sample
        if index == len(self._data) - 1:
            self.print_fim_stats()

        return {"tokens": final_tokens, "labels": final_labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    filter_fn: Optional[Callable] = None,
    fim_prob: float = 0.0,  # Default FIM disabled
    min_fim_middle_percent: float = 0.1,
    max_fim_middle_percent: float = 0.6,
    verbose: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDataset, PackedDataset]:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~TextCompletionDataset` directly, as it is made to be config friendly.

    This builder function instantiates the modified :class:`~TextCompletionDataset` which includes
    special handling for 'google/code_x_glue_cc_code_completion_token' and probabilistic
    Fill-in-the-Middle (FIM) formatting using random span splitting.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model. Needs FIM tokens for FIM option.
        source (str): path to dataset repository on Hugging Face or local file type.
        column (str): name of column containing text data. "code" for CodeXGlue. Default "text".
        add_eos (bool): Add EOS token in standard completion (ignored if FIM applied). Default True.
        packed (bool): Pack dataset to ``max_seq_len``. Default False.
        split_across_pack (bool): How to handle samples crossing pack boundaries. Default True.
        filter_fn (Optional[Callable]): callable used to filter the dataset.
        fim_prob (float): Probability (0.0 to 1.0) of applying FIM formatting. Default 0.0 (disabled).
        min_fim_middle_percent (float): Min percentage length for FIM middle span (relative to content). Default 0.1.
        max_fim_middle_percent (float): Max percentage length for FIM middle span (relative to content). Default 0.6.
        verbose (bool): If True, print debugging info during sample preparation. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments for ``load_dataset``.

    Examples:
        >>> # Config for CodeXGlue with FIM enabled (50% probability)
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="google/code_x_glue_cc_code_completion_token",
        ...   column="code",
        ...   packed=False,
        ...   fim_prob=0.5, # Enable FIM
        ...   split="train[:10%]",
        ... )

    This can also be accomplished via the yaml config::

        # Config for CodeXGlue with FIM enabled
        dataset:
            _component_: lora_tune.custom_datasets.text_completion_dataset
            source: google/code_x_glue_cc_code_completion_token
            column: code
            packed: False
            split: train
            fim_prob: 0.5 # Apply FIM to 50% of samples
            # Optional: Adjust middle span percentage range
            # min_fim_middle_percent: 0.05
            # max_fim_middle_percent: 0.5

    Returns:
        Union[TextCompletionDataset, PackedDataset]: the configured dataset.

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
        Warning: If FIM tokens (<|fim_prefix|>, etc.) are missing or invalid in tokenizer when fim_prob > 0 (FIM will be disabled).
    """
    # Pass all arguments, including FIM params and split via kwargs, to the constructor
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        add_eos=add_eos,
        filter_fn=filter_fn,
        fim_prob=fim_prob,
        min_fim_middle_percent=min_fim_middle_percent,
        max_fim_middle_percent=max_fim_middle_percent,
        verbose=verbose,
        **load_dataset_kwargs,  # Includes 'split' if provided in config/call
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        # Ensure max_seq_len used for packing matches tokenizer
        max_len = tokenizer.max_seq_len
        return PackedDataset(
            ds, max_seq_len=max_len, split_across_pack=split_across_pack
        )
    return ds
