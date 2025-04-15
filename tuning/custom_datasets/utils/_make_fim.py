# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified by @jacksontromero

import random  # Add import
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset, concatenate_datasets
from torch import Tensor
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.datasets import TextCompletionDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune import utils  # Import torchtune utils

# Use the torchtune logger with DEBUG level
log = utils.get_logger("DEBUG")


class FimDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Includes specific handling for 'google/code_x_glue_cc_code_completion_token'
    to load and concatenate Java and Python subsets and handle list-based code columns.

    Also includes probabilistic Fill-in-the-Middle (FIM) formatting using random
    span splitting, compatible with Qwen2.5 Models

    Args:
        source_ds (TextCompletionDataset): The source dataset to apply FIM to.
        fim_prob (float): Probability (0.0 to 1.0) of applying FIM formatting to a sample. Default is 0.0 (disabled).
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        min_fim_middle_percent (float): Minimum percentage of the original sequence length for the FIM middle span. Default 0.1.
        fim_prob (float): Probability (0.0 to 1.0) of applying FIM formatting to a sample. Default is 0.0 (disabled).
        min_fim_middle_percent (float): Minimum percentage of the original sequence length for the FIM middle span. Default 0.1.
        max_fim_middle_percent (float): Maximum percentage of the original sequence length for the FIM middle span. Default 0.6.
        verbose (bool): If True, print debugging info during sample preparation. Default is False.
    """

    def __init__(
        self,
        source_ds: TextCompletionDataset,
        fim_prob: float = 0.0,  # Default to 0 (disabled)
        min_fim_middle_percent: float = 0.05,
        max_fim_middle_percent: float = 0.1,
        verbose: bool = False,
    ) -> None:
        self.source_ds = source_ds
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
        self.eos_id = None
        self.bos_id = None

        if self.fim_prob > 0:  # Only attempt if FIM is requested
            prefix_tokens = self.source_ds._tokenizer.encode(
                "<|fim_prefix|>", add_bos=False, add_eos=False
            )
            suffix_tokens = self.source_ds._tokenizer.encode(
                "<|fim_suffix|>", add_bos=False, add_eos=False
            )
            middle_tokens = self.source_ds._tokenizer.encode(
                "<|fim_middle|>", add_bos=False, add_eos=False
            )
            eos_tokens = self.source_ds._tokenizer.encode(
                "<|endoftext|>", add_bos=False, add_eos=False
            )

            # Ensure tokens exist and are single tokens
            if len(prefix_tokens) == 1:
                self.fim_prefix_id = prefix_tokens[0]
            if len(suffix_tokens) == 1:
                self.fim_suffix_id = suffix_tokens[0]
            if len(middle_tokens) == 1:
                self.fim_middle_id = middle_tokens[0]
            if len(eos_tokens) == 1:
                self.eos_id = eos_tokens[0]

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
            test_prefix_decode = self.source_ds._tokenizer.decode(
                [self.fim_prefix_id], skip_special_tokens=False
            )
            test_suffix_decode = self.source_ds._tokenizer.decode(
                [self.fim_suffix_id], skip_special_tokens=False
            )
            test_middle_decode = self.source_ds._tokenizer.decode(
                [self.fim_middle_id], skip_special_tokens=False
            )
            test_eos_decode = self.source_ds._tokenizer.decode(
                [self.eos_id], skip_special_tokens=False
            )

            if not (
                "<|fim_prefix|>" in test_prefix_decode
                and "<|fim_suffix|>" in test_suffix_decode
                and "<|fim_middle|>" in test_middle_decode
                and "<|endoftext|>" in test_eos_decode
            ):
                log.error(
                    f"FIM tokens don't decode correctly: {test_prefix_decode}, {test_suffix_decode}, {test_middle_decode}, {test_eos_decode}",
                )
                raise ValueError(
                    f"FIM tokens don't decode correctly: {test_prefix_decode}, {test_suffix_decode}, {test_middle_decode}"
                )

            else:
                log.info(
                    f"FIM token decoding validated: {test_prefix_decode}, {test_suffix_decode}, {test_middle_decode}"
                )

    def __len__(self):
        return len(self.source_ds)

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
        sample = self.source_ds[index]
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
            error_msg = "FIM tokens not available in _transform_to_fim."
            log.error(error_msg)
            raise ValueError(error_msg)

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
            min_middle_len = max(1, int(content_len * self.min_fim_middle_percent))
            max_middle_len = max(
                min_middle_len, int(content_len * self.max_fim_middle_percent)
            )
            # Ensure max_middle leaves at least 1 token for prefix and 1 for suffix
            max_middle_len = min(max_middle_len, content_len - 2)

            if max_middle_len < min_middle_len:
                return None  # Invalid span range

            middle_len = random.randint(min_middle_len, max_middle_len)

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
            # Input: <prefix> PREFIX <suffix> SUFFIX <middle>
            fim_input = (
                [self.fim_prefix_id]
                + prefix_tokens
                + [self.fim_suffix_id]
                + suffix_tokens
                + [self.fim_middle_id]
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
        self, sample: Dict[str, List[int]] | Dict[str, Tensor], index: int
    ) -> Dict[str, List[int]]:

        # --- Attempt FIM Transformation ---
        final_tokens = sample["tokens"]
        final_labels = None

        if self.fim_prob > 0 and random.random() < self.fim_prob:
            fim_result = self._transform_to_fim(final_tokens, index)  # Pass original
            if fim_result is not None:
                final_tokens = fim_result["tokens"]
                final_labels = fim_result["labels"]
                self.fim_applied_count += 1
            else:
                log.debug(
                    f"Sample {index}: FIM skipped (condition not met or transform failed)."
                )
                self.fim_skipped_count += 1
        else:
            final_labels = sample[
                "tokens"
            ]  # No FIM applied, use original tokens as labels

        # if there was an error in the FIM transformation, use the original tokens as labels
        if not final_labels:
            final_labels = sample["tokens"]

        # --- Final Checks ---

        if len(final_tokens) != len(final_labels):
            error_msg = f"CRITICAL: Sample {index}: Token and label length mismatch! T={len(final_tokens)}, L={len(final_labels)}. Truncating labels."
            log.error(error_msg)
            raise ValueError(error_msg)

        if not final_tokens:
            error_msg = f"Sample {index} resulted in empty tokens after processing."
            log.error(error_msg)
            raise ValueError(error_msg)

        if not final_labels:
            error_msg = f"Sample {index} resulted in empty labels after processing."
            log.error(error_msg)
            raise ValueError(error_msg)

        # <<< Add verbose log #3 >>>
        if self.verbose:
            try:
                decoded_final_tokens = self.source_ds._tokenizer.decode(
                    final_tokens, skip_special_tokens=False
                )
                log.info(
                    f"Sample {index}: Final Tokens ({len(final_tokens)}):\n{decoded_final_tokens}"
                )

                target_tokens_from_labels = [
                    t for t, l in zip(final_tokens, final_labels) if l != -100
                ]
                if target_tokens_from_labels:
                    decoded_target = self.source_ds._tokenizer.decode(
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

        # Print FIM stats for the last-ish sample
        if index == len(self.source_ds) - 10:
            self.print_fim_stats()

        return {"tokens": final_tokens, "labels": final_labels}


def fim_dataset(
    source_ds: TextCompletionDataset,
    fim_prob: float = 0.0,  # Default FIM disabled
    min_fim_middle_percent: float = 0.05,
    max_fim_middle_percent: float = 0.1,
    verbose: bool = False,
) -> FimDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~TextCompletionDataset` directly, as it is made to be config friendly.

    This builder function takes a source dataset and applies probabilistic
    Fill-in-the-Middle (FIM) formatting using random span splitting.

    Args:
        source_ds: The source dataset to apply FIM to.
        fim_prob: The probability of applying FIM to a sample.
        min_fim_middle_percent: The minimum percentage of the original sequence length for the FIM middle span.
        max_fim_middle_percent: The maximum percentage of the original sequence length for the FIM middle span.
        verbose: If True, print debugging info during sample preparation.

    Examples:
        >>> # Config for dataset with FIM enabled (50% probability)
        >>> dataset = fim_dataset(
        ...   source_ds=source_ds,
        ...   fim_prob=0.5, # Enable FIM
        ...   min_fim_middle_percent=0.05,
        ...   max_fim_middle_percent=0.1,
        ...   verbose=True,
        ... )

    This can also be accomplished via the yaml config (see fim_config.example.yaml for more details)

    Returns:
        TextCompletionDataset: the configured dataset.

    Raises:
        Warning: If FIM tokens (<|fim_prefix|>, etc.) are missing or invalid in tokenizer when fim_prob > 0 (FIM will be disabled).
    """
    # Pass all arguments, including FIM params and split via kwargs, to the constructor
    ds = FimDataset(
        source_ds=source_ds,
        fim_prob=fim_prob,
        min_fim_middle_percent=min_fim_middle_percent,
        max_fim_middle_percent=max_fim_middle_percent,
        verbose=verbose,
    )
    return ds
