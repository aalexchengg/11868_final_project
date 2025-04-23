# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified by @jacksontromero

import random  # Add import
import re
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
        add_eos (bool): Whether to add an EOS token to the end of the sequence.
            Default is True.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
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
        add_eos: bool = False,
        filter_fn: Optional[Callable] = None,
        verbose: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._column = column
        self.add_eos = add_eos
        self._source = source  # Store source for checking
        self.verbose = verbose

        self.eos_id = self._tokenizer.eos_id

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

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        try:
            return self._prepare_sample(sample, index)  # Pass index for logging
        except Exception as e:
            log.error(f"Error processing sample {index}: {e}")
            raise e

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

        # remove dataset-specific tokens, standardize <EOL> to \n
        prompt = re.sub("<s>", "", prompt)
        prompt = re.sub("</s>", "", prompt)
        prompt = re.sub("<EOL>", "\n", prompt)

        # --- Initial Tokenization (add BOS/EOS based on config/tokenizer defaults) ---
        # Let tokenizer handle BOS based on its settings. Add EOS based on self.add_eos for standard completion.
        final_tokens = self._tokenizer.encode(
            text=prompt, add_bos=True, add_eos=self.add_eos
        )

        # --- Truncation ---
        if self._tokenizer.max_seq_len is not None:
            max_len = self._tokenizer.max_seq_len
            # Truncate the potentially modified tokens
            truncated_tokens = truncate(final_tokens, max_len)
            final_tokens = truncated_tokens.copy()  # Use the truncated tokens

        final_labels = final_tokens.copy()  # Standard case, labels are copy

        if len(final_tokens) != len(final_labels):
            error_msg = f"CRITICAL: Sample {index}: Token and label length mismatch! T={len(final_tokens)}, L={len(final_labels)}. Truncating labels."
            log.error(error_msg)
            raise ValueError(error_msg)

        # <<< Add verbose log #3 >>>
        if self.verbose:
            try:
                decoded_final_tokens = self._tokenizer.decode(
                    final_tokens, skip_special_tokens=False
                )
                log.info(
                    f"Sample {index}: Final Tokens ({len(final_tokens)}):\n{decoded_final_tokens}"
                )

                # -100 is the default ignore index of CEWithChunkedOutputLoss
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

        return {"tokens": final_tokens, "labels": final_labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = False,
    packed: bool = False,  # IGNORED
    split_across_pack: bool = True,
    filter_fn: Optional[Callable] = None,
    verbose: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDataset, PackedDataset]:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~TextCompletionDataset` directly, as it is made to be config friendly.

    This builder function instantiates the modified :class:`~TextCompletionDataset` which includes
    special handling for 'google/code_x_glue_cc_code_completion_token'.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model
        source (str): path to dataset repository on Hugging Face or local file type.
        column (str): name of column containing text data. "code" for CodeXGlue. Default "text".
        add_eos (bool): Add EOS token in standard completion. Default False.
        packed (bool): Handled in recipe, not here
        split_across_pack (bool): How to handle samples crossing pack boundaries. Default True.
        filter_fn (Optional[Callable]): callable used to filter the dataset.
        verbose (bool): If True, print debugging info during sample preparation. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments for ``load_dataset``.

    Examples:
        >>> # Config for CodeXGlue
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="google/code_x_glue_cc_code_completion_token",
        ...   column="code",
        ...   packed=False,
        ...   split="train[:10%]",
        ... )

    This can also be accomplished via the yaml config::

        # Config for CodeXGlue
        dataset:
            _component_: lora_tune.custom_datasets.text_completion_dataset
            source: google/code_x_glue_cc_code_completion_token
            column: code
            packed: False
            split: train

    Returns:
        Union[TextCompletionDataset, PackedDataset]: the configured dataset.

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
    """
    # Pass all arguments, including split via kwargs, to the constructor
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        add_eos=add_eos,
        filter_fn=filter_fn,
        verbose=verbose,
        **load_dataset_kwargs,  # Includes 'split' if provided in config/call
    )

    return ds
