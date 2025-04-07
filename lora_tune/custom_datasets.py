# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified by @jacksontromero

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

log = logging.getLogger(__name__)


class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Includes specific handling for 'google/code_x_glue_cc_code_completion_token'
    to load and concatenate Java and Python subsets and handle list-based code columns.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        column (str): name of column in the sample that contains the text data. For CodeXGlue, this should be "code".
            Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
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
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._column = column
        self.add_eos = add_eos
        self._source = source  # Store source for checking

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
                    f"Using CodeXGlue source but column is '{column}'. Expected 'code'."
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
                            "Python 'validation' split not found for CodeXGlue, attempting 'test' instead."
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
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        # Handle list vs string in the target column (specifically for CodeXGlue 'code')
        text_input = sample[self._column]
        if isinstance(text_input, list):
            prompt = " ".join(text_input)
        elif isinstance(text_input, str):
            prompt = text_input
        else:
            # Handle unexpected type: attempt string conversion with warning
            log.warning(
                f"Unexpected type for column '{self._column}' at index ??: {type(text_input)}. Attempting str conversion."
            )
            prompt = str(text_input)

        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            # Account for the possibility of BOS token being added by encode
            # Truncate considers the sequence length including BOS/EOS if added by tokenizer
            max_len = self._tokenizer.max_seq_len
            # Simple truncation: tokens = tokens[:max_len]
            # Using torchtune utility which might handle edge cases better:
            tokens = truncate(tokens, max_len)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    # `split` argument is now primarily handled within __init__ for CodeXGlue
    # It's still passed via **load_dataset_kwargs for other datasets or if needed by __init__
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDataset, PackedDataset]:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextCompletionDataset` directly, as it is made to be config friendly.

    This builder function instantiates the modified :class:`~TextCompletionDataset` which includes
    special handling for 'google/code_x_glue_cc_code_completion_token'.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face or local file type.
        column (str): name of column in the sample that contains the text data. Should be "code" for CodeXGlue.
            Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. This argument is ignored if ``packed=False``. Default is True.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.
            The 'split' argument here will be used by the TextCompletionDataset constructor, potentially
            overriding the default 'train' if CodeXGlue is used, or passed directly for other datasets.

    Examples:
        >>> # Example for CodeXGlue (assuming tokenizer is defined)
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="google/code_x_glue_cc_code_completion_token",
        ...   column="code",
        ...   packed=False,
        ...   split="train[:10%]", # Passed via kwargs
        ... )

    This can also be accomplished via the yaml config::

        # Config for CodeXGlue
        dataset:
            _component_: lora_tune.custom_datasets.text_completion_dataset # Point to this modified func
            source: google/code_x_glue_cc_code_completion_token
            column: code
            packed: False
            split: train # This will be used for both Java and Python loading

    Returns:
        Union[TextCompletionDataset, PackedDataset]: the configured :class:`~TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

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
