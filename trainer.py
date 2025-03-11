import torch
from torch import nn
from transformers import PreTrainedModel, TrainingArguments
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
from torch.utils.data import DataLoader, Dataset, IterableDataset
import datasets
import numpy as np
import argparse
import logging
import inspect
import time
from tqdm import tqdm
import random
from typing import Union, Optional, Dict, Tuple, Any, Callable, Type
from utils import (RemoveColumnsCollator)

logger = logging.getLogger(__name__)



def worker_init_fn(worker_id):
    """
    Sets seed for worker collator fn. 
    https://discuss.pytorch.org/t/how-to-fix-all-workers-seed-via-worker-init-fn-for-every-iter/127006
    """                                                                                                                                
    seed = 0                                                                                                                                           
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return


class Trainer:
    """
    Base Trainer class for eventual abstraction.
    Code was adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py.
    """
    def __init__(self, 
                model: Optional[Union[PreTrainedModel, nn.Module]] = None, 
                processing_class: Optional[Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
                ] = None,
                args: Optional[TrainingArguments] = None,
                data_collator: Optional[DataCollator] = None,
                train_dataset: Optional[Union["datasets.Dataset", IterableDataset, Dataset]] = None,
                eval_dataset: Optional[Union["datasets.Dataset", Dataset, Dict[str, Dataset]]] = None,
                compute_loss_func: Optional[Callable] = None,
                compute_metrics: Optional[Callable] = None,
                optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
                optimizer_cls_and_kwargs: Optional[Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]] = None,
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        """
        Initializes the Trainer class
        """
        self.model = model
        self.args = args
        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.loss_fn = compute_loss_func
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers 

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model

            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = worker_init_fn
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(train_dataset, **dataloader_params)
    
    def get_eval_dataloader(self) -> DataLoader:
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return eval_dataloader

    def train(self):
        """
        Runs the training loop.
        """
        pass
    def evaluate(self):
        """
        Runs the evaluation loop.
        """
        pass


    

