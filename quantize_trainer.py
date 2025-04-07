from fast_trainer import BaseTrainer
from typing import Union, Optional, Dict, Tuple, Any, Callable, Type, List
from transformers import PreTrainedModel, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.processing_utils import ProcessorMixin
import torch
from torch import nn
from utils import FastArguments

from torchao.sparsity.training import (
    SemiSparseLinear,
    SemiSparseActivationLinear,
    swap_linear_with_semi_sparse_linear,
    swap_semi_sparse_linear_with_linear,
)

class QuantizeTrainer(BaseTrainer):
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
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                fast_args: Optional[FastArguments] = None):
        # Initialize base class
        self.sparse_config = {"seq.0": SemiSparseLinear,
        # for activation sparsity, uncomment the below line
        # "seq.0": SemiSparseActivationLinear,
        }
        super().__init__(model = model,
                         processing_class = processing_class,
                         args = args,
                         data_collator = data_collator,
                         train_dataset = train_dataset,
                         eval_dataset = eval_dataset,
                         compute_loss_func = compute_loss_func,
                         compute_metrics = compute_metrics,
                         optimizers =optimizers,
                         optimizer_cls_and_kwargs = optimizer_cls_and_kwargs,
                         preprocess_logits_for_metrics = preprocess_logits_for_metrics)
        self.fast_args = fast_args
    
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,):
        # Swap nn.Linear with SemiSparseLinear
        swap_linear_with_semi_sparse_linear(self.model, self.sparse_config)
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        swap_semi_sparse_linear_with_linear(self.model)

