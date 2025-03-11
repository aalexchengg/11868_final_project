from transformers import PreTrainedModel, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from typing import Union, Optional, Dict, Tuple, Any, Callable, Type
from base_trainer import Trainer
from utils import FastArguments

class FastTrainer(Trainer):

    def __init__(self, 
                model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                model_name: Optional[str] = None, 
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
        super().__init__(model,
                         model_name,
                         processing_class,
                         args,
                         data_collator,
                         train_dataset,
                         eval_dataset,
                         compute_loss_func,
                         compute_metrics,
                         optimizers,
                         optimizer_cls_and_kwargs,
                         preprocess_logits_for_metrics)
        self.fast_args = fast_args
    
    def _training_loop(self):
        # TODO: write a new cool optimized one to overwrite.
        pass
        