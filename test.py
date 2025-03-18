from fast_trainer import BaseTrainer
from transformers import (AutoModelForSequenceClassification, 
                          AutoModelForSeq2SeqLM, 
                          AutoTokenizer,
                          TrainingArguments)
from datasets import load_dataset
from utils import FastArguments
import torch

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


if torch.cuda.is_available():
    print(f"{torch.cuda.device_count()} GPUs are available")
# Load in dataset, model, tokenizer
dataset = load_dataset("yelp_review_full")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# # tokenize and subset data.
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))
small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

# initialize the trainer.
training_args = TrainingArguments(output_dir="test_trainer")
fast_args = FastArguments(n_gpus = 1)
print(training_args.device)
trainer = BaseTrainer(model = model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset, fast_args = fast_args)
trainer.train()