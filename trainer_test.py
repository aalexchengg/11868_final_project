from base_trainer import Trainer
from transformers import (AutoModelForSequenceClassification, 
                          AutoModelForSeq2SeqLM, 
                          AutoTokenizer,
                          TrainingArguments)
from datasets import load_dataset


def test_dataload_1():
    """
    Tests dataloading with sample case taken from https://huggingface.co/docs/transformers/en/training
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Load in dataset, model, tokenizer
    dataset = load_dataset("yelp_review_full")
    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    # tokenize and subset data.
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))
    small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

    # initialize the trainer.
    training_args = TrainingArguments(output_dir="test_trainer")
    trainer = Trainer(model = model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

    # ensure that we can load the data.
    train_dataloader = trainer.get_train_dataloader()
    assert(len(next(iter(train_dataloader))['input_ids']) == training_args.train_batch_size)
    test_dataloader = trainer.get_eval_dataloader(small_eval_dataset)
    assert(len(next(iter(test_dataloader))['input_ids']) == training_args.eval_batch_size)


def test_dataload_2():
    """
    Tests dataloading with code LLM and dataset.
    """
    def tokenize_function(examples):
        return tokenizer(examples["content"], padding="max_length", truncation=True)

    # Load in dataset, model, tokenizer
    dataset = load_dataset("bigcode/the-stack-smol")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

    # tokenize and subset data.
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
    small_eval_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

    # initialize the trainer.
    training_args = TrainingArguments(output_dir="test_trainer")
    trainer = Trainer(model = model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

    # ensure that we can load the data.
    train_dataloader = trainer.get_train_dataloader()
    assert(len(next(iter(train_dataloader))['input_ids']) == training_args.train_batch_size)
    test_dataloader = trainer.get_eval_dataloader(small_eval_dataset)
    assert(len(next(iter(test_dataloader))['input_ids']) == training_args.eval_batch_size)


if __name__ == "__main__":
    test_dataload_1()
    test_dataload_2()
