from datasets import load_dataset
import datasets
import argparse
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# fim imports
from typing import Optional, Dict, List
import random
import logging
import evaluate
import numpy as np
import time
import torch

logger = logging.getLogger(__name__)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Model to do dataset cartography with",
    )
    parser.add_argument("-dataset", type=str, default="Nan-Do/code-search-net-python")
    parser.add_argument("-min_fim", type=float, default=0.01)
    parser.add_argument("-max_fim", type=float, default=0.1)
    parser.add_argument("-p", type=float, default=0.4)
    parser.add_argument("-token", type = str)
    parser.add_argument("-username", type = str, default = "aalexchengg")
    parser.add_argument("-size", type = int, default = -1)
    return parser


def fim_wrapper(args, token_mapping):
    def transform_to_fim(example) -> Optional[Dict[str, List[int]]]:
        """Attempts to transform a token sequence into FIM format."""
        tokens = example["input_ids"]
        try:
            # Content length excludes BOS/EOS for span calculation
            content_start_offset = 1
            content_end_offset = 1
            content_len = len(tokens) - content_start_offset - content_end_offset

            # # Need at least 1 token for prefix, 1 for middle, 1 for suffix
            # if content_len < 3:
            #     logging.warning(
            #         f"Sample {index}: Content length too short for FIM. Skipping FIM transformation.",
            #     )
            #     example["labels"] = []  # Too short to split meaningfully
            #     return example

            # Calculate middle span length based on percentages of content_len
            min_middle_len = max(1, int(content_len * args.min_fim))
            max_middle_len = max(min_middle_len, int(content_len * args.max_fim))
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
                [token_mapping["[FIM_START]"]]
                + prefix_tokens
                + [token_mapping["[FIM_END]"]]
                + suffix_tokens
                + [token_mapping["[FIM_MID]"]]
            )

            # Target: MIDDLE <eos>
            fim_target = middle_tokens + [token_mapping["[EOS]"]]

            # Final sequence for model input (PSM format for generation)
            final_tokens = fim_input + fim_target
            # Labels for loss calculation (Mask input, keep target)
            final_labels = [-100] * len(fim_input) + fim_target

            return {
                "input_ids": final_tokens,
                "labels": final_labels,
                "token_type_ids": [0] * len(final_tokens),
                "attention_mask": [1] * len(final_tokens),
            }

        except Exception as e:
            logging.warning(f"Error during FIM transformation logic: {e}")
            return None  # Fallback on any error within transformation

    return transform_to_fim


def tokenize_and_align(tokenizer):
    def inner_lambda(examples):
        return tokenizer(
            examples["code"], padding="max_length", max_length=500, truncation=True
        )

    return inner_lambda


def get_dataset(args, tokenizer, token_mapping):
    # load from huggingface
    ds = load_dataset(args.dataset, split="train", streaming=False)
    if args.size != -1:
        ds = ds.select(list(range(args.size)))
    # first we need to tokenize
    ds = ds.map(tokenize_and_align(tokenizer), batched=True)
    # then we need to generate labels with fim
    ds = ds.map(fim_wrapper(args, token_mapping))
    return ds


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def top_k(probabilities, p):
    """
    takes in the predictions and labels, as well as the percent of samples we are taking
    and return p% of those indices of those with lowest probability and highest variance
    """
    # predictions should have shape dataset size x sequence length x vocab length
    # labels should have shape dataset size x sequence length
    k = int(probabilities.shape[0] * p)
    averages = np.mean(probabilities, axis=1)
    averages = np.argsort(averages)[:k]  # only take top k
    variances = np.std(probabilities, axis=1)
    variances = np.argsort(variances)[::-1][:k]  # reverse, and then only take top k
    return averages, variances


def main(args):
    # step zero: load in model, tokenizer, dataset
    if args.model == "google-bert/bert-base-uncased":
        model = AutoModelForCausalLM.from_pretrained(args.model, is_decoder=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_pretrained("ref_model/checkpoint-1250")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # add special tokens
    n = len(tokenizer)
    tokens_added = ["[FIM_START]", "[FIM_MID]", "[FIM_END]", "[BOS]", "[EOS]"]
    # tokenizer.pad_token = tokenizer.eos_token
    num_added_toks = tokenizer.add_tokens(tokens_added, special_tokens=True)
    assert num_added_toks == 5
    token_mapping = {token: n + i for i, token in enumerate(tokens_added)}
    model.resize_token_embeddings(len(tokenizer))

    dataset = get_dataset(args, tokenizer, token_mapping)
    # step one: train the small model with our dataset
    training_args = TrainingArguments(
        output_dir="ref_model",
        push_to_hub=False,
        save_strategy = "no",
        eval_strategy="no",
        per_device_train_batch_size = 16,
        eval_accumulation_steps = 1,
        fp16 = True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    print("beginning train.")
    trainer.train()
    # step two: run predictions and get the probability of the tokens
    trainer._train_batch_size = 8 # number of GPUs.
    dataloader = trainer.get_train_dataloader()
    model.eval()
    vals = []
    print(len(dataloader))
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if (i % 100 == 0):
            print(f"on batch {i}")
        with torch.no_grad():
            outputs = model(**batch)
            labels = np.expand_dims(batch['labels'].cpu(), axis=2)
            probabilities = np.take_along_axis(outputs['logits'].cpu(), labels, axis=2).squeeze() 
            vals.append(probabilities)
            torch.cuda.empty_cache()
    vals = np.concatenate(vals)
    end_time = time.time()
    print(f"predictions for {len(dataset)} took {end_time - start_time} seconds")
    # step three: sort by probability and only take the top p% of tokens
    average_idx, variance_idx = top_k(vals, args.p)
    # step four: push the new dataset to the hub
    dataset_name = args.dataset.split("/")[1]  # get back half
    average_subset = dataset.select(average_idx)
    average_subset.push_to_hub(f"{args.username}/{dataset_name}_avg_subset", token = args.token)

    variance_subset = dataset.select(variance_idx)
    variance_subset.push_to_hub(f"{args.username}/{dataset_name}_variance_subset", token = args.token)
    # step four: push the new dataset to the hub

    print("All finished.")


if __name__ == "__main__":
    set_seed(15122)
    parser = setup_parser()
    args = parser.parse_args()
    logger.setLevel(logging.INFO)
    main(args)