from datasets import load_dataset
import datasets
import argparse
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# fim imports
from typing import Optional, Dict, List
import random
import logging
import evaluate

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
    parser.add_argument("-min_fim", type=float, default=0.1)
    parser.add_argument("-max_fim", type=float, default=0.3)
    parser.add_argument("-p", type=float, default=0.6)
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
    ds = ds.select(list(range(100)))
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


def top_k(eval_preds):
    print(eval_preds.shape)
    eval_predictions = np.argmax(eval_preds.predictions, axis=2)
    raise AssertionError("stop here")


def main(args):
    # step zero: load in model, tokenizer, dataset
    if args.model == "google-bert/bert-base-uncased":
        model = AutoModelForCausalLM.from_pretrained(args.model, is_decoder=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # add special tokens
    n = len(tokenizer)
    tokens_added = ["[FIM_START]", "[FIM_MID]", "[FIM_END]", "[BOS]", "[EOS]"]
    num_added_toks = tokenizer.add_tokens(tokens_added, special_tokens=True)
    assert num_added_toks == 5
    token_mapping = {token: n + i for i, token in enumerate(tokens_added)}
    model.resize_token_embeddings(len(tokenizer))

    dataset = get_dataset(args, tokenizer, token_mapping)
    # step one: train the small model with our dataset
    training_args = TrainingArguments(
        output_dir="yelp_review_classifier",
        push_to_hub=False,
        eval_strategy="no",
        per_device_eval_batch_size=32,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # step two: run predictions and get the probability of the tokens
    print("Getting predictions")
    trainer_preds = trainer.predict(dataset)
    # step three: sort by probability and only take the top p% of tokens
    # step four: push the new dataset to the hub

    print("All finished.")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
