import argparse
import logging
from fast_trainer import BaseTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import datasets
from utils import tokenize_codexglue, FastArguments
from functools import partial


def main(args):
    # get model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    # get dataset
    java_ds = load_dataset("google/code_x_glue_cc_code_completion_token", "java")
    python_ds = load_dataset("google/code_x_glue_cc_code_completion_token", "python")
    # combine 
    combine_train = datasets.concatenate_datasets([java_ds['train'], python_ds['train']])
    combine_eval = datasets.concatenate_datasets([java_ds['validation'], python_ds['test']])
    # should we shuffle these?
    combine_train = combine_train.shuffle(seed = 42).select(range(5000))
    combine_eval = combine_eval.shuffle(seed = 42).select(range(1000))
    # do a tokenize
    tokenize_function = partial(tokenize_codexglue, tokenizer)
    combine_train = combine_train.map(tokenize_function, batched=True)
    combine_eval = combine_eval.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="test_trainer",
    )
    fast_args = FastArguments(n_gpus = 1, device = 0)
    trainer = BaseTrainer(model=model, args=training_args, train_dataset=combine_train, eval_dataset=combine_eval, fast_args  = fast_args)

    train_dataloader = trainer.get_train_dataloader()
    trainer.train()
    trainer.push_to_hub()
    

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', type = int,
                        help = "batch size",
                        default = 16)
    parser.add_argument('-out', type = str,
                        help = "output filepath.")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    main(args)