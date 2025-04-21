from datasets import load_dataset, Dataset
import datasets
import argparse
from collections import defaultdict
import logging
import requests

logger = logging.getLogger(__name__)
github_repo = "codeparrot/github-code"


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type = int,
                        default = 1000,
                        help ='size of resulting dataset')
    parser.add_argument('-seed', type = int,
                        default = 15122,
                        help = "random seed for shuffling")
    parser.add_argument('-out', type = str,
                        default = "aalexchengg/github_smol",
                        help ='path to output directory')
    return parser


def get_generator(size):
    def generator():
        temp = load_dataset("codeparrot/github-code", streaming = True, split = "train")
        iterator = iter(temp)
        curr = 0
        while(curr < size):
            curr += 1
            yield next(iterator)
    return generator
    

def does_not_exist(out_name):
    # first check if this work has already been done
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={out_name}"
    response = requests.get(api_url)
    query = response.json()
    if 'error' not in query:
        raise AssertionError("This dataset has already been processed. If you want to overwrite, please manually delete on HuggingFace Hub and try again.")
    return True


def main(args):
    # make sure we aren't writing to an already existing dataset.
    assert(does_not_exist(args.out))
    logging.info("Loading in Github dataset...")
    github = load_dataset(github_repo, split = "train", streaming = True)
    logging.info(f"Shuffling with seed {args.seed}...")
    github = github.shuffle(seed = args.seed)
    logging.info(f"Creating early stopping generator with size {args.size}")
    logging.info("Now building dataset...")
    result = Dataset.from_generator(get_generator(args.size))
    logging.info("Pushing to hub ...")
    result.push_to_hub(args.out)
    logging.info(f"All complete. Dataset can be found at https://huggingface.com/{args.out}")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    main(args)