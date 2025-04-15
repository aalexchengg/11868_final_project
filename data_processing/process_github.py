"""
Processes Github-Code dataset into CodeXGLUE format.
"""
repo_name = "codeparrot/github-code"
out_name = "aalexchengg/github-code-formatted"

import requests
import datasets
import uuid
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


def format_text(text: str) -> str:
    # TODO: implement
    return text


def format_examples(examples):
    ids = []
    texts = []
    sources = []
    languages = []
    for i, text in enumerate(examples['code']):
        texts.append(format_text(text))
        ids.append(uuid.uuid4())
        languages.append(examples['languages'][i])
        sources.append(f"githubcode_{examples['repo_name'][i]}/{examples['path'][i]}")
    result = {"id": ids, "text": texts, "language": languages, "source": sources}
    return result



def process_github():
    # first check if this work has already been done
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={out_name}"
    response = requests.get(api_url)
    query = response.json()
    if 'error' not in query:
        raise AssertionError("This dataset has already been processed. If you want to overwrite, please manually delete on HuggingFace Hub and try again.")
    # otherwise, do the work
    logger.info("Loading in source dataset...")
    dataset = load_dataset(repo_name, streaming = True)
    logger.info("Mapping dataset...")
    dataset.map(format_examples, batched = True)
    logger.info("Pushing to hub...")
    dataset.push_to_hub("aalexchengg/github-code-formatted")
    logger.info("All finished.")



if __name__ == "__main__":
    process_github()