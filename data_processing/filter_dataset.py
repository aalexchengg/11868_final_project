"""
Lambda Function to Filter Datasets

Example Data Point:
{
'uuid': <some shit>
 'code': 'import mod189 from './mod189';\nvar value=mod189+1;\nexport default value;\n <|endoftext|> ',
 'language': 'JavaScript',
 'source': 'MirekSz/webpack-es6-ts/app/mods/mod190.js'
}

Filtering idea from DeepSeek-Coder https://arxiv.org/pdf/2401.14196

Assuming data comes from either https://huggingface.co/datasets/codeparrot/github-code or
https://huggingface.co/datasets/bigcode/the-stack

"""

from datasets import load_dataset
import time
from bs4 import BeautifulSoup
import json
import numpy as np
import os
from functools import cache
from tqdm import tqdm

@cache
def import_language_data():
    language_data = None
    with open('languages.json', 'r') as file:
        language_data = json.load(file)
    return language_data

def filter(example):
    code = example['code'] if 'code' in example else example['content']
    code_lang = example['language'] if 'language' in example else example['lang']
    path = example['source'] if 'source' in example else example['max_stars_repo_path']
    # else statements above to accommodate for the stack, remove for standardized
    extension = os.path.splitext(path)[1]
    lines = code.split('\n')
    lens = np.array([len(line) for line in lines])

    # ensure language matches
    ## TAKE THIS OUT START
    langs = import_language_data()
    found_lang = False
    for source in langs: # either github or thestack
        for lang in langs[source]: # all languages
            if extension in langs[source][lang] and code_lang == lang:
                found_lang = True
    if not found_lang: return False
    ## TAKE THIS OUT END

    # assuming newlines exist
    if '\n' not in code: return False

    # not empty
    if len(code) == 0: return False

    # avg line length > 100 or max line length > 1000
    if np.mean(lens) > 100 or np.max(lens) > 1000: return False

    # fewer than 25% alphabetic characters
    alpha_proportion = sum(1 for char in code if char.isalpha()) / float(len(code))
    if alpha_proportion < 0.25: return False

    # other than XSLT check for <?xml version=
    code_prefix = code[:100]
    if code_lang != "XSLT" and "<?xml version=" in code_prefix: return False

    # visible text constitutes >= 20% of code, no less than 100 characters
    if code_lang == "HTML":
        soup = BeautifulSoup(code, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()

        visible_text = soup.get_text(separator=' ', strip=True)
        visible_char_count = len(visible_text)
        if visible_char_count < 100 or visible_char_count / float(len(code)) < 0.2: return False

    # json and yaml files character count from 50 to 5000
    if code_lang == "YAML" or "JSON" in code_lang:
        if len(code) < 50 or len(code) > 5000: return False
    return True


if __name__ == '__main__':
    # print("===\nPROOF OF CONCEPT\n===")
    # github_data = load_dataset("codeparrot/github-code", split="train", streaming=True)
    # raw_start = time.time()
    # for i in tqdm(range(5), desc="no filtering github"):
    #     next(iter(github_data))
    # raw_end = time.time()
    # print(f"no filtering time: {raw_end - raw_start}")
    #
    # github_filter = github_data.filter(lambda example: filter(example))
    # filter_start = time.time()
    # for i in tqdm(range(5), desc="filtering github"):
    #     next(iter(github_filter))
    # filter_end = time.time()
    # print(f"filtering time: {filter_end - filter_start}")
    #
    # print("===\nSHUFFLE AND TRY AGAIN\n===")
    # github_data_shuffled = github_data.shuffle()
    #
    # raw_start = time.time()
    # for i in tqdm(range(5), desc="no filtering github, shuffled"):
    #     next(iter(github_data_shuffled))
    # raw_end = time.time()
    # print(f"no filtering time, shuffled: {raw_end - raw_start}")
    #
    # github_filter = github_data_shuffled.filter(lambda example: filter(example))
    # filter_start = time.time()
    # for i in tqdm(range(5), desc="filtering github, shuffled"):
    #     next(iter(github_filter))
    # filter_end = time.time()
    # print(f"filtering time, shuffled: {filter_end - filter_start}")



    print("===\nPROOF OF CONCEPT\n===")
    stack_data = load_dataset("bigcode/the-stack", split="train", streaming=True)
    raw_start = time.time()
    for i in tqdm(range(50), desc="no filtering stack"):
        next(iter(stack_data))
    raw_end = time.time()
    print(f"no filtering time: {raw_end - raw_start}")

    stack_filter = stack_data.filter(lambda example: filter(example))
    filter_start = time.time()
    for i in tqdm(range(50), desc="filtering stack"):
        next(iter(stack_filter))
    filter_end = time.time()
    print(f"filtering time: {filter_end - filter_start}")

    print("===\nSHUFFLE AND TRY AGAIN\n===")
    stack_data_shuffled = stack_data.shuffle()

    raw_start = time.time()
    for i in tqdm(range(50), desc="no filtering stack, shuffled"):
        next(iter(stack_data_shuffled))
    raw_end = time.time()
    print(f"no filtering time, shuffled: {raw_end - raw_start}")

    stack_filter = stack_data_shuffled.filter(lambda example: filter(example))
    filter_start = time.time()
    for i in tqdm(range(50), desc="filtering stack, shuffled"):
        next(iter(stack_filter))
    filter_end = time.time()
    print(f"filtering time, shuffled: {filter_end - filter_start}")
