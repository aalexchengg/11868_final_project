import datasets
from datasets import load_dataset

repo_name = "google/code_x_glue_cc_code_completion_line"
out_name = "aalexchengg/codexglue-code-formatted"

def format_text(text: str) -> str:
    #remove all the special tokens
    text = re.sub('<s>', '', text)
    text = re.sub('</s>', '', text)
    # keep <EOL> token for now
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
        sources.append(f"codexglue_{examples['id'][i]}")
    result = {"id": ids, "text": texts, "language": languages, "source": sources}
    return result


def process_codexglue():
    # first check if this work has already been done
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={out_name}"
    response = requests.get(api_url)
    query = response.json()
    if 'error' not in query:
        raise AssertionError("This dataset has already been processed. If you want to overwrite, please manually delete on HuggingFace Hub and try again.")
    # otherwise, do the work
    logger.info("Loading in source dataset...")
    py_dataset = load_dataset(repo_name, split = "train", "python") 
    java_dataset = load_dataset(repo_name, split = "train", "java")
    dataset = datasets.concatenate_datasets([py_dataset, java_dataset])
    logger.info("Mapping dataset...")
    dataset.map(format_examples, batched = True)
    logger.info("Pushing to hub...")
    dataset.push_to_hub("aalexchengg/github-code-formatted")
    logger.info("All finished.")



if __name__ == "__main__":
