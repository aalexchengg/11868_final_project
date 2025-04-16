from datasets import load_dataset


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

