import uuid

def format_text(text: str) -> str:
    # all we need to do is append the end of text special token
    return text + " <|endoftext|>"


def github_format_fn(examples):
    ids = []
    texts = []
    sources = []
    languages = []
    for i, text in enumerate(examples['code']):
        texts.append(format_text(text))
        ids.append(str(uuid.uuid4()))
        languages.append(examples['language'][i])
        sources.append(f"githubcode_{examples['repo_name'][i]}/{examples['path'][i]}")
    result = {"id": ids, "code": texts, "language": languages, "source": sources}
    return result

def stack_format_fn(examples):
    ids = []
    texts = []
    sources = []
    languages = []
    for i, text in enumerate(examples['content']):
        texts.append(format_text(text))
        ids.append(str(uuid.uuid4()))
        languages.append(examples['lang'][i])
        sources.append(f"stack_{examples['hexsha'][i]}")
    result = {"id": ids, "code": texts, "language": languages, "source": sources}
    return result
