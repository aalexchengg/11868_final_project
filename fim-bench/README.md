# Fill-In-the-Middle (FIM) benchmarks

## Jackson Notes

- Changed `cceval.py` call to `LLM` to have `dtype="float16"` so it works on my local GPU - others should change back.
- E.g. run benchmark command: `./run_cceval.sh Qwen/Qwen2.5-0.5B ./cceval_out 1`
- Looks like we'd train a model locally, store it, then pass the file path to the model into the .sh script instead of one from Hugging Face?
- Need the .so files in /build, can re-create them if needed by running ./build_treesitter.sh

## CrossCodeEval

- Data Preparation

```bash
cd cceval
bash prepare_data.sh
```

- Usage

```bash
bash ./run_cceval.sh <model_path> <output_dir> <tp>
```

- Parameter Description

* `<model_path>`: Path to the pre-trained model
* `<output_dir>`: Directory to save the evaluation results
* `<tp>`: Number of parallel GPUs

- Whether to use cross-file context
  The script supports two context modes, controlled by the `model_type` parameter:

* `codelm_right_cfc_left`: Enable cross-file context mode
* `codelm_leftright_context`: Disable cross-file context mode

- Main Parameters

* `cfc_seq_length`: Maximum length of cross-file context (default: 2048)
* `right_context_length`: Maximum length of right context (default: 2048)
* `gen_length`: Length of generated code completion (default: 50)
* `max_seq_length`: Maximum total sequence length (default: 8192)

## CrossCodeLongEval

- Data Preparation

```bash
cd cclongeval
bash prepare_data.sh
```

- Usage

```bash
bash ./run_cclongeval.sh <model_path> <output_dir> <tp>
```

## RepoEval

- Usage

```bash
bash ./run_repoeval.sh <model_path> <output_dir> <tp>
```

## humaneval-infilling

- Usage

```bash
bash ./run_hm_fim.sh <model_path> <output_dir> <tp>
```

## Environment Description

_tree-sitter == 0.20.1_
