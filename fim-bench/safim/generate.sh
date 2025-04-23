INPUT_MODEL=$1

echo "Generating for Algorithmic Block Completion..."
mkdir -p cache outputs_block
python generate.py \
  ${INPUT_MODEL} \
  block \
  cache/${INPUT_MODEL}.json \
  outputs_block/${INPUT_MODEL}-fim-tb.jsonl \
  infilling \
  --post_processors truncate_line_until_block

echo "Generating for Control-Flow Expression Completion..."
mkdir -p cache outputs_control
python generate.py \
  ${INPUT_MODEL} \
  control \
  cache/${INPUT_MODEL}.json \
  outputs_control/${INPUT_MODEL}-fim-tc.jsonl \
  infilling \
  --post_processors truncate_control

echo "Generating for API Function Call Completion..."

mkdir -p cache outputs_api
python generate.py \
  ${INPUT_MODEL} \
  api \
  cache/${INPUT_MODEL}.json \
  outputs_api/${INPUT_MODEL}-fim-ta.jsonl \
  infilling \
  --post_processors truncate_api_call