export LC_ALL="POSIX"

INPUT_MODEL=$1
OUTPUT_DIR=$2
TP=$3


mkdir -p $2/humaneval-infilling
python hm_fim/humaneval_fim.py \
    --model_type codelm_leftright_context \
    --model_name_or_path ${INPUT_MODEL} \
    --right_context_length 2048 \
    --input_file ./hm_fim/data/fim_singleline.jsonl \
    --gen_length 50 \
    --max_seq_length 8192 \
    --output_dir ${OUTPUT_DIR}/humaneval-infilling \
    --tp ${TP}