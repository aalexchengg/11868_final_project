
INPUT_MODEL=$1
OUTPUT_DIR=$2
TP=$3

bash run_cceval.sh ${INPUT_MODEL} ${OUTPUT_DIR} ${TP}

bash run_cclongeval.sh ${INPUT_MODEL} ${OUTPUT_DIR} ${TP}

bash run_hm_fim.sh ${INPUT_MODEL} ${OUTPUT_DIR} ${TP}

bash run_repoeval.sh ${INPUT_MODEL} ${OUTPUT_DIR} ${TP}