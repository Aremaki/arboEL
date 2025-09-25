#!/usr/bin/env bash
set -euo pipefail

# This script submits one sbatch job per dataset to train the biencoder.
# Datasets: Medmentions, EMEA, Medline, Medmentions_augmented, EMEA_augmented, Medline_augmented

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="${ROOT_DIR}/scripts/train_biencoder.slurm"

# Allow overriding the partition/node constraint and other sbatch opts via env
SBATCH_EXTRA_OPTS=${SBATCH_EXTRA_OPTS:-}

declare -a MODELS=(
	"biobert"
	"coder-all"
)

declare -a DATASETS=(
	"MedMentions"
	"EMEA"
	"MEDLINE"
	# "Medmentions_augmented"
	# "EMEA_augmented"
	# "Medline_augmented"
)

# Map model to paths. Adjust here if your data layout differs.
model_path_for() {
	local model="$1"
	case "$model" in
		biobert) echo "${ROOT_DIR}/models/biobert-base-cased-v1.1" ;;
		coder-all) echo "${ROOT_DIR}/models/coder-all" ;;
		*) echo "Unknown model: $model" >&2; return 1 ;;
	esac
}

# Map dataset to paths. Adjust here if your data layout differs.
data_path_for() {
	local ds="$1"
	echo "${ROOT_DIR}/data/final_data_encoder/${ds}"
}

pickle_path_for() {
pickle_path_for() {
	local model="$1"
	local ds="$2"
	echo "${ROOT_DIR}/models/trained/${ds}_${model}"
}
output_path_for() {
output_path_for() {
	local model="$1"
	local ds="$2"
	local base="${ROOT_DIR}/models/trained"
	echo "${base}/${ds}_${model}/pos_neg_loss/with_type"
}
mkdir -p "${ROOT_DIR}/logs"
for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
    	DATA_PATH="$(data_path_for "$ds")"
    	OUTPUT_PATH="$(output_path_for "$model" "$ds")"
    	PICKLE_SRC_PATH="$(pickle_path_for "$model" "$ds")"
    	BERT_MODEL="$(model_path_for "$model")"
    
    	job_name="biencoder_${ds}"
    	log_out="${ROOT_DIR}/logs/${job_name}_%j.out"
    	log_err="${ROOT_DIR}/logs/${job_name}_%j.err"
    
    	echo "Submitting: ${job_name}"
    
    	sbatch \
    		-J "${job_name}" \
    		-o "${log_out}" \
    		-e "${log_err}" \
			-A ssq@a100 \
    		--export=ALL,DATASET="${ds}",DATA_PATH="${DATA_PATH}",OUTPUT_PATH="${OUTPUT_PATH}",PICKLE_SRC_PATH="${PICKLE_SRC_PATH}",BERT_MODEL="${BERT_MODEL}" \
    		${SBATCH_EXTRA_OPTS} \
    		"${SLURM_SCRIPT}"
    done

echo "All jobs submitted."
