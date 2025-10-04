#!/usr/bin/env bash
set -euo pipefail

# This script submits one sbatch job per dataset to train the crosscoder.
# Datasets: Medmentions, EMEA, Medline, Medmentions_augmented, EMEA_augmented, Medline_augmented

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="${ROOT_DIR}/scripts/train_crossencoder.slurm"

# Allow overriding the partition/node constraint and other sbatch opts via env
SBATCH_EXTRA_OPTS=${SBATCH_EXTRA_OPTS:-}

declare -a MODELS=(
	"biobert_v1"
	"biobert"
	"coder-all"
)

declare -a DATASETS=(
	"MedMentions"
	"EMEA"
	"MEDLINE"
	"MedMentions_augmented"
	"EMEA_augmented"
	"MEDLINE_augmented"
)

# Map model to paths. Adjust here if your data layout differs.
model_path_for() {
	local model="$1"
	case "$model" in
		biobert) echo "${ROOT_DIR}/models/biobert-base-cased-v1.2" ;;
		biobert_v1) echo "${ROOT_DIR}/models/biobert-base-cased-v1.1" ;;
		coder-all) echo "${ROOT_DIR}/models/coder-all" ;;
		*) echo "Unknown model: $model" >&2; return 1 ;;
	esac
}
# Map epoch to dataset. Adjust here if your data layout differs.
epoch_for() {
	local ds="$1"
	case "$ds" in
		MedMentions) echo 5 ;;
		EMEA) echo 10 ;;
		MEDLINE) echo 10 ;;
		MedMentions_augmented) echo 1 ;;
		EMEA_augmented) echo 2 ;;
		MEDLINE_augmented) echo 2 ;;
		*) echo "Unknown dataset: $ds" >&2; return 1 ;;
	esac
}
# Map dataset to paths. Adjust here if your data layout differs.
data_path_for() {
	local ds="$1"
	echo "${ROOT_DIR}/data/final_data_encoder/${ds}"
}

pickle_path_for() {
	local model="$1"
	local ds="$2"
	echo "${ROOT_DIR}/models/trained/${ds}_${model}"
}
output_path_for() {
	local model="$1"
	local ds="$2"
	local base="${ROOT_DIR}/models/trained"
	echo "${base}/${ds}_${model}/crossencoder/arbo"
}
biencoder_candidates_path_for() {
	local model="$1"
	local ds="$2"
	local base="${ROOT_DIR}/models/trained"
	echo "${base}/${ds}_${model}/candidates/arbo"
}
biencoder_path_for() {
	local model="$1"
	local ds="$2"
	local base="${ROOT_DIR}/models/trained"
	echo "${base}/${ds}_${model}/pos_neg_loss/with_type/pytorch_model.bin"
}
mkdir -p "${ROOT_DIR}/logs"
for model in "${MODELS[@]}"; do
    for ds in "${DATASETS[@]}"; do
    	DATA_PATH="$(data_path_for "$ds")"
    	OUTPUT_PATH="$(output_path_for "$model" "$ds")"
    	BIENCODER_CAND="$(biencoder_candidates_path_for "$model" "$ds")"
    	PICKLE_SRC_PATH="$(pickle_path_for "$model" "$ds")"
    	BIENCODER_PATH="$(biencoder_path_for "$model" "$ds")"
    	BERT_MODEL="$(model_path_for "$model")"
		EPOCHS="$(epoch_for "$ds")"

    	job_name="crossencoder_${ds}"
    	log_out="${ROOT_DIR}/logs/${job_name}_%j.out"
    	log_err="${ROOT_DIR}/logs/${job_name}_%j.err"
    
    	echo "Submitting: ${job_name}"
    
    	sbatch \
    		-J "${job_name}" \
    		-o "${log_out}" \
    		-e "${log_err}" \
			-A ssq@h100 \
    		--export=ALL,DATASET="${ds}",DATA_PATH="${DATA_PATH}",OUTPUT_PATH="${OUTPUT_PATH}",PICKLE_SRC_PATH="${PICKLE_SRC_PATH}",BERT_MODEL="${BERT_MODEL}",BIENCODER_PATH="${BIENCODER_PATH}",EPOCHS="${EPOCHS}",BIENCODER_CAND="${BIENCODER_CAND}" \
    		${SBATCH_EXTRA_OPTS} \
    		"${SLURM_SCRIPT}"
    done
done
echo "All jobs submitted."
