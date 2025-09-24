#!/usr/bin/env bash
set -euo pipefail

# This script submits one sbatch job per dataset to train the biencoder.
# Datasets: Medmentions, EMEA, Medline, Medmentions_augmented, EMEA_augmented, Medline_augmented

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_SCRIPT="${ROOT_DIR}/scripts/train_biencoder.slurm"

# Allow overriding the partition/node constraint and other sbatch opts via env
SBATCH_EXTRA_OPTS=${SBATCH_EXTRA_OPTS:-}

declare -a DATASETS=(
	"Medmentions"
	"EMEA"
	"Medline"
	"Medmentions_augmented"
	"EMEA_augmented"
	"Medline_augmented"
)

# Map dataset to paths. Adjust here if your data layout differs.
data_path_for() {
	local ds="$1"
	case "$ds" in
		Medmentions) echo "${ROOT_DIR}/data/medmentions/processed" ;;
		EMEA) echo "${ROOT_DIR}/data/emea/processed" ;;
		Medline) echo "${ROOT_DIR}/data/medline/processed" ;;
		Medmentions_augmented) echo "${ROOT_DIR}/data/medmentions_augmented/processed" ;;
		EMEA_augmented) echo "${ROOT_DIR}/data/emea_augmented/processed" ;;
		Medline_augmented) echo "${ROOT_DIR}/data/medline_augmented/processed" ;;
		*) echo "Unknown dataset: $ds" >&2; return 1 ;;
	esac
}

pickle_path_for() {
	local ds="$1"
	case "$ds" in
		Medmentions) echo "${ROOT_DIR}/models/trained/medmentions" ;;
		EMEA) echo "${ROOT_DIR}/models/trained/emea" ;;
		Medline) echo "${ROOT_DIR}/models/trained/medline" ;;
		Medmentions_augmented) echo "${ROOT_DIR}/models/trained/medmentions_augmented" ;;
		EMEA_augmented) echo "${ROOT_DIR}/models/trained/emea_augmented" ;;
		Medline_augmented) echo "${ROOT_DIR}/models/trained/medline_augmented" ;;
		*) echo "Unknown dataset: $ds" >&2; return 1 ;;
	esac
}

output_path_for() {
	local ds="$1"
	local base="${ROOT_DIR}/models/trained"
	case "$ds" in
		Medmentions) echo "${base}/medmentions_mst/pos_neg_loss/no_type" ;;
		EMEA) echo "${base}/emea_mst/pos_neg_loss/no_type" ;;
		Medline) echo "${base}/medline_mst/pos_neg_loss/no_type" ;;
		Medmentions_augmented) echo "${base}/medmentions_augmented_mst/pos_neg_loss/no_type" ;;
		EMEA_augmented) echo "${base}/emea_augmented_mst/pos_neg_loss/no_type" ;;
		Medline_augmented) echo "${base}/medline_augmented_mst/pos_neg_loss/no_type" ;;
		*) echo "Unknown dataset: $ds" >&2; return 1 ;;
	esac
}

mkdir -p "${ROOT_DIR}/logs"

for ds in "${DATASETS[@]}"; do
	DATA_PATH="$(data_path_for "$ds")"
	OUTPUT_PATH="$(output_path_for "$ds")"
	PICKLE_SRC_PATH="$(pickle_path_for "$ds")"

	job_name="biencoder_${ds}"
	log_out="${ROOT_DIR}/logs/${job_name}_%j.out"
	log_err="${ROOT_DIR}/logs/${job_name}_%j.err"

	echo "Submitting: ${job_name}"

	sbatch \
		-J "${job_name}" \
		-o "${log_out}" \
		-e "${log_err}" \
		--export=ALL,DATASET="${ds}",DATA_PATH="${DATA_PATH}",OUTPUT_PATH="${OUTPUT_PATH}",PICKLE_SRC_PATH="${PICKLE_SRC_PATH}" \
		${SBATCH_EXTRA_OPTS} \
		"${SLURM_SCRIPT}"
done

echo "All jobs submitted."
