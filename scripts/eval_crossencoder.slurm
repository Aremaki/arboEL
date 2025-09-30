#!/usr/bin/env bash
#SBATCH -C h100              # target H100 nodes
#SBATCH --ntasks=1                   # total MPI tasks (= total GPUs)
#SBATCH --ntasks-per-node=1         # MPI tasks per node (= GPUs per node)
#SBATCH --gres=gpu:1                 # GPUs per node
#SBATCH --cpus-per-task=12           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU V100)
#SBATCH --hint=nomultithread         # disable hyperthreading
#SBATCH --time=20:00:00              # walltime (HH:MM:SS)
# NOTE: Job name and log files can be overridden via sbatch -J/-o/-e from the wrapper.
#SBATCH --job-name=eval_cross
#SBATCH --output=logs/eval_cross_%j.out
#SBATCH --error=logs/eval_cross_%j.err

set -euo pipefail

# Clean any inherited modules and load cluster toolchains
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# Resolve parameters (override via sbatch --export or environment)
DATASET_NAME="${DATASET:-medmentions}"
DATA_PATH="${DATA_PATH:-data/medmentions/processed}"
OUTPUT_PATH="${OUTPUT_PATH:-models/trained/medmentions_mst/pos_neg_loss/no_type}"
PICKLE_SRC_PATH="${PICKLE_SRC_PATH:-models/trained/medmentions}"
BERT_MODEL="${BERT_MODEL:-models/biobert-base-cased-v1.1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
BIENCODER_CAND="${BIENCODER_CAND:-models/trained/medmentions/candidates/arbo}"
CROSSENCODER_PATH="${CROSSENCODER_PATH:-models/trained/medmentions}"

echo "Starting crossencoder evaluation"
echo "Dataset:        ${DATASET_NAME}"
echo "Cross-encoder path:      ${CROSSENCODER_PATH}"
echo "Bi-encoder candidates path:      ${BIENCODER_CAND}"
echo "Data path:      ${DATA_PATH}"
echo "Output path:    ${OUTPUT_PATH}"
echo "Pickle src:     ${PICKLE_SRC_PATH}"
echo "BERT model:     ${BERT_MODEL}"
echo "Eval batch size:     ${EVAL_BATCH_SIZE}"

# Ensure directories exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p logs

# Evaluate crossencoder
python -u blink/crossencoder/original/train_cross.py \
	--path_to_model="${CROSSENCODER_PATH}" \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${OUTPUT_PATH}" \
	--biencoder_indices_path="${BIENCODER_CAND}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--eval_batch_size="${EVAL_BATCH_SIZE}" \
    --only_evaluate \
echo "SCRIPT FINISHED"
