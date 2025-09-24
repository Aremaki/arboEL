#!/usr/bin/env bash
#SBATCH -C a100                      # target A100 nodes
#SBATCH --ntasks=1                   # total MPI tasks (= total GPUs)
#SBATCH --ntasks-per-node=1          # MPI tasks per node (= GPUs per node)
#SBATCH --gres=gpu:1                 # GPUs per node
#SBATCH --cpus-per-task=24           # CPU cores per task
#SBATCH --hint=nomultithread         # disable hyperthreading
#SBATCH --time=20:00:00              # walltime (HH:MM:SS)
# NOTE: Job name and log files can be overridden via sbatch -J/-o/-e from the wrapper.
#SBATCH --job-name=biencoder
#SBATCH --output=logs/biencoder_%j.out
#SBATCH --error=logs/biencoder_%j.err

set -euo pipefail

# Clean any inherited modules and load cluster toolchains
module purge
module load arch/a100
module load pytorch-gpu/py3/1.11.0

# Resolve parameters (override via sbatch --export or environment)
DATASET_NAME="${DATASET:-medmentions}"
DATA_PATH="${DATA_PATH:-data/medmentions/processed}"
OUTPUT_PATH="${OUTPUT_PATH:-models/trained/medmentions_mst/pos_neg_loss/no_type}"
PICKLE_SRC_PATH="${PICKLE_SRC_PATH:-models/trained/medmentions}"

BERT_MODEL="${BERT_MODEL:-models/biobert-base-cased-v1.1}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM_STEPS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-3500}"

echo "Starting biencoder training"
echo "Dataset:        ${DATASET_NAME}"
echo "Data path:      ${DATA_PATH}"
echo "Output path:    ${OUTPUT_PATH}"
echo "Pickle src:     ${PICKLE_SRC_PATH}"
echo "BERT model:     ${BERT_MODEL}"
echo "Epochs:         ${EPOCHS}"
echo "Batch size:     ${TRAIN_BATCH_SIZE} (grad accum ${GRADIENT_ACCUMULATION_STEPS})"
echo "Eval interval:  ${EVAL_INTERVAL}"
echo "Embed batch:    ${EMBED_BATCH_SIZE}"

# Ensure directories exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p logs

# Train biencoder
python -u blink/biencoder/train_biencoder_mst.py \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${OUTPUT_PATH}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--num_train_epochs="${EPOCHS}" \
	--train_batch_size="${TRAIN_BATCH_SIZE}" \
	--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
	--eval_interval="${EVAL_INTERVAL}" \
	--pos_neg_loss \
	--force_exact_search \
	--embed_batch_size="${EMBED_BATCH_SIZE}" \
	--data_parallel

echo "SCRIPT FINISHED"
