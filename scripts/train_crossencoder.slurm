#!/usr/bin/env bash
#SBATCH -C a100              # target A100 nodes
#SBATCH --ntasks=1                   # total MPI tasks (= total GPUs)
#SBATCH --ntasks-per-node=1         # MPI tasks per node (= GPUs per node)
#SBATCH --gres=gpu:1                 # GPUs per node
#SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU V100)
#SBATCH --hint=nomultithread         # disable hyperthreading
#SBATCH --time=20:00:00              # walltime (HH:MM:SS)
# NOTE: Job name and log files can be overridden via sbatch -J/-o/-e from the wrapper.
#SBATCH --job-name=train_cross
#SBATCH --output=logs/train_cross_%j.out
#SBATCH --error=logs/train_cross_%j.err

set -euo pipefail

# Clean any inherited modules and load cluster toolchains
module purge
module load arch/a100
module load pytorch-gpu/py3/2.1.1
module load magma/2.8.0-cuda

# Resolve parameters (override via sbatch --export or environment)
DATASET_NAME="${DATASET:-medmentions}"
DATA_PATH="${DATA_PATH:-data/medmentions/processed}"
OUTPUT_PATH="${OUTPUT_PATH:-models/trained/medmentions_mst/pos_neg_loss/no_type}"
PICKLE_SRC_PATH="${PICKLE_SRC_PATH:-models/trained/medmentions}"
BIENCODER_PATH="${BIENCODER_PATH:-models/trained/medmentions}"
EPOCHS="${EPOCHS:-5}"
BIENCODER_CAND="${BIENCODER_CAND:-models/trained/medmentions/candidates/arbo}"

BERT_MODEL="${BERT_MODEL:-models/biobert-base-cased-v1.1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-64}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-3500}"
RECALL_K="${RECALL_K:-64}"
EMBED_CROSS="${EMBED_CROSS:-768}"

echo "Starting crossencoder training"
echo "Dataset:        ${DATASET_NAME}"
echo "Data path:      ${DATA_PATH}"
echo "Output path:    ${OUTPUT_PATH}"
echo "Pickle src:     ${PICKLE_SRC_PATH}"
echo "Biencoder src:  ${BIENCODER_PATH}"
echo "Bi-encoder candidates path:      ${BIENCODER_CAND}"
echo "BERT model:     ${BERT_MODEL}"
echo "Epochs:         ${EPOCHS}"
echo "Learning rate:      ${LEARNING_RATE}"
echo "Train batch size:     ${TRAIN_BATCH_SIZE}"
echo "Eval batch size:     ${EVAL_BATCH_SIZE}"
echo "Scoring batch size: ${SCORING_BATCH_SIZE}"
echo "Recall k: ${RECALL_K}"
echo "Embed batch:    ${EMBED_BATCH_SIZE}"

# Ensure directories exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${BIENCODER_CAND}"
mkdir -p logs

# Evaluate Bi-encoder model (optional, but useful to have the results)
python -u blink/biencoder/eval_cluster_linking.py \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${BIENCODER_CAND}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--path_to_model="${BIENCODER_PATH}" \
	--scoring_batch_size="${SCORING_BATCH_SIZE}" \
	--embed_batch_size="${EMBED_BATCH_SIZE}" \
	--recall_k="${RECALL_K}" \
	--save_topk_result \
	--use_types \

# Generate dual-encoder candidates
python -u blink/crossencoder/eval_cluster_linking.py \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${BIENCODER_CAND}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--path_to_biencoder_model="${BIENCODER_PATH}" \
	--scoring_batch_size="${SCORING_BATCH_SIZE}" \
	--save_topk_result \
	--use_types \

# Run cross-encoder training
python -u blink/crossencoder/original/train_cross.py \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${OUTPUT_PATH}" \
	--biencoder_indices_path="${BIENCODER_CAND}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--num_train_epochs="${EPOCHS}" \
	--learning_rate="${LEARNING_RATE}" \
	--train_batch_size="${TRAIN_BATCH_SIZE}" \
	--eval_batch_size="${EVAL_BATCH_SIZE}" \
	--embed_batch_size="${EMBED_CROSS}" \
	--eval_interval=-1 \
	--add_linear \
	--skip_initial_eval \

# Evaluate the trained cross-encoder
python -u blink/crossencoder/original/train_cross.py \
	--bert_model="${BERT_MODEL}" \
	--data_path="${DATA_PATH}" \
	--output_path="${OUTPUT_PATH}" \
	--biencoder_indices_path="${BIENCODER_CAND}" \
	--pickle_src_path="${PICKLE_SRC_PATH}" \
	--learning_rate="${LEARNING_RATE}" \
	--eval_batch_size="${EVAL_BATCH_SIZE}" \
	--add_linear \
	--only_evaluate \
	--path_to_model="${OUTPUT_PATH}/pytorch_model.bin" \


echo "SCRIPT FINISHED"
