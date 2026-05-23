#!/bin/bash

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

echo "[logging] Starting LLaMA-style Graformer LM training..."

# ==========================================
# LOCAL PATHS
# ==========================================

local_root="/mnt/storage/divya/exam/graformer_workspace"

# dataset root
local_dataset_path="/mnt/storage/divya/exam/"

# checkpoints
local_checkpoint_path=${local_root}/lmcheckpoints

mkdir -p ${local_checkpoint_path}

echo "[logging] dataset: ${local_dataset_path}"
echo "[logging] checkpoints: ${local_checkpoint_path}"

# ==========================================
# PYTHON
# ==========================================

CMD="python3"

# ==========================================
# DATA
# ==========================================

lang_list="hi,mr"

# multilingual_language_modeling expects data-bin
local_data="${local_dataset_path}/data-bin/lm-data"

# ==========================================
# TRAIN
# ==========================================

$CMD ../train.py \
    --num-workers 2 \
    ${local_data} \
    --task multilingual_language_modeling \
    --langs ${lang_list} \
    --multilang-sampling-alpha 0.7 \
    --sample-break-mode eos \
    --valid-subset none \
    --disable-validation \
    \
    --arch llama_lm_base \
    \
    --decoder-layers 6 \
    --decoder-embed-dim 1024 \
    --decoder-ffn-embed-dim 4096 \
    --decoder-attention-heads 16 \
    \
    --decoder-normalize-before \
    --decoder-use-rmsnorm \
    \
    --share-decoder-input-output-embed \
    --no-scale-embedding \
    --no-token-positional-embeddings \
    \
    --use-rope \
    --rope-theta 10000 \
    \
    --decoder-kv-attention-heads 4 \
    \
    --activation-fn gelu \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    \
    --tokens-per-sample 2048 \
    --max-tokens 8000 \
    \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --lr 0.001 \
    --warmup-init-lr 1e-07 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --update-freq 5 \
    \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    \
    --max-epoch 1 \
    --min-loss-scale 0.0 \
    \
    --save-interval 1 \
    --keep-last-epochs 2 \
    --save-dir ${local_checkpoint_path} \
    \
    --fp16 \
    --ddp-backend no_c10d

echo "[logging] Training Finished!"
