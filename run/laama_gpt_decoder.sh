#!/bin/bash

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4

echo "[logging] Starting GPT-124M style Graformer LM training..."

# ==========================================
# LOCAL PATHS
# ==========================================

local_root="/mnt/storage/divya/exam/graformer_workspace"

# dataset root
#local_dataset_path="/mnt/storage/divya/exam/"
local_dataset_path="/mnt/storage/divya/exam/part2/data-bin"
# checkpoints
local_checkpoint_path=${local_root}/dgpt124m_checkpoints

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
#local_data="${local_dataset_path}/data-bin/lm-data"
local_data="${local_dataset_path}"
# ==========================================
# TRAIN
# ==========================================

$CMD ../train.py \
    --num-workers 2 \
    ${local_data} \
    --task multilingual_language_modeling \
    --valid-subset valid \
    --validate-interval-updates 1000 \
    --best-checkpoint-metric loss \
    --langs ${lang_list} \
    --multilang-sampling-alpha 0.7 \
    --sample-break-mode eos \
    \
    --arch llama_lm_base \
    \
    --decoder-layers 12 \
    --decoder-embed-dim 768 \
    --decoder-ffn-embed-dim 3072 \
    --decoder-attention-heads 12 \
    \
    --decoder-normalize-before \
    --decoder-use-rmsnorm \
    \
    --share-decoder-input-output-embed \
    --no-scale-embedding \
    \
    --use-rope \
    --rope-theta 10000 \
    \
    --decoder-kv-attention-heads 4 \
    \
    --activation-fn gelu \
    --dropout 0.2 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    \
    --tokens-per-sample 512 \
    --max-tokens 4096 \
    \
    --optimizer adam \
    --adam-betas '(0.9,0.95)' \
    --lr 3e-4 \
    --warmup-init-lr 1e-07 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 \
    --update-freq 4 \
    \
    --criterion cross_entropy \
    \
    --max-epoch 10 \
    --min-loss-scale 0.0 \
    \
    --clip-norm 1.0 \
    \
    --save-interval 2 \
    --keep-last-epochs 2 \
    --save-dir ${local_checkpoint_path} \
    \
    --fp16 \
    --log-format json \
    --log-interval 10 \
    --tensorboard-logdir ${local_checkpoint_path}/tensorboard \
    \
    --ddp-backend no_c10d

echo "[logging] Training Finished!"
