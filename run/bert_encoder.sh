export PYTHONPATH=/Data/divya/Graformer:$PYTHONPATH

#!/bin/bash

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=2

echo "[logging] Starting GPT-124M style GQA MLM training..."

# ==========================================
# PATHS
# ==========================================

local_root="/mnt/storage/divya/exam/graformer_workspace"

# local_dataset_path="/mnt/storage/divya/exam/data-bin/lm-data"
local_dataset_path="/mnt/storage/divya/exam/part2/data-bin"

local_checkpoint_path=${local_root}/ebert124m_checkpoints

mkdir -p ${local_checkpoint_path}

echo "[logging] local_dataset_path: ${local_dataset_path}"
echo "[logging] local_checkpoint_path: ${local_checkpoint_path}"

CMD="python3"

lang_list="hi,mr"

local_data="${local_dataset_path}"

# ==========================================
# TRAIN
# ==========================================

$CMD ../train.py \
    ${local_data} \
    \
    --task new_multilingual_masked_lm \
    \
    --langs ${lang_list} \
    --valid-subset none \
    --disable-validation \
    --multilang-sampling-alpha 0.7 \
    \
    --encoder-learned-pos \
    --sample-break-mode eos \
    --replace-mask-with-bos \
    \
    --arch transformer_encoder_model_12l_12h_768 \
    \
    --activation-fn gelu \
    \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    \
    --encoder-normalize-before \
    \
    --share-encoder-input-output-embed \
    \
    --no-scale-embedding \
    \
    --tokens-per-sample 1024 \
    \
    --max-tokens 4096 \
    \
    --optimizer adam \
    \
    --adam-betas '(0.9,0.95)' \
    \
    --adam-eps 1e-8 \
    \
    --clip-norm 1.0 \
    \
    --weight-decay 0.1 \
    \
    --lr 2e-4 \
    \
    --warmup-init-lr 1e-7 \
    \
    --lr-scheduler inverse_sqrt \
    \
    --warmup-updates 4000 \
    \
    --update-freq 8 \
    \
    --criterion label_smoothed_cross_entropy \
    \
    --label-smoothing 0.1 \
    \
    --max-epoch 10 \
    \
    --save-interval 1 \
    \
    --keep-last-epochs 2 \
    \
    --save-dir ${local_checkpoint_path} \
    \
    --log-format json \
    --log-interval 10 \
    --tensorboard-logdir ${local_checkpoint_path}/tensorboard \
    \
    --fp16 \
    \
    --ddp-backend no_c10d

echo "[logging] Training Finished!"
