#!/bin/bash

export PYTHONPATH=/Data/divya/Graformer:$PYTHONPATH
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=3,4

DATA_BIN="/mnt/storage/divya/exam/part2/data-bin"

SAVE_DIR="/mnt/storage/divya/exam/graformer_workspace/gpt_nmt"

DECODER_CKPT="/mnt/storage/divya/exam/graformer_workspace/dgpt124m_checkpoints/checkpoint_last.pt"

ENCODER_CKPT="/mnt/storage/divya/exam/graformer_workspace/egpt124m_checkpoints/checkpoint_last.pt"

mkdir -p ${SAVE_DIR}

CMD="python3"

lang_pairs="mr-hi,hi-mr"
lang_list="mr,hi"

echo "[INFO] DATA_BIN: ${DATA_BIN}"
echo "[INFO] SAVE_DIR: ${SAVE_DIR}"
echo "[INFO] ENCODER_CKPT: ${ENCODER_CKPT}"
echo "[INFO] DECODER_CKPT: ${DECODER_CKPT}"

$CMD ../train.py \
    ${DATA_BIN} \
    --task translation_multi_simple_epoch \
    --langs ${lang_list} \
    --lang-pairs ${lang_pairs} \
    --sampling-method temperature \
    --sampling-temperature 5 \
    --decoder-langtok \
    --lang-tok-replacing-bos-eos \
    --arch llama_bridge_transformer \
    --encoder-layers 8 \
    --decoder-layers 12 \
    --encoder-embed-dim 768 \
    --decoder-embed-dim 768 \
    --encoder-ffn-embed-dim 3072 \
    --decoder-ffn-embed-dim 3072 \
    --encoder-attention-heads 12 \
    --decoder-attention-heads 12 \
    --encoder-kv-attention-heads 4 \
    --decoder-kv-attention-heads 4 \
    --use-rmsnorm \
    --use-rope \
    --rope-theta 10000 \
    --attn-implementation fast_gqa \
    --cross-attention-implementation mha \
    --no-token-positional-embeddings \
    --no-scale-embedding \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --activation-fn gelu \
    --no-encoder-attn-layers 0,1,2,3 \
    --finetune-from-model ${ENCODER_CKPT},${DECODER_CKPT} \
    --freeze-params "(.embed_tokens.)|(.layers\.(0|1|2|3)\.)" \
    --transfer-params "decoder.embed_tokens.weight:decoder.lm_output_projection.weight" \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9,0.95)' \
    --adam-eps 1e-8 \
    --weight-decay 0.1 \
    --clip-norm 1.0 \
    --lr 3e-4 \
    --warmup-init-lr 1e-07 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --max-tokens 1024 \
    --update-freq 4 \
    --num-workers 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    --checkpoint-activations \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --disable-validation \
    --save-dir ${SAVE_DIR} \
    --fp16 \
    --fp16-init-scale 128 \
    --min-loss-scale 0.0 \
    --ddp-backend no_c10d

echo "[INFO] Training Finished!"
