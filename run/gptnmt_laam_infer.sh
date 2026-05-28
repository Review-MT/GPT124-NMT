export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/Data/divya/Graformer:$PYTHONPATH
#!/bin/bash

# =========================================================
# Usage:
# bash infer_mr_hi.sh
# =========================================================

DATA_BIN="/mnt/storage/divya/exam/raw/data-bin/"

CKPT_PATH="/mnt/storage/divya/exam/graformer_workspace/gpt_nmt2/checkpoint_last.pt"

OUTPUT_DIR="/mnt/storage/divya/exam/graformer_workspace/inference_outputs"

USER_DIR="/mnt/storage/divya/exam/graformer_workspace/gpt_nmt2" #"/Data/divya/Graformer"

mkdir -p ${OUTPUT_DIR}

CMD="python3"

echo "[INFO] =========================================="
echo "[INFO] DATA_BIN: ${DATA_BIN}"
echo "[INFO] CKPT_PATH: ${CKPT_PATH}"
echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
echo "[INFO] USER_DIR: ${USER_DIR}"
echo "[INFO] =========================================="

# =========================================================
# Marathi-Hindi bilingual setup
# =========================================================

lang_pairs="mr-hi,hi-mr"
lang_list="mr,hi"

IFS=',' read -ra lang_pair_array <<< ${lang_pairs}

for lang_pair in "${lang_pair_array[@]}"
do

    IFS='-' read -ra langs <<< ${lang_pair}

    SRC_LANG=${langs[0]}
    TGT_LANG=${langs[1]}

    echo ""
    echo "=================================================="
    echo "[INFO] Translating ${SRC_LANG} -> ${TGT_LANG}"
    echo "=================================================="

    OUT_PREFIX=${OUTPUT_DIR}/${SRC_LANG}2${TGT_LANG}

    $CMD ../fairseq_cli/generate.py \
        ${DATA_BIN} \
        \
        --user-dir ${USER_DIR} \
        \
        --task translation_multi_simple_epoch \
        \
        --path ${CKPT_PATH} \
        \
        --langs ${lang_list} \
        --lang-pairs ${lang_pairs} \
        \
        --source-lang ${SRC_LANG} \
        --target-lang ${TGT_LANG} \
        \
        --decoder-langtok \
        --lang-tok-replacing-bos-eos \
        \
        --gen-subset test \
        \
        --beam 4 \
        --lenpen 0.6 \
        \
        --batch-size 32 \
        --max-tokens 2048 \
        \
        --remove-bpe sentencepiece \
        \
        --sacrebleu \
        \
        --fp16 \
        \
        --ddp-backend=no_c10d \
        \
        > ${OUT_PREFIX}.log

    # =====================================================
    # Extract hypotheses
    # =====================================================

    grep "^H" ${OUT_PREFIX}.log \
        | sort -V \
        | cut -f3- \
        > ${OUT_PREFIX}.hyp

    # =====================================================
    # Extract references
    # =====================================================

    grep "^T" ${OUT_PREFIX}.log \
        | sort -V \
        | cut -f2- \
        > ${OUT_PREFIX}.ref

    # =====================================================
    # BLEU
    # =====================================================

    echo ""
    echo "[INFO] BLEU SCORE (${SRC_LANG}->${TGT_LANG})"

    sacrebleu \
        ${OUT_PREFIX}.ref \
        -i ${OUT_PREFIX}.hyp \
        -l ${SRC_LANG}-${TGT_LANG}

    echo ""
    echo "[INFO] Outputs saved:"
    echo "  Hypothesis : ${OUT_PREFIX}.hyp"
    echo "  Reference  : ${OUT_PREFIX}.ref"
    echo "  Full log   : ${OUT_PREFIX}.log"

done

echo ""
echo "[INFO] =========================================="
echo "[INFO] Validation inference completed successfully."
echo "[INFO] =========================================="
