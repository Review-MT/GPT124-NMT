#!/bin/bash

set -e

############################################
# PATHS
############################################

ROOT=/mnt/storage/divya/exam/raw

RAW=$ROOT
SPM=$ROOT/spm
DATA_BIN=$ROOT/data-bin
TMP=$ROOT/tmp_dict

mkdir -p $SPM
mkdir -p $DATA_BIN
mkdir -p $TMP
mkdir -p $DATA_BIN/hi
mkdir -p $DATA_BIN/mr

############################################
# STEP 1: TRAIN SHARED SENTENCEPIECE MODEL
############################################

echo "======================================="
echo "Training shared SentencePiece model"
echo "======================================="

cat \
    $RAW/clean.train.hi \
    $RAW/clean.train.mr \
    > $SPM/all_train.txt

python - <<EOF
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="$SPM/all_train.txt",
    model_prefix="$SPM/spm",
    vocab_size=8000,
    character_coverage=1.0,
    model_type="unigram"
)
EOF

############################################
# STEP 2: APPLY SENTENCEPIECE
############################################

echo "======================================="
echo "Applying SentencePiece"
echo "======================================="

python - <<EOF
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("$SPM/spm.model")

for split in ["train", "valid", "test"]:
    for lang in ["hi", "mr"]:

        inp = f"$RAW/clean.{split}.{lang}"
        out = f"$SPM/{split}.{lang}"

        with open(inp, "r", encoding="utf-8") as f_in, \
             open(out, "w", encoding="utf-8") as f_out:

            for line in f_in:
                pieces = sp.encode(line.strip(), out_type=str)
                f_out.write(" ".join(pieces) + "\n")
EOF

############################################
# STEP 3: BUILD SHARED DICTIONARY
############################################

echo "======================================="
echo "Building shared dictionary"
echo "======================================="

python -m fairseq_cli.preprocess  \
  --only-source \
  --trainpref $SPM/train.hi \
  --destdir $TMP

cp $TMP/dict.txt $DATA_BIN/dict.txt

############################################
# STEP 4: MONOLINGUAL LM DATA
############################################

echo "======================================="
echo "Preparing multilingual LM data"
echo "======================================="

############################################
# Hindi monolingual data
############################################

python -m fairseq_cli.preprocess \
  --only-source \
  --trainpref $SPM/train.hi \
  --validpref $SPM/valid.hi \
  --destdir $TMP/hi \
  --srcdict $DATA_BIN/dict.txt

cp $TMP/hi/train.bin $DATA_BIN/hi/train.bin
cp $TMP/hi/train.idx $DATA_BIN/hi/train.idx
cp $TMP/hi/valid.bin $DATA_BIN/hi/valid.bin
cp $TMP/hi/valid.idx $DATA_BIN/hi/valid.idx

############################################
# Marathi monolingual data
############################################

python -m fairseq_cli.preprocess \
  --only-source \
  --trainpref $SPM/train.mr \
  --validpref $SPM/valid.mr \
  --destdir $TMP/mr \
  --srcdict $DATA_BIN/dict.txt

cp $TMP/mr/train.bin $DATA_BIN/mr/train.bin
cp $TMP/mr/train.idx $DATA_BIN/mr/train.idx
cp $TMP/mr/valid.bin $DATA_BIN/mr/valid.bin
cp $TMP/mr/valid.idx $DATA_BIN/mr/valid.idx

############################################
# STEP 5: BILINGUAL NMT DATA
############################################

echo "======================================="
echo "Preparing bilingual NMT data"
echo "======================================="

python -m fairseq_cli.preprocess \
  --source-lang mr \
  --target-lang hi \
  --trainpref $SPM/train \
  --validpref $SPM/valid \
  --testpref $SPM/test \
  --destdir $DATA_BIN \
  --srcdict $DATA_BIN/dict.txt \
  --tgtdict $DATA_BIN/dict.txt

python -m fairseq_cli.preprocess \
  --source-lang hi \
  --target-lang mr \
  --trainpref $SPM/train \
  --validpref $SPM/valid \
  --testpref $SPM/test \
  --destdir $DATA_BIN \
  --srcdict $DATA_BIN/dict.txt \
  --tgtdict $DATA_BIN/dict.txt

############################################
# DONE
############################################

echo "======================================="
echo "ALL PREPROCESSING FINISHED"
echo "======================================="

echo "SentencePiece model:"
echo "$SPM/spm.model"

echo "Shared dictionary:"
echo "$DATA_BIN/dict.txt"

echo "Unified data-bin directory:"
echo "$DATA_BIN"
