# GPT-Based Neural Machine Translation using Graformer


# Overview

This repository presents a **Fairseq-based GPT Neural Machine Translation (NMT)** framework using a multi-stage training pipeline involving:

- **BERT-style Encoder Pretraining**
- **GPT-style Decoder Pretraining**
- **Graformer-based NMT Training**

The system integrates bidirectional contextual encoding with autoregressive generation for efficient and high-quality machine translation.

The implementation is built on top of the Fairseq framework with several modern Transformer optimizations including:
BERT Encoder:
    + Grouped Query Attention (GQA)

GPT Decoder:
    Grouped Query Attention (GQA)
    + RoPE
    + RMSNorm
---

# Architecture Pipeline

```text
Stage 1A:
Monolingual Corpus
        ↓
BERT-style Encoder Pretraining with MLM -> Pretrained Encoder

Stage 1B:
Monolingual Corpus
        ↓
GPT-style Decoder Pretraining -> Pretrained Decoder

Stage 2:
Parallel Translation Corpus
        ↓
Graformer NMT Training  -> (Encoder + Decoder Integration) to obtain Final Translation Model
```

---

# Key Features

- Fairseq-based implementation
- BERT-style contextual encoder
- GPT-style autoregressive decoder
- Graformer encoder-decoder integration
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm normalization

---


# Dataset Directory Structure

The project uses both:

- **Monolingual datasets** for encoder and decoder pretraining
- **Parallel translation datasets** for Graformer NMT training


All datasets are preprocessed using **preprocess.sh** file and stored in Fairseq binary format inside the `data-bin/` directory.

---

# Complete Dataset Structure

```text
data-bin/
│
├── hi/                                      # Hindi monolingual corpus used in Pretaining
│   ├── train.bin
│   ├── train.idx
│   ├── valid.bin
│   └── valid.idx
│
├── mr/                                      # Marathi monolingual corpus used in Pretraining
│   ├── train.bin
│   ├── train.idx
│   ├── valid.bin
│   └── valid.idx
│
├── dict.txt                                 # Shared vocabulary (we used shared Vocab)
│
├── train.mr-hi.hi.bin                       # Parallel training target data
├── train.mr-hi.hi.idx
├── train.mr-hi.mr.bin                       # Parallel training source data
├── train.mr-hi.mr.idx
│
├── valid.mr-hi.hi.bin                       # Validation target data
├── valid.mr-hi.hi.idx
├── valid.mr-hi.mr.bin                       # Validation source data
├── valid.mr-hi.mr.idx
│
├── test.mr-hi.hi.bin                        # Test target data
├── test.mr-hi.hi.idx
├── test.mr-hi.mr.bin                        # Test source data
├── test.mr-hi.mr.idx
├── test.hi-mr.mr.idx (similaryly all other parallel files for hi-mr)
└── preprocess.log                           # Fairseq preprocessing logs
```

---

# Dataset Usage Across Training Stages

| Dataset Type | Purpose | Training Stage |
|---|---|---|
| `hi/` monolingual corpus | GPT Decoder /BERT pretraining | Stage 1 |
| `mr/` monolingual corpus | GPT Decoder /BERT pretraining | Stage 1 |
| `train.mr-hi.*` | NMT training | Stage 2 |
| `valid.mr-hi.*` | Validation during NMT training | Stage 2 |
| `test.mr-hi.*` | Translation evaluation | Stage 2 |

---

# Main Repository Structure

```text
GPT-NMT/
│
├── fairseq/                           # Fairseq framework
│
├── fairseq_cli/                       # Fairseq command line tools
│
├── examples/
│
├── run/                               # Training and inference scripts
│   ├── preprocess.sh (dataset creatiom)
│   ├── laama_encoder.sh (pretrain bert like encoder)
│   ├── laama_decoder.sh  (pretrain laama style GPT decoder)
│   ├── laama_bridgenmt.sh (train NMT model)
│   ├── gptnmt_laam_infer.sh (inference file)

├── checkpoints/
│   ├── encoder/
│   ├── decoder/
│   └── NMT/
│
├── models/
│   ├── laama_lm.py (GPT decoder with GQA)
│   ├── encoder_mlm.py ( BERT like with GQA )
    ├── laama_bridge_transformer_model.py (fusing encoder and decoder)
│ 
├── tasks/
│   └── new_multilingual_masked_lm.py (task used in masked language modeling)
    ├── multilingual_language_modeling.py (task used in language modeling)
    ├── translation_multi_simple_epoch.py (task used in translation)
│
├── criterions/
│   └── label_smoothed_cross_entropy.py
│
├── modules/
│   ├── fast_grouped_query_attention.py
│   ├── rotary_embedding.py
│   └── rotary_positional_embedding.py
    └── rms_norm.py
    └── transformer_layer.py
├── requirements.txt
└── README.md
```

---

# Stage 1A: BERT-style Encoder Pretraining

The encoder is pretrained independently using the Masked Language Modeling (MLM) objective to learn contextual bidirectional representations.

## Encoder Features

- Bidirectional Transformer encoder
- Context-aware semantic representations
- MLM training objective
- Deep contextual embeddings

## MLM Objective

\[
\mathcal{L}_{MLM}=-\sum_{i\in M}\log P(x_i|x_{\backslash M})
\]

---

# Stage 1B: GPT-style Decoder Pretraining

The decoder is pretrained using autoregressive next-token prediction.

## Decoder Features

- Decoder-only Transformer
- Causal self-attention
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm normalization
## Autoregressive Objective

\[
\mathcal{L}_{CLM}=-\sum_{t=1}^{T}\log P(x_t|x_{<t})
\]

---

# Stage 2: Graformer NMT Training

The pretrained encoder and decoder are integrated into the Graformer architecture with cross attention for translation training.

## Graformer Features

- Pretrained encoder-decoder fusion
- Cross-attention translation modeling
- Efficient autoregressive decoding
- Long-context translation support
- Beam-search generation

# Architectural Enhancements

| Component | Purpose |
|---|---|
| GQA | Reduces KV cache memory |
| RoPE | Improves positional modeling |
| RMSNorm | Stabilizes deep Transformer training |


---

# Hyperparameter Configuration
| Hyperparameter | Stage 1A: Encoder Pretraining | Stage 1B: Decoder Pretraining | Stage 2: Graformer NMT |
|---|---|---|---|
| Model Type | Transformer Encoder (MLM) | GPT-style Decoder LM | Encoder–Decoder Graformer |
| Objective | Masked Language Modeling | Autoregressive Causal LM | Multilingual Translation |
| Architecture | Encoder-only Transformer | Decoder-only Transformer | Grafted Transformer |
| Encoder Layers | 12 | — | 16 |
| Decoder Layers | — | 12 | 16 |
| Hidden Size | 768 | 768 | 768 |
| FFN Dimension | 3072 | 3072 | 3072 |
| Attention Heads | 12 | 12 | 12 |
| KV Heads (GQA) | 4 | 4 | 4 |
| Attention Mechanism | Fast GQA | Fast GQA | Fast GQA + Cross-Attention |
| Cross Attention | No | No | Yes |
| Max Sequence Length | 512 | 512 | 1000 Tokens |
| Positional Encoding | RoPE | RoPE | RoPE |
| Positional Embeddings | Disabled | Disabled | Disabled |
| Normalization | RMSNorm | RMSNorm | RMSNorm |
| Activation Function | GELU | GELU | GELU |
| Optimizer | Adam | Adam | Adam |
| Adam Betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| Weight Decay | 0.1 | — | 0.1 |
| Learning Rate | 2e-4 | 3e-4 | 3e-4 |
| Warmup Updates | 4000 | 1000 | 4000 |
| LR Scheduler | Inverse Square Root | Inverse Square Root | Inverse Square Root |
| Batching Strategy | Max Tokens = 4096 | Max Tokens = 4096 | Max Tokens = 1000 |
| Update Frequency | 8 | 4 | 1 |
| Epochs | 10 | 10 | 50000 Updates |
| Dropout | 0.1 | 0.2 | 0.2 |
| Attention Dropout | 0.1 | 0.1 | 0.1 |
| Activation Dropout | 0.1 | 0.1 | 0.1 |
| Gradient Clipping | 1.0 | 1.0 | 1.0 |
| Label Smoothing | 0.1 | — | 0.1 |
| Mixed Precision | FP16 | FP16 | FP16 |
| Language Setup | Multilingual (hi,mr) | Multilingual (hi,mr) | hi↔mr Translation |
| Frozen Layers | No | No | First 12 Layers Frozen |
| LM Fusion | No | No | Enabled |
| Shared Embeddings | Encoder I/O Shared | Decoder I/O Shared | All Embeddings Shared |
| Grafting Layers | — | — | Added Translation Bridge Layers |
| Cross-Attention Implementation | — | — | Multi-Head Attention (MHA) |
| Beam Size | — | — | Not Explicitly Defined |

# Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/GPT-NMT.git
cd GPT-NMT
```

---

## Create Environment

```bash
conda create -n graformer python=3.10
conda activate graformer
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---
# Dataset Preparation

Parallel corpus format:

```text
train.en
train.hi

valid.en
valid.hi

test.en
test.hi
```
Put the pointer to this folder in preprocess.sh file
---

# Preprocessing

```bash
bash run/preprocess.sh
```

This step:
- Cleans text
- Tokenizes corpus
- Learns a joint SentencePiece vocabulary of size 8000 shared across both languages.
- Converts datasets int o Fairseq binary format

# Encoder Pretraining

```bash
bash run/laama_gpt_encoder.sh```

---

# Decoder Pretraining

```bash
bash run/laama_gpt_decoder.sh
```

---

# Graformer NMT Training

```bash
bash run/laama_bridgenmt.sh
```

---

# Convergence Training 

<p align="center">
  <img 
    src="https://raw.githubusercontent.com/Review-MT/GPT124-NMT/main/train_vs_val_loss.png" 
    width="750"
  />
</p>

<p align="center">
  <em>Training vs Validation Loss Curve for GRAFORMER</em>
</p># Inference


<p align="center">
  <img 
    src="https://raw.githubusercontent.com/Review-MT/GPT124-NMT/main/train_vs_val_nmt1.png" 
    width="750"
  />
</p>

<p align="center">
  <em>Training vs Validation Loss Curve for Language Model</em>
</p># Inference



```bash
bash run/gptnmt_laam_infer.sh
```

---
---

# Example Translation

## Input


## Output
`

---

# Results

| Metric | Score |
|---|---|
| BLEU | XX.XX |
| ChrF++ | XX.XX |

---

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This work uses components from below with serveral modification:
- Graformer
- Fairseq
- PyTorch
- Hugging Face Transformers
- SentencePiece
- SacreBLEU# Example

