# Copyright (c) Facebook, Inc. and its affiliates.
#
# Modified for:
# - GQA (Grouped Query Attention)
# - RMSNorm
# - RoPE
# - LLaMA-style encoder MLM
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)

from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)

from omegaconf import II


DEFAULT_MAX_SOURCE_POSITIONS = 1024


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TransformerEncoderModelConfig(FairseqDataclass):

    # -----------------------------------------------------
    # CORE
    # -----------------------------------------------------

    activation_fn: ChoiceEnum(
        utils.get_available_activation_fns()
    ) = field(
        default="gelu"
    )

    dropout: float = field(default=0.1)

    attention_dropout: float = field(default=0.1)

    activation_dropout: float = field(default=0.1)

    relu_dropout: float = field(default=0.0)

    # -----------------------------------------------------
    # ENCODER
    # -----------------------------------------------------

    encoder_embed_dim: int = field(default=1024)

    encoder_output_dim: int = field(default=1024)

    encoder_input_dim: int = field(default=1024)

    encoder_ffn_embed_dim: int = field(default=4096)

    encoder_layers: int = field(default=6)

    encoder_attention_heads: int = field(default=16)

    # -----------------------------------------------------
    # GQA
    # -----------------------------------------------------

    encoder_kv_attention_heads: int = field(default=4)

    # -----------------------------------------------------
    # RMSNORM
    # -----------------------------------------------------

    use_rmsnorm: bool = field(default=False)

    # -----------------------------------------------------
    # ROPE
    # -----------------------------------------------------

    use_rope: bool = field(default=False)

    rope_theta: float = field(default=10000.0)

    # -----------------------------------------------------
    # ATTENTION
    # -----------------------------------------------------

    attn_implementation: str = field(
        default="fast_gqa"
    )

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    encoder_normalize_before: bool = field(
        default=True
    )

    no_encoder_final_norm: bool = field(
        default=False
    )

    # -----------------------------------------------------
    # POSITIONAL
    # -----------------------------------------------------

    no_token_positional_embeddings: bool = field(
        default=True
    )

    encoder_learned_pos: bool = field(
        default=False
    )

    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    share_encoder_input_output_embed: bool = field(
        default=False
    )

    no_scale_embedding: bool = field(
        default=False
    )

    layernorm_embedding: bool = field(
        default=False
    )

    # -----------------------------------------------------
    # ADAPTIVE SOFTMAX
    # -----------------------------------------------------

    adaptive_softmax_cutoff: Optional[str] = field(
        default=None
    )

    adaptive_softmax_dropout: float = field(
        default=0
    )

    adaptive_softmax_factor: float = field(
        default=4
    )

    # -----------------------------------------------------
    # ADAPTIVE INPUT
    # -----------------------------------------------------

    adaptive_input: bool = field(
        default=False
    )

    adaptive_input_factor: float = field(
        default=4
    )

    adaptive_input_cutoff: Optional[str] = field(
        default=None
    )

    tie_adaptive_weights: bool = field(
        default=False
    )

    tie_adaptive_proj: bool = field(
        default=False
    )

    # -----------------------------------------------------
    # CHARACTER EMBEDDINGS
    # -----------------------------------------------------

    character_embeddings: bool = field(
        default=False
    )

    character_filters: str = field(
        default="[(1,64),(2,128),(3,192)]"
    )

    character_embedding_dim: int = field(
        default=4
    )

    char_embedder_highway_layers: int = field(
        default=2
    )

    # -----------------------------------------------------
    # LAYERDROP
    # -----------------------------------------------------

    encoder_layerdrop: float = field(
        default=0.0
    )

    encoder_layers_to_keep: Optional[str] = field(
        default=None
    )

    # -----------------------------------------------------
    # QUANTIZATION
    # -----------------------------------------------------

    quant_noise_pq: float = field(
        default=0.0
    )

    quant_noise_pq_block_size: int = field(
        default=8
    )

    quant_noise_scalar: float = field(
        default=0.0
    )

    # -----------------------------------------------------
    # ACTIVATION CHECKPOINTING
    # -----------------------------------------------------

    checkpoint_activations: bool = field(
        default=False
    )

    offload_activations: bool = field(
        default=False
    )

    # -----------------------------------------------------
    # TASK
    # -----------------------------------------------------

    add_bos_token: bool = II("task.add_bos_token")

    tokens_per_sample: int = II("task.tokens_per_sample")

    max_source_positions: Optional[int] = II(
        "task.max_source_positions"
    )

    tpu: bool = II("common.tpu")


# =========================================================
# MODEL
# =========================================================

@register_model(
    "encoder_mlm",
    dataclass=TransformerEncoderModelConfig
)
class TransformerEncoderModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):

        encoder_mlm_gqa(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(
                args.encoder_layers_to_keep.split(",")
            )

        if getattr(args, "max_source_positions", None) is None:

            args.max_source_positions = getattr(
                args,
                "tokens_per_sample",
                DEFAULT_MAX_SOURCE_POSITIONS,
            )

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------

        if args.character_embeddings:

            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.encoder_embed_dim,
                args.char_embedder_highway_layers,
            )

        elif args.adaptive_input:

            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.encoder_input_dim,
                args.adaptive_input_factor,
                args.encoder_embed_dim,
                options.eval_str_list(
                    args.adaptive_input_cutoff,
                    type=int
                ),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )

        else:

            embed_tokens = cls.build_embedding(
                args,
                task.source_dictionary,
                args.encoder_input_dim,
            )

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------

        encoder = TransformerEncoder(
            args,
            task.source_dictionary,
            embed_tokens,
        )

        # -------------------------------------------------
        # MLM HEAD
        # -------------------------------------------------

        encoder.output_projection = nn.Linear(
            encoder.embed_tokens.weight.shape[1],
            encoder.embed_tokens.weight.shape[0],
            bias=False,
        )

        if args.share_encoder_input_output_embed:

            encoder.output_projection.weight = (
                encoder.embed_tokens.weight
            )

        else:

            nn.init.normal_(
                encoder.output_projection.weight,
                mean=0,
                std=encoder.embed_tokens.weight.shape[1] ** -0.5,
            )

        return cls(encoder)

    @classmethod
    def build_embedding(
        cls,
        args,
        dictionary,
        embed_dim,
        path=None,
    ):

        return Embedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
        )

    def get_normalized_probs(
        self,
        net_output,
        log_probs,
        sample=None,
    ):

        encoder_out = net_output["encoder_out"][0]

        encoder_out = encoder_out.transpose(0, 1)

        encoder_out = self.encoder.output_projection(
            encoder_out
        )

        logits = encoder_out.float()

        if log_probs:
            return F.log_softmax(logits, dim=-1)

        return F.softmax(logits, dim=-1)


# =========================================================
# BASE ARCH
# =========================================================

def base_encoder_model_architecture(args):

    # -----------------------------------------------------
    # CORE
    # -----------------------------------------------------

    args.dropout = getattr(args, "dropout", 0.1)

    args.attention_dropout = getattr(
        args,
        "attention_dropout",
        0.1,
    )

    args.activation_dropout = getattr(
        args,
        "activation_dropout",
        0.1,
    )

    args.activation_fn = getattr(
        args,
        "activation_fn",
        "gelu",
    )

    # -----------------------------------------------------
    # ENCODER
    # -----------------------------------------------------

    args.encoder_embed_dim = getattr(
        args,
        "encoder_embed_dim",
        1024,
    )

    args.encoder_output_dim = getattr(
        args,
        "encoder_output_dim",
        args.encoder_embed_dim,
    )

    args.encoder_input_dim = getattr(
        args,
        "encoder_input_dim",
        args.encoder_embed_dim,
    )

    args.encoder_ffn_embed_dim = getattr(
        args,
        "encoder_ffn_embed_dim",
        4096,
    )

    args.encoder_layers = getattr(
        args,
        "encoder_layers",
        6,
    )

    args.encoder_attention_heads = getattr(
        args,
        "encoder_attention_heads",
        16,
    )

    # -----------------------------------------------------
    # GQA
    # -----------------------------------------------------

    args.encoder_kv_attention_heads = getattr(
        args,
        "encoder_kv_attention_heads",
        4,
    )

    # -----------------------------------------------------
    # RMSNORM
    # -----------------------------------------------------

    args.use_rmsnorm = getattr(
        args,
        "use_rmsnorm",
        True,
    )

    # -----------------------------------------------------
    # ROPE
    # -----------------------------------------------------

    args.use_rope = getattr(
        args,
        "use_rope",
        True,
    )

    args.rope_theta = getattr(
        args,
        "rope_theta",
        10000.0,
    )

    # -----------------------------------------------------
    # ATTENTION
    # -----------------------------------------------------

    args.attn_implementation = getattr(
        args,
        "attn_implementation",
        "fast_gqa",
    )

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    args.encoder_normalize_before = getattr(
        args,
        "encoder_normalize_before",
        True,
    )

    args.no_encoder_final_norm = getattr(
        args,
        "no_encoder_final_norm",
        False,
    )

    # -----------------------------------------------------
    # POSITIONAL
    # -----------------------------------------------------

    args.no_token_positional_embeddings = getattr(
        args,
        "no_token_positional_embeddings",
        True,
    )

    args.encoder_learned_pos = getattr(
        args,
        "encoder_learned_pos",
        False,
    )

    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

    args.share_encoder_input_output_embed = getattr(
        args,
        "share_encoder_input_output_embed",
        False,
    )

    args.no_scale_embedding = getattr(
        args,
        "no_scale_embedding",
        False,
    )

    args.layernorm_embedding = getattr(
        args,
        "layernorm_embedding",
        False,
    )

    # -----------------------------------------------------
    # ADAPTIVE SOFTMAX
    # -----------------------------------------------------

    args.adaptive_softmax_cutoff = getattr(
        args,
        "adaptive_softmax_cutoff",
        None,
    )

    args.adaptive_softmax_dropout = getattr(
        args,
        "adaptive_softmax_dropout",
        0,
    )

    args.adaptive_softmax_factor = getattr(
        args,
        "adaptive_softmax_factor",
        4,
    )

    # -----------------------------------------------------
    # ADAPTIVE INPUT
    # -----------------------------------------------------

    args.adaptive_input = getattr(
        args,
        "adaptive_input",
        False,
    )

    args.adaptive_input_factor = getattr(
        args,
        "adaptive_input_factor",
        4,
    )

    args.adaptive_input_cutoff = getattr(
        args,
        "adaptive_input_cutoff",
        None,
    )

    args.tie_adaptive_weights = getattr(
        args,
        "tie_adaptive_weights",
        False,
    )

    args.tie_adaptive_proj = getattr(
        args,
        "tie_adaptive_proj",
        False,
    )

    # -----------------------------------------------------
    # CHARACTER
    # -----------------------------------------------------

    args.character_embeddings = getattr(
        args,
        "character_embeddings",
        False,
    )

    # -----------------------------------------------------
    # LAYERDROP
    # -----------------------------------------------------

    args.encoder_layerdrop = getattr(
        args,
        "encoder_layerdrop",
        0.0,
    )

    args.encoder_layers_to_keep = getattr(
        args,
        "encoder_layers_to_keep",
        None,
    )

    # -----------------------------------------------------
    # QUANT NOISE
    # -----------------------------------------------------

    args.quant_noise_pq = getattr(
        args,
        "quant_noise_pq",
        0.0,
    )

    args.quant_noise_pq_block_size = getattr(
        args,
        "quant_noise_pq_block_size",
        8,
    )

    args.quant_noise_scalar = getattr(
        args,
        "quant_noise_scalar",
        0.0,
    )

    # -----------------------------------------------------
    # CHECKPOINTING
    # -----------------------------------------------------

    args.checkpoint_activations = getattr(
        args,
        "checkpoint_activations",
        False,
    )

    args.offload_activations = getattr(
        args,
        "offload_activations",
        False,
    )

    if args.offload_activations:
        args.checkpoint_activations = True

    # -----------------------------------------------------
    # MAX POSITIONS
    # -----------------------------------------------------

    args.max_source_positions = getattr(
        args,
        "max_source_positions",
        1024,
    )


# =========================================================
# ARCH
# =========================================================

@register_model_architecture(
    "encoder_mlm",
    "encoder_mlm_gqa"
)
def encoder_mlm_gqa(args):

    args.encoder_layers = getattr(
        args,
        "encoder_layers",
        6,
    )

    args.encoder_embed_dim = getattr(
        args,
        "encoder_embed_dim",
        1024,
    )

    args.encoder_ffn_embed_dim = getattr(
        args,
        "encoder_ffn_embed_dim",
        4096,
    )

    args.encoder_attention_heads = getattr(
        args,
        "encoder_attention_heads",
        16,
    )

    args.encoder_kv_attention_heads = getattr(
        args,
        "encoder_kv_attention_heads",
        4,
    )

    args.activation_fn = getattr(
        args,
        "activation_fn",
        "gelu",
    )

    args.use_rmsnorm = getattr(
        args,
        "use_rmsnorm",
        True,
    )

    args.use_rope = getattr(
        args,
        "use_rope",
        True,
    )

    args.rope_theta = getattr(
        args,
        "rope_theta",
        10000.0,
    )

    args.attn_implementation = getattr(
        args,
        "attn_implementation",
        "fast_gqa",
    )

    args.encoder_normalize_before = True

    args.no_token_positional_embeddings = True

    base_encoder_model_architecture(args)
