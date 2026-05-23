# Copyright (c) Facebook, Inc. and its affiliates.
#
# Modified for:
# - GQA (Grouped Query Attention)
# - RMSNorm
# - RoPE
# - LLaMA-style Bridge Transformer
#

import logging
import re
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    base_architecture,
    register_model,
)

from fairseq.modules import LayerNorm
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


# =========================================================
# MODEL
# =========================================================

@register_model("llama_bridge_transformer_model")
class LLaMABridgeTransformerModel(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        if args.freeze_params is not None:
            self.freeze_params(args)

    # -----------------------------------------------------

    def freeze_params(self, args):

        freeze_pattern = re.compile(args.freeze_params)

        for name, parameter in self.named_parameters():

            if freeze_pattern.search(name):
                parameter.requires_grad = False
                logger.info(f"Freeze: {name}")

        for name, parameter in self.named_parameters():

            if not freeze_pattern.search(name):
                logger.info(f"Unfreeze: {name}")

    # -----------------------------------------------------

    @staticmethod
    def add_args(parser):

        TransformerModel.add_args(parser)

        # -------------------------------------------------
        # BRIDGE
        # -------------------------------------------------

        parser.add_argument(
            "--no-encoder-attn-layers",
            type=str,
            default=None,
        )

        parser.add_argument(
            "--freeze-params",
            type=str,
            default=None,
        )

        parser.add_argument(
            "--transfer-params",
            type=str,
            default=None,
        )

        parser.add_argument(
            "--lm-fusion",
            action="store_true",
            default=False,
        )

        # -------------------------------------------------
        # GQA
        # -------------------------------------------------

        parser.add_argument(
            "--attn-implementation",
            type=str,
            default="fast_gqa",
        )
        
        parser.add_argument(
              "--cross-attention-implementation",
              type=str,
              default="mha",
          )

        parser.add_argument(
            "--encoder-kv-attention-heads",
            type=int,
            default=4,
        )

        parser.add_argument(
            "--decoder-kv-attention-heads",
            type=int,
            default=4,
        )

        # -------------------------------------------------
        # RMSNORM
        # -------------------------------------------------

        parser.add_argument(
            "--use-rmsnorm",
            action="store_true",
        )

        # -------------------------------------------------
        # ROPE
        # -------------------------------------------------

        parser.add_argument(
            "--use-rope",
            action="store_true",
        )

        parser.add_argument(
            "--rope-theta",
            type=float,
            default=10000.0,
        )

    # -----------------------------------------------------

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):

        return LLaMABridgeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(
                args,
                "no_cross_attention",
                False
            ),
        )

    # -----------------------------------------------------

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):

        if (
            self.args.transfer_params is not None
            and "inference" not in vars(model_cfg)
        ):

            pretrained_model_prefix = [*state_dict][0].split('.')[0]

            pairs = self.args.transfer_params.split(',')

            for pair in pairs:

                from_param, to_param = pair.split(':')

                if from_param in state_dict:

                    state_dict[to_param] = state_dict[from_param]

                    logger.info(
                        f"Transfer {from_param} "
                        f"to {to_param} "
                        f"in model [{pretrained_model_prefix}]"
                    )

        return torch.nn.Module.load_state_dict(
            self,
            state_dict,
            strict=False
        )


# =========================================================
# DECODER
# =========================================================

class LLaMABridgeTransformerDecoder(TransformerDecoder):

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False
    ):

        super().__init__(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn
        )

        no_encoder_attn_layers = (
            args.no_encoder_attn_layers.split(',')
            if args.no_encoder_attn_layers is not None
            else []
        )

        self.layers = nn.ModuleList([])

        self.layers.extend(
            [
                self.build_decoder_layer(
                    args,
                    no_encoder_attn=True
                )
                if str(layer) in no_encoder_attn_layers
                else self.build_decoder_layer(
                    args,
                    no_encoder_attn=False
                )
                for layer in range(args.decoder_layers)
            ]
        )

        # -------------------------------------------------
        # LM FUSION
        # -------------------------------------------------

        self.lm_layer_norm = None
        self.lm_output_projection = None

        if args.lm_fusion:

            self.lm_layer_norm = LayerNorm(
                self.embed_tokens.embedding_dim
            )

            self.lm_output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )

            self.lm_output_projection.weight = (
                self.embed_tokens.weight
            )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[
            Dict[str, List[Tensor]]
        ] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[Tensor]]]
        ] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:

            x = self.output_layer(x)

            if self.lm_output_projection:

                lm_state = self.lm_output_projection(
                    extra["lm_state"]
                )

                x = x + lm_state

        return x, extra

    # =====================================================

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[
            Dict[str, List[Tensor]]
        ],
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[Tensor]]]
        ] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    # =====================================================

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[
            Dict[str, List[Tensor]]
        ],
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[Tensor]]]
        ] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # -------------------------------------------------
        # POSITIONS
        # -------------------------------------------------

        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:

            prev_output_tokens = prev_output_tokens[:, -1:]

            if positions is not None:
                positions = positions[:, -1:]

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------

        x = self.embed_scale * self.embed_tokens(
            prev_output_tokens
        )

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C

        x = x.transpose(0, 1)

        # -------------------------------------------------
        # PADDING MASK
        # -------------------------------------------------

        self_attn_padding_mask: Optional[Tensor] = None

        if (
            self.cross_self_attention
            or prev_output_tokens.eq(
                self.padding_idx
            ).any()
        ):
            self_attn_padding_mask = (
                prev_output_tokens.eq(self.padding_idx)
            )

        # -------------------------------------------------
        # DECODER LAYERS
        # -------------------------------------------------

        attn: Optional[Tensor] = None

        inner_states: List[Optional[Tensor]] = [x]

        lm_state = None

        for idx, layer in enumerate(self.layers):

            if (
                incremental_state is None
                and not full_context_alignment
            ):
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_out"]) > 0
                )
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(
                        encoder_out[
                            "encoder_padding_mask"
                        ]
                    ) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=(
                    self_attn_padding_mask
                ),
                need_attn=bool(
                    idx == alignment_layer
                ),
                need_head_weights=bool(
                    idx == alignment_layer
                ),
            )

            inner_states.append(x)

            if (
                layer_attn is not None
                and idx == alignment_layer
            ):
                attn = layer_attn.float().to(x)

            if (
                self.layers[idx].encoder_attn is None
                and idx + 1 < len(self.layers)
                and self.layers[
                    idx + 1
                ].encoder_attn is not None
            ):
                lm_state = x

        # -------------------------------------------------
        # ALIGNMENT
        # -------------------------------------------------

        if attn is not None:

            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            attn = attn.mean(dim=0)

        # -------------------------------------------------
        # FINAL NORM
        # -------------------------------------------------

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C

        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # -------------------------------------------------
        # LM STATE
        # -------------------------------------------------

        if (
            self.lm_layer_norm is not None
            and lm_state is not None
        ):

            lm_state = self.lm_layer_norm(lm_state)

            lm_state = lm_state.transpose(0, 1)

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "lm_state": lm_state,
        }

    # =====================================================

    def get_normalized_probs(
        self,
        net_output: Tuple[
            Tensor,
            Optional[
                Dict[
                    str,
                    List[Optional[Tensor]]
                ]
            ],
        ],
        log_probs: bool,
        sample: Optional[
            Dict[str, Tensor]
        ] = None,
    ):

        logits = net_output[0]

        if log_probs:

            return utils.log_softmax(
                logits,
                dim=-1,
                onnx_trace=self.onnx_trace,
            )

        return utils.softmax(
            logits,
            dim=-1,
            onnx_trace=self.onnx_trace,
        )


# =========================================================
# ARCHITECTURE
# =========================================================

@register_model_architecture(
    "llama_bridge_transformer_model",
    "llama_bridge_transformer",
)
def llama_bridge_transformer(args):

    # -------------------------------------------------
    # ENCODER
    # -------------------------------------------------

    args.encoder_layers = getattr(
        args,
        "encoder_layers",
        12,
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

    # -------------------------------------------------
    # DECODER
    # -------------------------------------------------

    args.decoder_layers = getattr(
        args,
        "decoder_layers",
        12,
    )

    args.decoder_embed_dim = getattr(
        args,
        "decoder_embed_dim",
        1024,
    )

    args.decoder_ffn_embed_dim = getattr(
        args,
        "decoder_ffn_embed_dim",
        4096,
    )

    args.decoder_attention_heads = getattr(
        args,
        "decoder_attention_heads",
        16,
    )

    # -------------------------------------------------
    # GQA
    # -------------------------------------------------

    args.encoder_kv_attention_heads = getattr(
        args,
        "encoder_kv_attention_heads",
        4,
    )

    args.decoder_kv_attention_heads = getattr(
        args,
        "decoder_kv_attention_heads",
        4,
    )

    # -------------------------------------------------
    # RMSNORM
    # -------------------------------------------------

    args.use_rmsnorm = getattr(
        args,
        "use_rmsnorm",
        True,
    )

    # -------------------------------------------------
    # ROPE
    # -------------------------------------------------

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

    # -------------------------------------------------
    # ATTENTION BACKEND
    # -------------------------------------------------

    args.attn_implementation = getattr(
        args,
        "attn_implementation",
        "fast_gqa",
    )
    
    
    

    # -------------------------------------------------
    # PRENORM
    # -------------------------------------------------

    args.encoder_normalize_before = getattr(
        args,
        "encoder_normalize_before",
        True,
    )

    args.decoder_normalize_before = getattr(
        args,
        "decoder_normalize_before",
        True,
    )

    # -------------------------------------------------
    # POSITIONAL
    # -------------------------------------------------

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

    args.decoder_learned_pos = getattr(
        args,
        "decoder_learned_pos",
        False,
    )

    # -------------------------------------------------
    # ACTIVATION
    # -------------------------------------------------

    args.activation_fn = getattr(
        args,
        "activation_fn",
        "gelu",
    )

    # -------------------------------------------------
    # DROPOUT
    # -------------------------------------------------

    args.dropout = getattr(
        args,
        "dropout",
        0.1,
    )

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

    # -------------------------------------------------
    # SHARE EMBEDDINGS
    # -------------------------------------------------

    args.share_decoder_input_output_embed = getattr(
        args,
        "share_decoder_input_output_embed",
        True,
    )

    # -------------------------------------------------

    base_architecture(args)
