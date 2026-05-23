# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fairseq import utils

# UPDATED IMPORTS
#from fairseq.modules import (
#    LayerNorm,
#    MultiheadAttention,
#    FastMultiheadAttention,
#    FastGroupedQueryAttention,
#)

from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.multihead_attention import MultiheadAttention

from fairseq.modules.rms_norm import RMSNorm
from fairseq.modules.fast_multihead_attention import FastMultiheadAttention
from fairseq.modules.fast_grouped_query_attention import FastGroupedQueryAttention

from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from torch import Tensor


# =========================================================
# HELPER
# =========================================================

def get_norm_fn(use_rmsnorm=False):
    return RMSNorm if use_rmsnorm else LayerNorm


# =========================================================
# ENCODER
# =========================================================

class TransformerEncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.embed_dim = args.encoder_embed_dim

        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(
            args,
            'quant_noise_pq_block_size',
            8
        ) or 8

        self.attn_implementation = getattr(
            args,
            "attn_implementation",
            "fairseq"
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args
        )

        norm_fn = get_norm_fn(
            getattr(args, "use_rmsnorm", False)
        )

        self.self_attn_layer_norm = norm_fn(self.embed_dim)

        self.dropout_module = FairseqDropout(
            args.dropout,
            module_name=self.__class__.__name__
        )

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )

        activation_dropout_p = getattr(
            args,
            "activation_dropout",
            0
        ) or 0

        if activation_dropout_p == 0:
            activation_dropout_p = getattr(
                args,
                "relu_dropout",
                0
            ) or 0

        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p),
            module_name=self.__class__.__name__
        )

        self.normalize_before = args.encoder_normalize_before

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = norm_fn(self.embed_dim)

    # -----------------------------------------------------

    def build_fc1(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size
    ):
        return quant_noise(
            nn.Linear(input_dim, output_dim),
            p=q_noise,
            block_size=qn_block_size
        )

    def build_fc2(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size
    ):
        return quant_noise(
            nn.Linear(input_dim, output_dim),
            p=q_noise,
            block_size=qn_block_size
        )

    # -----------------------------------------------------
    # UPDATED ATTENTION
    # -----------------------------------------------------

    def build_self_attention(self, embed_dim, args):

        attn_impl = getattr(
            args,
            "attn_implementation",
            "fairseq"
        )

        rope_args = {
            "rope_theta": getattr(args, "rope_theta", 10000.0)
        }

        # ---------------- GQA ----------------

        if attn_impl.startswith("fast_gqa"):

            kv_heads = getattr(
                args,
                "encoder_kv_attention_heads",
                max(1, args.encoder_attention_heads // 4)
            )

            return  FastGroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            kv_heads=args.encoder_kv_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

        # ---------------- FAST MHA ----------------

        elif attn_impl.startswith("fast"):

            return FastMultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                fused_qkv=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                rope_args=rope_args,
            )

        # ---------------- STANDARD ----------------

        else:

            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    # -----------------------------------------------------

    def residual_connection(self, x, residual):
        return residual + x

    # -----------------------------------------------------

    def forward(
        self,
        x,
        encoder_padding_mask,
        attn_mask: Optional[Tensor] = None
    ):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8
            )

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x

        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x


# =========================================================
# DECODER
# =========================================================

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False
    ):
        super().__init__()

        self.embed_dim = args.decoder_embed_dim

        self.dropout_module = FairseqDropout(
            args.dropout,
            module_name=self.__class__.__name__
        )

        self.quant_noise = getattr(args, "quant_noise_pq", 0)

        self.quant_noise_block_size = getattr(
            args,
            "quant_noise_pq_block_size",
            8
        )

        self.cross_self_attention = getattr(
            args,
            "cross_self_attention",
            False
        )

        self.attn_implementation = getattr(
            args,
            "attn_implementation",
            "fairseq"
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )

        activation_dropout_p = getattr(
            args,
            "activation_dropout",
            0
        ) or 0

        if activation_dropout_p == 0:
            activation_dropout_p = getattr(
                args,
                "relu_dropout",
                0
            ) or 0

        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p),
            module_name=self.__class__.__name__
        )

        self.normalize_before = args.decoder_normalize_before

        norm_fn = get_norm_fn(
            getattr(args, "use_rmsnorm", False)
        )

        self.self_attn_layer_norm = norm_fn(self.embed_dim)

        # -------------------------------------------------

        if no_encoder_attn:

            self.encoder_attn = None
            self.encoder_attn_layer_norm = None

        else:

            self.encoder_attn = self.build_encoder_attention(
                self.embed_dim,
                args
            )

            self.encoder_attn_layer_norm = norm_fn(
                self.embed_dim
            )

        # -------------------------------------------------

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = norm_fn(self.embed_dim)

        self.need_attn = True
        self.onnx_trace = False

    # -----------------------------------------------------

    def build_fc1(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size
    ):
        return quant_noise(
            nn.Linear(input_dim, output_dim),
            q_noise,
            qn_block_size
        )

    def build_fc2(
        self,
        input_dim,
        output_dim,
        q_noise,
        qn_block_size
    ):
        return quant_noise(
            nn.Linear(input_dim, output_dim),
            q_noise,
            qn_block_size
        )

    # =====================================================
    # SELF ATTENTION
    # =====================================================

    def build_self_attention(
        self,
        embed_dim,
        args,
        add_bias_kv=False,
        add_zero_attn=False
    ):

        attn_impl = getattr(
            args,
            "attn_implementation",
            "fairseq"
        )

        rope_args = {
            "rope_theta": getattr(args, "rope_theta", 10000.0)
        }

        # ---------------- GQA ----------------

        if attn_impl.startswith("fast_gqa"):

            kv_heads = getattr(
                args,
                "decoder_kv_attention_heads",
                max(1, args.decoder_attention_heads // 4)
            )

            return FastGroupedQueryAttention(
                embed_dim,
                args.decoder_attention_heads,
                kv_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                is_decoder=True,
                fused_qkv=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                rope_args=rope_args,
            )

        # ---------------- FAST ----------------

        elif attn_impl.startswith("fast"):

            return FastMultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                is_decoder=True,
                fused_qkv=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                rope_args=rope_args,
            )

        # ---------------- STANDARD ----------------

        else:

            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    # =====================================================
    # CROSS ATTENTION
    # =====================================================

    def build_encoder_attention(self, embed_dim, args):

        
        #Encoder Decoder Cross Attention is still MHA
        attn_impl = getattr(
            args,
            "cross-attention-implementation",
            "fairseq"
        )

        if attn_impl.startswith("fast"):

            return FastMultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                is_decoder=True,
                fused_qkv=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

        else:

            return MultiheadAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
            )

    # =====================================================

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    # =====================================================

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[Tensor]]]
        ] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):

        if need_head_weights:
            need_attn = True

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # -------------------------------------------------
        # ENCODER ATTENTION
        # -------------------------------------------------

        if self.encoder_attn is not None and encoder_out is not None:

            residual = x

            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn,
                need_head_weights=need_head_weights,
            )

            x = self.dropout_module(x)

            x = self.residual_connection(x, residual)

            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        # -------------------------------------------------

        residual = x

        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))

        x = self.activation_dropout_module(x)

        x = self.fc2(x)

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None

    # =====================================================

    def make_generation_fast_(
        self,
        need_attn: bool = False,
        **kwargs
    ):
        self.need_attn = need_attn
