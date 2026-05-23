# fairseq/modules/fast_grouped_query_attention.py

import math
import json
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq import utils
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class FastGroupedQueryAttention(MultiheadAttention):
    """
    Minimal GQA implementation compatible with old Fairseq MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kv_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        rope=False,
        rope_theta=10000,
        rope_args=None,
        **kwargs,
    ):

        self.kv_heads = kv_heads

        # ---------------------------------------------------
        # IMPORTANT:
        # Parent class MUST think kdim/vdim == embed_dim
        # otherwise old Fairseq assertion breaks
        # ---------------------------------------------------

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        self.dropout_p = dropout

        # ---------------------------------------------------
        # Replace K/V projections for GQA
        # ---------------------------------------------------

        self.k_proj = quant_noise(
            nn.Linear(embed_dim, kv_heads * self.head_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        self.v_proj = quant_noise(
            nn.Linear(embed_dim, kv_heads * self.head_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        # q_proj remains full-size
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        # out proj
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        # ---------------------------------------------------
        # Rope
        # ---------------------------------------------------

        self.use_rope = rope
        self.rope_theta = rope_theta

        if rope_args is None:
            rope_args = {}

        if isinstance(rope_args, str):
            rope_args = json.loads(rope_args)

        self.rope_args = rope_args

    # =======================================================
    # Rotary embeddings
    # =======================================================

    def apply_rotary(self, x):
        # Minimal placeholder implementation
        # You can later replace with proper RoPE

        return x

    # =======================================================
    # Forward
    # =======================================================

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()

        if self.self_attention:

            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)

        elif self.encoder_decoder_attention:

            q = self.q_proj(query)

            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:

            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        src_len = k.size(0)

        # ===================================================
        # Reshape Q
        # ===================================================

        q = q.view(
            tgt_len,
            bsz,
            self.num_heads,
            self.head_dim,
        )

        q = q.permute(1, 2, 0, 3)

        # ===================================================
        # Reshape K/V
        # ===================================================

        k = k.view(
            src_len,
            bsz,
            self.kv_heads,
            self.head_dim,
        )

        v = v.view(
            src_len,
            bsz,
            self.kv_heads,
            self.head_dim,
        )

        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # ===================================================
        # Apply RoPE
        # ===================================================

        if self.use_rope:
            q = self.apply_rotary(q)
            k = self.apply_rotary(k)

        # ===================================================
        # Repeat KV heads
        # ===================================================

        repeat_factor = self.num_heads // self.kv_heads

        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # ===================================================
        # Flatten
        # ===================================================

        q = q.reshape(
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        )

        k = k.reshape(
            bsz * self.num_heads,
            src_len,
            self.head_dim,
        )

        v = v.reshape(
            bsz * self.num_heads,
            src_len,
            self.head_dim,
        )

        # ===================================================
        # Attention
        # ===================================================

        attn_weights = torch.bmm(
            q,
            k.transpose(1, 2),
        )

        # ===================================================
        # Mask
        # ===================================================

        if attn_mask is not None:

            attn_mask = attn_mask.unsqueeze(0)

            attn_weights += attn_mask

        if key_padding_mask is not None:

            attn_weights = attn_weights.view(
                bsz,
                self.num_heads,
                tgt_len,
                src_len,
            )

            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),
                float("-inf"),
            )

            attn_weights = attn_weights.view(
                bsz * self.num_heads,
                tgt_len,
                src_len,
            )

        # ===================================================
        # Softmax
        # ===================================================

        attn_weights_float = utils.softmax(
            attn_weights,
            dim=-1,
            onnx_trace=False,
        )

        attn_probs = F.dropout(
            attn_weights_float,
            p=self.dropout_p,
            training=self.training,
        )
                
        # FP16 compatibility
        attn_probs = attn_probs.type_as(v)

        # ===================================================
        # Attention output
        # ===================================================

        attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(
            tgt_len,
            bsz,
            embed_dim,
        )

        attn = self.out_proj(attn)

        attn_weights_out = None

        if need_weights:

            attn_weights_out = attn_weights_float.view(
                bsz,
                self.num_heads,
                tgt_len,
                src_len,
            ).transpose(1, 0)

            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out