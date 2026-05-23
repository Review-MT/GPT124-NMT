from typing import Dict, Optional, Tuple
import json
import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from fairseq.modules.quant_noise import quant_noise
from torch.nn.functional import scaled_dot_product_attention
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules.rotary_embedding import RotaryEmbedding


class FastMultiheadAttention(MultiheadAttention):

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        is_decoder=False,
        rope_args=None,
        fused_qkv=False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # =====================================================
        # FIX 1: restore missing dropout scalar
        # =====================================================
        self.dropout_p = dropout

        # remove heavy module safely
        if hasattr(self, "dropout_module"):
            del self.dropout_module

        self.is_decoder = is_decoder
        self.fused_qkv = fused_qkv
        self.rope = rope_args is not None and self_attention

        # =====================================================
        # FIX 2: safe RoPE parsing
        # =====================================================
        if isinstance(rope_args, str):
            rope_args = json.loads(rope_args)
        elif rope_args is None:
            rope_args = {}

        self.rotary_pos_embed = (
            RotaryEmbedding(
                dim=self.head_dim // 2,
                theta=rope_args.get("theta", 10000),
                scaling_factor=rope_args.get("scaling_factor", 1.0),
            )
            if self.rope
            else None
        )

        # =====================================================
        # FUSED QKV (unchanged logic)
        # =====================================================
        if self.fused_qkv:
            if self_attention:
                del self.q_proj, self.k_proj, self.v_proj
                self.qkv_proj = quant_noise(
                    nn.Linear(embed_dim, 3 * embed_dim, bias=bias),
                    q_noise,
                    qn_block_size,
                )
            elif encoder_decoder_attention:
                del self.k_proj, self.v_proj
                self.kv_proj = quant_noise(
                    nn.Linear(embed_dim, 2 * embed_dim, bias=bias),
                    q_noise,
                    qn_block_size,
                )
            else:
                raise NotImplementedError(
                    "Fused QKV only supports self-attention or encoder-decoder attention."
                )

        self.reset_parameters()

    # =========================================================
    # ROTARY POSITION EMBEDDING
    # =========================================================
    def _apply_rotary_pos_emb(self, q, k, is_inference=False):
        offset = (k.shape[-2] - 1) if is_inference else 0
        q = self.rotary_pos_embed.rotate_queries_or_keys(q, offset=offset)
        k = self.rotary_pos_embed.rotate_queries_or_keys(k)
        return q, k

    # =========================================================
    # INIT
    # =========================================================
    def reset_parameters(self):
        if self.qkv_same_dim:
            if hasattr(self, "qkv_proj"):
                nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1 / math.sqrt(2))
            elif hasattr(self, "kv_proj"):
                nn.init.xavier_uniform_(self.kv_proj.weight, gain=1 / math.sqrt(2))
            else:
                nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
                nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    # =========================================================
    # FORWARD
    # =========================================================
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        need_weights: bool = False,
        static_kv: bool = False,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()
        dropout_p = self.dropout_p if self.training else 0.0

        # -------------------------
        # projection
        # -------------------------
        if self.self_attention:
            if not self.fused_qkv:
                q = self.q_proj(query)
                k = self.k_proj(query)
                v = self.v_proj(query)
            else:
                q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:
            q = self.q_proj(query)
            if key is None:
                k = v = None
            else:
                if not self.fused_qkv:
                    k = self.k_proj(key)
                    v = self.v_proj(key)
                else:
                    k, v = self.kv_proj(key).chunk(2, dim=-1)

        # -------------------------
        # reshape
        # -------------------------
        q = rearrange(q, "t b (h d) -> (b h) t d", h=self.num_heads, d=self.head_dim)

        if k is not None:
            k = rearrange(k, "t b (h d) -> (b h) t d", h=self.num_heads, d=self.head_dim)

        if v is not None:
            v = rearrange(v, "t b (h d) -> (b h) t d", h=self.num_heads, d=self.head_dim)

        # -------------------------
        # RoPE
        # -------------------------
        if self.rotary_pos_embed is not None and k is not None:
            q, k = self._apply_rotary_pos_emb(
                q, k, is_inference=incremental_state is not None
            )

        # -------------------------
        # attention mask
        # -------------------------
        combined_mask = None

        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .expand(-1, self.num_heads, tgt_len, -1)
                .reshape(bsz * self.num_heads, tgt_len, -1)
            )

            key_padding_mask = key_padding_mask.to(torch.float32) * torch.finfo(q.dtype).min
            combined_mask = key_padding_mask

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
            combined_mask = attn_mask if combined_mask is None else combined_mask + attn_mask

        # -------------------------
        # SDPA attention
        # -------------------------
        attn = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=combined_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        # -------------------------
        # output
        # -------------------------
        attn = rearrange(attn, "(b h) t d -> t b (h d)", h=self.num_heads, d=self.head_dim)
        attn = self.out_proj(attn)

        return attn, None