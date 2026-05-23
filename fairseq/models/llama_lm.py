print("LOADING CUSTOM TRANSFORMER_LM")

from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)

from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)

from fairseq import options

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("llama_lm")
class TransformerLanguageModel(FairseqLanguageModel):

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):

        print("REGISTERING LLAMA ARGS")

        # =========================
        # STANDARD LM PARAMS
        # =========================

        parser.add_argument("--decoder-layers", type=int, default=24)

        parser.add_argument("--decoder-embed-dim", type=int, default=1024)

        parser.add_argument("--decoder-ffn-embed-dim", type=int, default=4096)

        parser.add_argument("--decoder-attention-heads", type=int, default=16)

        parser.add_argument("--dropout", type=float, default=0.1)

        parser.add_argument("--attention-dropout", type=float, default=0.1)

        parser.add_argument("--activation-dropout", type=float, default=0.1)

        parser.add_argument("--activation-fn", type=str, default="gelu")

        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
        )

        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
        )

        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
        )

        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
        )

        # =========================
        # GQA
        # =========================

        parser.add_argument(
            "--decoder-kv-attention-heads",
            type=int,
            default=4,
        )

        # =========================
        # RMSNORM
        # =========================

        parser.add_argument(
            "--decoder-use-rmsnorm",
            action="store_true",
        )

        # =========================
        # ROPE
        # =========================

        parser.add_argument(
            "--use-rope",
            action="store_true",
        )

        parser.add_argument(
            "--rope-theta",
            type=float,
            default=10000.0,
        )

        # =========================
        # ATTENTION BACKEND
        # =========================

        parser.add_argument(
            "--attn-implementation",
            type=str,
            default="fast_gqa",
        )

    @classmethod
    def build_model(cls, args, task):

        llama_lm_base(args)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args,
                "tokens_per_sample",
                DEFAULT_MAX_TARGET_POSITIONS,
            )

        args.rope_args = {
            "theta": args.rope_theta
        }

        embed_tokens = cls.build_embedding(
            args,
            task.source_dictionary,
            args.decoder_input_dim,
        )

        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )

        return cls(decoder)

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

def base_lm_architecture(args):

    # =========================================
    # CORE ARCH
    # =========================================

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

    # =========================================
    # DECODER
    # =========================================

    args.decoder_embed_dim = getattr(
        args,
        "decoder_embed_dim",
        1024,
    )

    args.decoder_output_dim = getattr(
        args,
        "decoder_output_dim",
        args.decoder_embed_dim,
    )

    args.decoder_input_dim = getattr(
        args,
        "decoder_input_dim",
        args.decoder_embed_dim,
    )

    args.decoder_ffn_embed_dim = getattr(
        args,
        "decoder_ffn_embed_dim",
        4096,
    )

    args.decoder_layers = getattr(
        args,
        "decoder_layers",
        24,
    )

    args.decoder_attention_heads = getattr(
        args,
        "decoder_attention_heads",
        16,
    )

    # =========================================
    # GQA
    # =========================================

    args.decoder_kv_attention_heads = getattr(
        args,
        "decoder_kv_attention_heads",
        4,
    )

    # =========================================
    # RMSNORM
    # =========================================

    args.decoder_use_rmsnorm = getattr(
        args,
        "decoder_use_rmsnorm",
        True,
    )

    # =========================================
    # ROPE
    # =========================================

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

    args.rope_args = {
        "theta": args.rope_theta
    }

    # =========================================
    # ATTENTION
    # =========================================

    args.attn_implementation = getattr(
        args,
        "attn_implementation",
        "fast_gqa",
    )

    # =========================================
    # REQUIRED TRANSFORMER DEFAULTS
    # =========================================

    args.decoder_normalize_before = getattr(
        args,
        "decoder_normalize_before",
        True,
    )

    args.no_decoder_final_norm = getattr(
        args,
        "no_decoder_final_norm",
        False,
    )

    args.no_token_positional_embeddings = getattr(
        args,
        "no_token_positional_embeddings",
        False,
    )

    args.decoder_learned_pos = getattr(
        args,
        "encoder_learned_pos",
        False,
    )

    args.share_decoder_input_output_embed = getattr(
        args,
        "share_decoder_input_output_embed",
        False,
    )

    args.layernorm_embedding = getattr(
        args,
        "layernorm_embedding",
        False,
    )

    args.no_scale_embedding = getattr(
        args,
        "no_scale_embedding",
        False,
    )

    # =========================================
    # LAYERDROP
    # =========================================

    args.decoder_layerdrop = getattr(
        args,
        "decoder_layerdrop",
        0.0,
    )

    args.decoder_layers_to_keep = getattr(
        args,
        "decoder_layers_to_keep",
        None,
    )

    # =========================================
    # QUANT NOISE
    # =========================================

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

    # =========================================
    # CHECKPOINTING
    # =========================================

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

    # =========================================
    # POSITIONS
    # =========================================

    args.max_target_positions = getattr(
        args,
        "max_target_positions",
        1024,
    )

    # =========================================
    # ADAPTIVE INPUT
    # =========================================

    args.adaptive_input = getattr(
        args,
        "adaptive_input",
        False,
    )

    args.character_embeddings = getattr(
        args,
        "character_embeddings",
        False,
    )
    
    args.adaptive_softmax_cutoff = getattr(
    args, "adaptive_softmax_cutoff", None)

    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0
    )
    
    args.adaptive_input = getattr(
        args, "adaptive_input", False
    )
    
    args.tie_adaptive_weights = getattr(
        args, "tie_adaptive_weights", False
    )
    
    args.tie_adaptive_proj = getattr(
        args, "tie_adaptive_proj", False
    )
@register_model_architecture(
    "llama_lm",
    "llama_lm_base",
)
def llama_lm_base(args):

    base_lm_architecture(args)