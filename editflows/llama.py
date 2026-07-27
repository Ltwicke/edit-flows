import math

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend
from transformers import AttentionInterface, AttentionMaskInterface, LlamaConfig, LlamaModel

from editflows import DataBatch, LogRate
from editflows.mlp import MLP
from editflows.pos_emb import NyquistPositionalEmbedding


def nested_tensor_sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert query.shape[0] == 1, "Only works with batch size 1"
    assert key.shape[0] == 1, "Only works with batch size 1"
    assert value.shape[0] == 1, "Only works with batch size 1"

    seq_lens = attention_mask
    q = torch.nested.nested_tensor_from_jagged(
        query[0].transpose(0, 1).contiguous(), lengths=seq_lens
    ).transpose(1, 2)
    k = torch.nested.nested_tensor_from_jagged(
        key[0].transpose(0, 1).contiguous(), lengths=seq_lens
    ).transpose(1, 2)
    v = torch.nested.nested_tensor_from_jagged(
        value[0].transpose(0, 1).contiguous(), lengths=seq_lens
    ).transpose(1, 2)

    if query.device.type == "cpu":
        backends = [SDPBackend.MATH]
    else:
        backends = [
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
    with torch.nn.attention.sdpa_kernel(backends):
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=scaling, dropout_p=dropout)

    # I have no clue why we do not need to transpose(1, 2) again here, but if we do, it doesn't work
    # anymore (model does not learn anything).
    return attn_out.values().transpose(0, 1)[None], None


def nested_tensor_sdpa_mask(attention_mask, **kwargs):
    return attention_mask


AttentionInterface.register("nested_tensor_sdpa", nested_tensor_sdpa)
AttentionMaskInterface.register("nested_tensor_sdpa", nested_tensor_sdpa_mask)


class LlamaWithHead(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_seq_len: int,
        max_log_lambda: float = 10.0,
    ):
        super().__init__()

        self.max_log_lambda = max_log_lambda

        # We add time tokens in position 0
        max_position_embeddings = 1 + max_seq_len
        # Size the SwiGLU MLP per the Llama convention (8/3 * hidden size
        # rounded up to a multiple of 256) to make the parameter counts equal to
        # the MLPs with 4x width below.
        intermediate_size = 256 * math.ceil(8 * hidden_size / 3 / 256)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attn_implementation="nested_tensor_sdpa",
        )
        self.model = LlamaModel(config)

        self.lambda_head = MLP(
            config.hidden_size,
            3,
            hidden_features=[4 * config.hidden_size],
            actfn=nn.SiLU,
        )
        self.lambda_rearrange = Rearrange("... inssubdel -> inssubdel ...")
        self.Q_head = MLP(
            config.hidden_size,
            2 * config.hidden_size,
            hidden_features=[4 * config.hidden_size],
            actfn=nn.SiLU,
        )
        self.Q_rearrange = Rearrange(
            "... (inssub hidden) -> ... inssub hidden",
            hidden=config.hidden_size,
            inssub=2,
        )

        self.t_emb = nn.Sequential(
            NyquistPositionalEmbedding(config.hidden_size, 100),
            MLP(
                config.hidden_size,
                config.hidden_size,
                hidden_features=[4 * config.hidden_size],
                actfn=nn.SiLU,
            ),
        )

    def forward(self, x_t: DataBatch, t: Float[Tensor, "batch_size"]):
        # Compute language token and time token embeddings
        token_embeds = self.model.embed_tokens(x_t.x)
        t_embeds = self.t_emb(t.to(dtype=token_embeds.dtype))

        # Prepend time token embeddings to each sequence
        inputs = torch.cat(
            [
                val
                for pair in zip(
                    torch.split(t_embeds, [1] * x_t.batch_size),
                    torch.split(token_embeds, list(x_t.seq_lens)),
                )
                for val in pair
            ]
        )

        # Construct position IDs for the extended sequences (with time tokens at position 0)
        seq_lens = x_t.seq_lens_tensor + 1
        seq_idx = torch.repeat_interleave(
            torch.arange(x_t.batch_size, device=x_t.device, dtype=seq_lens.dtype),
            seq_lens,
            output_size=sum(x_t.seq_lens) + x_t.batch_size,
        )
        shifts = F.pad((seq_lens)[:-1].cumsum(dim=0), (1, 0))
        position_ids = torch.arange(len(x_t.x) + len(t), device=x_t.x.device) - shifts[seq_idx]

        outputs = self.model(
            inputs_embeds=inputs[None],
            attention_mask=seq_lens,
            position_ids=position_ids[None],
        )

        # Drop outputs from time tokens
        hidden_states = torch.cat(
            [
                part[1:]
                for part in torch.split(outputs.last_hidden_state[0], [l + 1 for l in x_t.seq_lens])
            ]
        )

        log_lambdas = self.lambda_rearrange(self.lambda_head(hidden_states))
        Qs = torch.einsum(
            "tn, ...hn -> h...t",
            self.model.embed_tokens.weight,
            self.Q_rearrange(self.Q_head(hidden_states)),
        )

        log_lambdas = torch.tanh(log_lambdas) * self.max_log_lambda

        return LogRate(log_lambdas, Qs)
