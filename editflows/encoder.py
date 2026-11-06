"""Sequence encoders: the interface between a (possibly pretrained) sequence encoder and the edit-flow head.

An encoder maps a packed batch of token sequences to one hidden vector per token. It does not see the flow time;
time enters in `editflows.edit_head.EditFlowHead`, so a fixed, pretrained sequence encoder can take the place of
the placeholder below without changes to the head or the training code.
"""

from __future__ import annotations

import math
from typing import Protocol

from jaxtyping import Float
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaModel

import editflows.llama  # noqa: F401  (registers the "nested_tensor_sdpa" attention implementation)
from editflows import DataBatch


class SequenceEncoder(Protocol):
    """Maps a packed batch of token sequences to one hidden vector per token."""

    hidden_size: int

    def __call__(self, x_t: DataBatch) -> Float[Tensor, "n hidden"]: ...


class LlamaEncoder(nn.Module):
    """Placeholder encoder: a small Llama trunk with bidirectional attention within each sequence.

    This is the trunk of `editflows.llama.LlamaWithHead` without the time token. Sized small enough for CPU use
    by default; the real sequence encoder replaces it through the `encoder` node of the model config.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Size the SwiGLU MLP per the Llama convention (8/3 * hidden size rounded up to a multiple of 256)
        intermediate_size = 256 * math.ceil(8 * hidden_size / 3 / 256)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_seq_len,
            attn_implementation="nested_tensor_sdpa",
        )
        self.model = LlamaModel(config)

    def forward(self, x_t: DataBatch) -> Float[Tensor, "n hidden"]:
        # The whole batch is one packed sequence; the attention implementation splits it by `seq_lens` and
        # positions restart at 0 in every sequence.
        outputs = self.model(
            inputs_embeds=self.model.embed_tokens(x_t.x)[None],
            attention_mask=x_t.seq_lens_tensor,
            position_ids=x_t.token_pos[None],
        )
        return outputs.last_hidden_state[0]
