"""The edit-flow head on top of a sequence encoder."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from editflows import DataBatch, LogRate
from editflows.mlp import MLP
from editflows.pos_emb import NyquistPositionalEmbedding


class EditFlowHead(nn.Module):
    r"""Turns per-token hidden states and the time $t$ into the rate function $u_t^\theta(\cdot \mid x_t)$.

    The time embedding (as in `editflows.llama.LlamaWithHead`) is added to every hidden state; the log-rates are
    squashed to $(-\lambda_\max, \lambda_\max)$ and the token logits come from a plain linear layer, so the head
    does not depend on the encoder's token embedding.
    """

    def __init__(self, *, hidden_size: int, vocab_size: int, max_log_lambda: float = 10.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_log_lambda = max_log_lambda

        self.t_emb = nn.Sequential(
            NyquistPositionalEmbedding(hidden_size, 100),
            MLP(hidden_size, hidden_size, hidden_features=[4 * hidden_size], actfn=nn.SiLU),
        )
        self.lambda_head = MLP(hidden_size, 3, hidden_features=[4 * hidden_size], actfn=nn.SiLU)
        self.Q_head = MLP(hidden_size, 2 * vocab_size, hidden_features=[4 * hidden_size], actfn=nn.SiLU)

    def forward(
        self, hidden: Float[Tensor, "n hidden"], x_t: DataBatch, t: Float[Tensor, "batch_size"]
    ) -> LogRate:
        h = hidden + self.t_emb(t.to(dtype=hidden.dtype))[x_t.seq_idx]
        log_lambdas = torch.tanh(self.lambda_head(h)).transpose(0, 1) * self.max_log_lambda
        Q_logits = self.Q_head(h).unflatten(-1, (2, self.vocab_size)).transpose(0, 1)
        return LogRate(log_lambdas, Q_logits)


class EncoderWithEditHead(nn.Module):
    """An `EditFlowModel`: sequence encoder followed by the edit-flow head."""

    def __init__(self, *, encoder: nn.Module, vocab_size: int, max_log_lambda: float = 10.0):
        super().__init__()

        self.encoder = encoder
        self.head = EditFlowHead(
            hidden_size=encoder.hidden_size, vocab_size=vocab_size, max_log_lambda=max_log_lambda
        )

    def forward(self, x_t: DataBatch, t: Float[Tensor, "batch_size"]) -> LogRate:
        return self.head(self.encoder(x_t), x_t, t)
