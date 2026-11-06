import torch
from torch import nn

from editflows import CubicKappaSchedule, DataBatch, EditFlow, TokenWiseMixturePath
from editflows.coupling import ExtendedCoupling, KnownAligner, NullSource
from editflows.edit_head import EncoderWithEditHead
from editflows.encoder import LlamaEncoder

VOCAB = 21
BOS = 20


def make_model(seed=0):
    torch.manual_seed(seed)
    encoder = LlamaEncoder(
        vocab_size=VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, max_seq_len=64
    )
    return EncoderWithEditHead(encoder=encoder, vocab_size=VOCAB)


def batch(seed=0, n_seqs=3):
    g = torch.Generator().manual_seed(seed)
    seqs = [
        torch.cat((torch.tensor([BOS]), torch.randint(0, 20, (int(torch.randint(5, 15, (), generator=g)),), generator=g)))
        for _ in range(n_seqs)
    ]
    return DataBatch.from_sequences(seqs)


def test_forward_shapes_and_ranges():
    model = make_model()
    x_t = batch()
    t = torch.rand(x_t.batch_size)
    rate = model(x_t, t)

    n = len(x_t.x)
    assert rate.log_lambdas.shape == (3, n)
    assert rate.Q_logits.shape == (2, n, VOCAB)
    assert torch.isfinite(rate.log_lambdas).all() and torch.isfinite(rate.Q_logits).all()
    assert (rate.log_lambdas.abs() < 10).all()

    other = model(x_t, torch.zeros(x_t.batch_size))
    assert not torch.allclose(other.log_lambdas, rate.log_lambdas), "time enters the rates"


def test_sequences_are_encoded_independently():
    model = make_model().eval()
    x_t = batch()
    with torch.no_grad():
        hidden_batch = model.encoder(x_t)
        first = DataBatch.from_sequences([x_t.split_sequences()[0]])
        hidden_single = model.encoder(first)
    assert torch.allclose(hidden_batch[: x_t.seq_lens[0]], hidden_single, atol=1e-5)


def test_bregman_divergence_backward():
    model = make_model()
    x_1 = batch(seed=1)
    coupling = ExtendedCoupling(original_fraction=(0.5, 0.8), n_letters=20, bos_token=BOS)
    g = torch.Generator().manual_seed(0)
    x_0, _ = coupling.couple(NullSource().sample(x_1.batch_size, generator=g), x_1, generator=g)
    z_0, z_1 = KnownAligner().align(x_0, x_1)

    edit_flow = EditFlow(probability_path=TokenWiseMixturePath(CubicKappaSchedule()), bos_token=BOS)
    loss = edit_flow.bregman_divergence(model, z_0=z_0, z_1=z_1, generator=g)
    assert loss.shape == (x_1.batch_size,) and torch.isfinite(loss).all()
    loss.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    x_hat = edit_flow.sample_euler(model, x_0=x_0, n_steps=5, generator=g)
    assert x_hat.batch_size == x_0.batch_size
    for seq in x_hat.split_sequences():
        assert seq[0] == BOS


class ContextFreeEncoder(nn.Module):
    """Minimal encoder implementing the interface: a token embedding only."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x_t):
        return self.embedding(x_t.x)


def test_any_encoder_plugs_in():
    model = EncoderWithEditHead(encoder=ContextFreeEncoder(VOCAB, 16), vocab_size=VOCAB)
    x_t = batch()
    rate = model(x_t, torch.rand(x_t.batch_size))
    assert rate.log_lambdas.shape == (3, len(x_t.x)) and rate.Q_logits.shape == (2, len(x_t.x), VOCAB)
