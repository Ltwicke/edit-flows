import torch

from editflows import DataBatch, EditFlow, LinearKappaSchedule, TokenWiseMixturePath
from editflows.editflows import Operations

BOS = 20


def with_origin(seqs):
    x = DataBatch.from_sequences(seqs)
    x.origin = torch.arange(len(x.x))
    return x


def ops(n, ins=(), sub=(), delete=(), ins_tok=1, sub_tok=2):
    masks = torch.zeros(3, n, dtype=torch.bool)
    masks[0, list(ins)] = True
    masks[1, list(sub)] = True
    masks[2, list(delete)] = True
    tokens = torch.stack((torch.full((n,), ins_tok), torch.full((n,), sub_tok)))
    return Operations(op_masks=masks, op_tokens=tokens)


def test_origin_follows_tokens_through_edits():
    x_t = with_origin([[BOS, 5, 6, 7], [BOS, 8, 9]])
    # sequence 0: insert after token 1 (BOS), substitute token 2, delete token 3; sequence 1: delete token 5
    out = ops(7, ins=[0], sub=[2], delete=[3, 5]).apply(x_t, bos_token=BOS)

    assert out.seq_lens == (4, 2)
    assert out.x.tolist() == [BOS, 1, 5, 2, BOS, 9]
    assert out.origin.tolist() == [0, -1, 1, 2, 4, 6]


def test_no_origin_means_no_origin():
    x_t = DataBatch.from_sequences([[BOS, 5, 6]])
    out = ops(3, delete=[2]).apply(x_t, bos_token=BOS)
    assert out.origin is None and out.x.tolist() == [BOS, 5]


def test_whole_sequence_deletion_is_prevented_with_origin():
    x_t = with_origin([[3, 4]])
    out = ops(2, delete=[0, 1]).apply(x_t)
    assert out.x.tolist() == [3, 4] and out.origin.tolist() == [0, 1]


class ConstantRates:
    """A rate model that always proposes to delete every non-BOS token and never inserts/substitutes."""

    def __call__(self, x_t, t):
        from editflows import LogRate

        n = len(x_t.x)
        log_lambdas = torch.full((3, n), -10.0)
        log_lambdas[2] = 10.0
        return LogRate(log_lambdas, torch.zeros(2, n, BOS + 1))


def test_sampler_returns_provenance():
    x_0 = with_origin([[BOS, 1, 2, 3], [BOS, 4, 5]])
    edit_flow = EditFlow(probability_path=TokenWiseMixturePath(LinearKappaSchedule()), bos_token=BOS)
    x_hat = edit_flow.sample_euler(ConstantRates(), x_0=x_0, n_steps=3, generator=torch.Generator().manual_seed(0))

    assert x_hat.origin is not None
    assert x_hat.x.tolist() == [BOS, BOS], "everything but BOS is deleted"
    assert x_hat.origin.tolist() == [0, 4], "BOS tokens keep the index of their source position"
