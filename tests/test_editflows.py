import torch

from editflows.editflows import (
    DataBatch,
    EditFlow,
    LinearKappaSchedule,
    LogRate,
    Operations,
    OptimalAligner,
    ShiftAligner,
    TokenWiseMixturePath,
    rm_blanks,
    rm_blanks_idx,
)


def test_seq_idx_is_sequence_index_of_each_token():
    seq_lens = (3, 1, 4)
    batch = DataBatch(seq_lens, torch.zeros(sum(seq_lens), dtype=torch.long))
    assert batch.seq_idx.tolist() == [0, 0, 0, 1, 2, 2, 2, 2]


def test_token_pos_good_case():
    seq_lens = (3, 1, 4)
    batch = DataBatch(seq_lens, torch.zeros(sum(seq_lens), dtype=torch.long))
    assert batch.token_pos.tolist() == [0, 1, 2, 0, 0, 1, 2, 3]


def test_token_pos_of_empty_batch_is_empty():
    assert DataBatch((), torch.empty(0, dtype=torch.long)).token_pos.tolist() == []


def test_splitting_batch_gives_subsequences():
    batch = DataBatch((2, 2, 4), torch.tensor([9, 10, 3, 4, 5, 6, 7, 8]))
    subsequences = batch.split_sequences()
    assert len(subsequences) == 3
    assert subsequences[0].tolist() == [9, 10]
    assert subsequences[1].tolist() == [3, 4]
    assert subsequences[2].tolist() == [5, 6, 7, 8]


def test_data_batch_from_equilength_batch():
    x = torch.randint(0, 100, (3, 5))
    batch = DataBatch.from_equilength_batch(x)
    assert batch.seq_lens == (5, 5, 5)
    assert batch.seq_idx.shape == (15,)
    assert batch.seq_idx.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert batch.x.shape == (15,)


def test_rm_blanks_removes_blanks():
    e = EditFlow.EPSILON_TOKEN
    z = [[8, 2, e, e, 3, 8, e], [4, 5, 6, e, e, e, e]]
    batch = rm_blanks(torch.asarray(z))
    assert batch.seq_idx.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert batch.x.tolist() == [8, 2, 3, 8, 4, 5, 6]


def test_rm_blanks_idx_points_to_previous_nonepsilon_token():
    e = EditFlow.EPSILON_TOKEN
    z = [[8, 2, e, e, 3, 8, e], [4, 5, 6, e, e, e, e]]
    idx = rm_blanks_idx(torch.asarray(z))
    assert idx.tolist() == [[0, 1, 1, 1, 2, 3, 3], [4, 5, 6, 6, 6, 6, 6]]


def test_rm_blanks_idx_marks_epsilon_tokens_without_predecessor():
    e = EditFlow.EPSILON_TOKEN
    z = [[e, 8, 2, 8, e], [e, e, 1, 2, 1]]
    idx = rm_blanks_idx(torch.asarray(z))
    assert idx.tolist() == [[-1, 0, 1, 2, 2], [-1, -1, 3, 4, 5]]


def test_shift_aligner_aligns_as_prescribed():
    aligner = ShiftAligner(k=2)
    x_0 = DataBatch.from_sequences([[1, 2], [4, 5, 6]])
    x_1 = DataBatch.from_sequences([[7], [8, 9, 10]])
    z_0, z_1 = aligner.align(x_0, x_1)
    e = EditFlow.EPSILON_TOKEN
    assert z_0.tolist() == [[1, 2, e, e, e], [4, 5, 6, e, e]]
    assert z_1.tolist() == [[e, e, 7, e, e], [e, e, 8, 9, 10]]


def test_shift_aligner_aligns_bos_tokens_left():
    aligner = ShiftAligner(k=2, bos_token=33)
    x_0 = DataBatch.from_sequences([[33, 2], [4, 5, 6]])
    x_1 = DataBatch.from_sequences([[33], [33, 9, 10]])
    z_0, z_1 = aligner.align(x_0, x_1)
    e = EditFlow.EPSILON_TOKEN
    assert z_0.tolist() == [[33, 2, e, e, e], [4, 5, 6, e, e]]
    assert z_1.tolist() == [[33, e, e, e, e], [33, e, e, 9, 10]]


def test_optimal_aligner():
    e = EditFlow.EPSILON_TOKEN

    def tok(s):
        if isinstance(s, str):
            return [e if c == " " else ord(c) for c in s]
        else:
            return [tok(sub) for sub in s]

    xs = [
        ("abc", ""),
        ("same", "same"),
        ("kitten", "smitten"),
        ("sitting", "kitten"),
        ("saturday", "sunday"),
    ]
    x_0 = DataBatch.from_sequences([tok(x[0]) for x in xs])
    x_1 = DataBatch.from_sequences([tok(x[1]) for x in xs])

    aligner = OptimalAligner()
    z_0, z_1 = aligner.align(x_0, x_1)
    # Stack aligned sequences so that x_0 and x_1 phrases are next to each other
    assert torch.stack((z_0, z_1), dim=1).tolist() == tok(
        [
            [
                "abc     ",
                "        ",
            ],
            [
                "same    ",
                "same    ",
            ],
            [
                " kitten ",
                "smitten ",
            ],
            [
                "sitting ",
                "kitten  ",
            ],
            [
                "saturday",
                "s  unday",
            ],
        ]
    )


def test_token_wise_mixture_path():
    kappa = LinearKappaSchedule()
    path = TokenWiseMixturePath(kappa)
    batch_size = 16
    gen = torch.Generator()
    t = torch.rand(batch_size)
    z_0, z_1 = torch.randint(0, 256, (2, batch_size, 64))
    zt, _ = path.sample(t=t, z_0=z_0, z_1=z_1, generator=gen)
    assert ((zt == z_0) | (zt == z_1)).all()


def test_rate_sample_ops_applies_no_ops_with_h_0():
    gen = torch.Generator().manual_seed(0)
    u_theta = LogRate(torch.log1p(torch.arange(3 * 7).reshape((3, -1))), torch.randn((2, 7, 5)))
    ops = u_theta.sample_euler_ops(h=0.0, generator=gen)
    assert (ops.op_masks == 0).all()
    assert ops.op_masks.shape == (3, 7)
    assert ops.op_tokens.shape == (2, 7)


def test_insertion_op_inserts_tokens():
    x_t = DataBatch.from_sequences([[1, 2], [3, 4], [5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[6, 7, 8, 9, 10], [0] * 5]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == (3, 2, 2)
    assert x_tp1.x.tolist() == [1, 2, 7, 3, 4, 5, 10]


def test_inserting_bos_is_noop():
    x_t = DataBatch.from_sequences([[1, 2], [3]])
    op = Operations(
        op_masks=torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        op_tokens=torch.tensor([[6, 99, 8], [0] * 3]),
    )
    x_tp1 = op.apply(x_t, bos_token=99)
    assert x_tp1.seq_lens == x_t.seq_lens
    assert x_tp1.x.tolist() == x_t.x.tolist()


def test_inserting_after_bos_works_as_normal():
    x_t = DataBatch.from_sequences([[1, 99], [3]])
    op = Operations(
        op_masks=torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.bool),
        op_tokens=torch.tensor([[6, 13, 8], [0] * 3]),
    )
    x_tp1 = op.apply(x_t, bos_token=99)
    assert x_tp1.seq_lens == (3, 1)
    assert x_tp1.x.tolist() == [1, 99, 13, 3]


def test_substitution_op_substitutes_tokens():
    x_t = DataBatch.from_sequences([[1, 2], [3, 4, 5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[0] * 5, [9, 8, 7, 6, 5]]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == x_t.seq_lens
    assert x_tp1.x.tolist() == [9, 2, 3, 6, 5]


def test_substituting_bos_is_noop():
    x_t = DataBatch.from_sequences([[1, 2], [99]])
    op = Operations(
        op_masks=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool),
        op_tokens=torch.tensor([[0] * 3, [0] * 3]),
    )
    x_tp1 = op.apply(x_t, bos_token=99)
    assert x_tp1.seq_lens == x_t.seq_lens
    assert x_tp1.x.tolist() == x_t.x.tolist()


def test_substituting_for_bos_is_noop():
    x_t = DataBatch.from_sequences([[1, 2], [3]])
    op = Operations(
        op_masks=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool),
        op_tokens=torch.tensor([[0] * 3, [0, 0, 99]]),
    )
    x_tp1 = op.apply(x_t, bos_token=99)
    assert x_tp1.seq_lens == x_t.seq_lens
    assert x_tp1.x.tolist() == x_t.x.tolist()


def test_deletion_op_deletes_tokens():
    x_t = DataBatch.from_sequences([[1, 2], [3, 4, 5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 1, 1]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[0] * 5, [9] * 5]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == (1, 1)
    assert x_tp1.x.tolist() == [2, 3]


def test_deleting_bos_is_noop():
    x_t = DataBatch.from_sequences([[1, 2], [99, 3]])
    op = Operations(
        op_masks=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.bool),
        op_tokens=torch.tensor([[0] * 4, [0] * 4]),
    )
    x_tp1 = op.apply(x_t, bos_token=99)
    assert x_tp1.seq_lens == x_t.seq_lens
    assert x_tp1.x.tolist() == x_t.x.tolist()


def test_sub_and_del_deletes_token():
    x_t = DataBatch.from_sequences([[1, 2], [3], [4, 5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[0] * 5, [9] * 5]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == (2, 1, 1)
    assert x_tp1.x.tolist() == [1, 2, 3, 5]


def test_del_and_ins_substitutes_token():
    x_t = DataBatch.from_sequences([[1, 2], [3], [4, 5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[9, 8, 7, 6, 5], [0] * 5]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == (2, 1, 2)
    assert x_tp1.x.tolist() == [1, 2, 7, 4, 5]


def test_deleting_all_tokens_of_a_sequence_does_not_delete_it():
    x_t = DataBatch.from_sequences([[1, 2], [3], [4, 5]])
    op = Operations(
        op_masks=torch.tensor(
            [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 1]], dtype=torch.bool
        ),
        op_tokens=torch.tensor([[9, 8, 7, 6, 5], [10, 11, 12, 13, 14]]),
    )
    x_tp1 = op.apply(x_t)
    assert x_tp1.seq_lens == (1, 1, 2)
    assert x_tp1.x.tolist() == [1, 3, 13, 5]
