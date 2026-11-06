import pandas as pd
import torch

from editflows import DataBatch
from editflows.coupling import CoupledBatch
from editflows.metrics import COUNTS, Buckets, DeletionCounts, write_metrics

BOS = 20


def test_buckets():
    b = Buckets([100, 200, 300, 400, 501], [0.4, 0.55, 0.7, 0.9])
    assert len(b) == 12 and b.n_target == 4 and b.n_fraction == 3
    target_len = torch.tensor([50, 100, 199, 200, 450, 500, 9999])
    fraction = torch.tensor([0.1, 0.4, 0.54, 0.55, 0.7, 0.89, 1.0])
    assert b.index(target_len, fraction).tolist() == [0, 0, 0, 4, 11, 11, 11]
    assert b.labels()[0] == ("100-200", "0.4-0.55") and b.labels()[-1] == ("400-501", "0.7-0.9")


def source(tokens, inserted):
    x_0 = CoupledBatch(seq_lens=(len(tokens),), x=torch.tensor(tokens), inserted=torch.tensor(inserted))
    x_0.origin = torch.arange(len(tokens))
    return x_0


def test_confusion_counts_from_provenance():
    buckets = Buckets([0, 1000], [0.0, 1.0])
    counts = DeletionCounts(buckets)
    # source: BOS a b I c I I d   (I = inserted)
    x_0 = source([BOS, 1, 2, 7, 3, 8, 9, 4], [False, False, False, True, False, True, True, False])
    x_1 = DataBatch.from_sequences([[BOS, 1, 2, 3, 4]])
    # model output: kept BOS, a, deleted b (FP), deleted first I (TP), kept c, kept second I (FN),
    # deleted third I (TP), substituted d (kept, sub event), inserted one new token
    x_hat = DataBatch(seq_lens=(6,), x=torch.tensor([BOS, 1, 3, 8, 5, 6]), origin=torch.tensor([0, 1, 4, 5, 7, -1]))
    counts.update_sampling(x_0, x_1, x_hat, torch.tensor([0]))

    row = {name: int(v) for name, v in zip(COUNTS, counts.counts[0].tolist())}
    assert row["tp"] == 2 and row["fp"] == 1 and row["fn"] == 1 and row["tn"] == 3
    assert row["ins_events"] == 1 and row["sub_events"] == 1
    assert row["n_seq"] == 1 and row["exact"] == 0 and row["abs_len_err"] == 1

    r = counts.summary()
    assert abs(r["del_precision"] - 2 / 3) < 1e-9 and abs(r["del_recall"] - 2 / 3) < 1e-9
    assert abs(r["del_chance"] - 3 / 7) < 1e-9 and r["ins_per_seq"] == 1 and r["sub_per_seq"] == 1

    # A perfect sample
    perfect = DeletionCounts(buckets)
    x_hat = DataBatch(seq_lens=(5,), x=torch.tensor([BOS, 1, 2, 3, 4]), origin=torch.tensor([0, 1, 2, 4, 7]))
    perfect.update_sampling(x_0, x_1, x_hat, torch.tensor([0]))
    r = perfect.summary()
    assert r["del_precision"] == 1 and r["del_recall"] == 1 and r["del_f1"] == 1 and r["exact_match"] == 1


def test_ranking_precision_at_k():
    buckets = Buckets([0, 1000], [0.0, 1.0])
    counts = DeletionCounts(buckets)
    x_0 = source([BOS, 1, 2, 7, 3, 8, 9, 4], [False, False, False, True, False, True, True, False])
    # BOS gets the highest rate but is excluded; of the 3 top non-BOS rates 2 are insertions
    log_lambda_del = torch.tensor([9.0, 0.0, 5.0, 4.0, 0.0, 3.0, 0.0, 0.0])
    counts.update_ranking(log_lambda_del, x_0, torch.tensor([0]))
    assert counts.summary()["rank_p_at_k"] == 2 / 3


def test_write_metrics_replaces_resumed_epochs(tmp_path):
    buckets = Buckets([0, 1000], [0.0, 1.0])
    counts = DeletionCounts(buckets)
    counts.counts[0, COUNTS.index("n_seq")] = 4
    for epoch in (0, 1, 2):
        write_metrics(tmp_path, counts, stage="val", epoch=epoch, step=10 * epoch)
    write_metrics(tmp_path, counts, stage="val", epoch=1, step=99)

    df = pd.read_csv(tmp_path / "deletion_metrics.csv")
    assert sorted(df["epoch"].unique().tolist()) == [0, 1]
    assert df[df["epoch"] == 1]["step"].unique().tolist() == [99]
    assert (tmp_path / "val_epoch_0002.json").is_file()
    assert set(df.columns) >= {"stage", "epoch", "step", "target_len", "fraction", "tp", "del_f1"}
