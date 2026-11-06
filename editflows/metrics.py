r"""Deletion metrics: did the flow delete exactly the inserted tokens?

Two measurements, both bucketed by target length and by the original fraction $n_1 / n_0$ of the source:

* end-to-end: sample $\hat{x}_1$ from the source with provenance tracking (`DataBatch.origin`) and classify every
  source token as deleted / kept; against the inserted mask this gives the confusion matrix of the delete
  operation. Kept tokens whose letter changed count as substitution events, tokens without origin as insertion
  events.
* ranking: at $t = 0$ rank the source tokens by their deletion rate; precision@k with k the number of inserted
  tokens. Threshold-free, so it works although the rates near $t = 0$ are small by construction of the path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor, distributed

from editflows import DataBatch
from editflows.coupling import CoupledBatch

COUNTS = ("n_seq", "tp", "fp", "fn", "tn", "ins_events", "sub_events", "exact", "abs_len_err", "rank_hits", "rank_k")
RANK_COLUMNS = (COUNTS.index("rank_hits"), COUNTS.index("rank_k"))


class Buckets:
    """Sequence buckets: bins of the target length times bins of the original fraction n_1 / n_0."""

    def __init__(self, target_len_bins: list[int], fraction_bins: list[float]):
        assert len(target_len_bins) >= 2 and len(fraction_bins) >= 2
        self.target_edges = [float(e) for e in target_len_bins]
        self.fraction_edges = [float(e) for e in fraction_bins]

    @property
    def n_target(self) -> int:
        return len(self.target_edges) - 1

    @property
    def n_fraction(self) -> int:
        return len(self.fraction_edges) - 1

    def __len__(self) -> int:
        return self.n_target * self.n_fraction

    def index(self, target_len: Tensor, fraction: Tensor) -> Int[Tensor, "batch_size"]:
        """Bucket of each sequence; values outside the outer edges fall into the first / last bin."""
        t_edges = torch.tensor(self.target_edges[1:-1], device=target_len.device)
        f_edges = torch.tensor(self.fraction_edges[1:-1], device=fraction.device)
        t = torch.bucketize(target_len.double(), t_edges.double(), right=True)
        f = torch.bucketize(fraction.double(), f_edges.double(), right=True)
        return t * self.n_fraction + f

    def index_pair(self, x_0: DataBatch, x_1: DataBatch) -> Int[Tensor, "batch_size"]:
        """Bucket of each (source, target) pair; both sequences start with a BOS token."""
        target_len = x_1.seq_lens_tensor - 1
        source_len = (x_0.seq_lens_tensor - 1).clamp(min=1)
        return self.index(target_len, target_len / source_len)

    def labels(self) -> list[tuple[str, str]]:
        target = [f"{int(a)}-{int(b)}" for a, b in zip(self.target_edges[:-1], self.target_edges[1:])]
        fraction = [f"{a:g}-{b:g}" for a, b in zip(self.fraction_edges[:-1], self.fraction_edges[1:])]
        return [(t, f) for t in target for f in fraction]


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def rates(counts: Tensor) -> dict[str, float]:
    """Metrics from one row of counts (any bucket or the sum over buckets)."""
    c = {name: float(v) for name, v in zip(COUNTS, counts.tolist())}
    precision = _div(c["tp"], c["tp"] + c["fp"])
    recall = _div(c["tp"], c["tp"] + c["fn"])
    return {
        "del_precision": precision,
        "del_recall": recall,
        "del_f1": _div(2 * precision * recall, precision + recall),
        "del_chance": _div(c["tp"] + c["fn"], c["tp"] + c["fp"] + c["fn"] + c["tn"]),
        "rank_p_at_k": _div(c["rank_hits"], c["rank_k"]),
        "ins_per_seq": _div(c["ins_events"], c["n_seq"]),
        "sub_per_seq": _div(c["sub_events"], c["n_seq"]),
        "exact_match": _div(c["exact"], c["n_seq"]),
        "len_err": _div(c["abs_len_err"], c["n_seq"]),
    }


class DeletionCounts:
    """Integer counts per bucket, summed over sequences and reducible across processes."""

    def __init__(self, buckets: Buckets, device: torch.device | None = None):
        self.buckets = buckets
        self.counts = torch.zeros(len(buckets), len(COUNTS), dtype=torch.int64, device=device)

    def _add(self, name: str, bucket_ids: Int[Tensor, "m"], values: Tensor | None = None):
        if values is None:
            values = torch.ones_like(bucket_ids)
        self.counts[:, COUNTS.index(name)].index_add_(0, bucket_ids, values.to(torch.int64))

    def update_sampling(
        self,
        x_0: CoupledBatch,
        x_1: DataBatch,
        x_hat: DataBatch,
        bucket_ids: Int[Tensor, "batch_size"],
    ):
        """Count the outcome of sampling `x_hat` from `x_0` (with `origin`) against the inserted mask.

        The BOS token (position 0) is excluded from the token-level counts.
        """
        assert x_0.inserted is not None and x_hat.origin is not None

        token_bucket = bucket_ids[x_0.seq_idx]
        not_bos = x_0.token_pos > 0
        deleted = torch.ones(len(x_0.x), dtype=torch.bool, device=x_0.device)
        deleted[x_hat.origin[x_hat.origin >= 0]] = False
        inserted = x_0.inserted
        for name, mask in (
            ("tp", inserted & deleted),
            ("fp", ~inserted & deleted),
            ("fn", inserted & ~deleted),
            ("tn", ~inserted & ~deleted),
        ):
            self._add(name, token_bucket[mask & not_bos])

        hat_bucket = bucket_ids[x_hat.seq_idx]
        kept = x_hat.origin >= 0
        self._add("ins_events", hat_bucket[~kept])
        substituted = kept.clone()
        substituted[kept] = x_hat.x[kept] != x_0.x[x_hat.origin[kept]]
        self._add("sub_events", hat_bucket[substituted])

        self._add("n_seq", bucket_ids)
        self._add("abs_len_err", bucket_ids, (x_hat.seq_lens_tensor - x_1.seq_lens_tensor).abs())
        exact = [
            len(a) == len(b) and bool(torch.equal(a, b))
            for a, b in zip(x_hat.split_sequences(), x_1.split_sequences())
        ]
        self._add("exact", bucket_ids[torch.tensor(exact, device=bucket_ids.device)])

    def update_ranking(
        self, log_lambda_del: Float[Tensor, "n"], x_0: CoupledBatch, bucket_ids: Int[Tensor, "batch_size"]
    ):
        """Precision@k of the deletion rates: among the k tokens of a sequence with the highest rate, with k
        the number of inserted tokens, count how many are insertions."""
        assert x_0.inserted is not None
        scores = log_lambda_del.detach().float().clone()
        scores[x_0.token_pos == 0] = -torch.inf
        lens = list(x_0.seq_lens)
        for i, (s, ins) in enumerate(zip(torch.split(scores, lens), torch.split(x_0.inserted, lens))):
            k = int(ins.sum())
            if k == 0:
                continue
            hits = int(ins[torch.topk(s, k).indices].sum())
            self.counts[bucket_ids[i], RANK_COLUMNS[0]] += hits
            self.counts[bucket_ids[i], RANK_COLUMNS[1]] += k

    def all_reduce(self):
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(self.counts, op=distributed.ReduceOp.SUM)

    def summary(self) -> dict[str, float]:
        return rates(self.counts.sum(dim=0))

    def per_bucket(self) -> dict[tuple[str, str], dict[str, float]]:
        return {label: rates(row) for label, row in zip(self.buckets.labels(), self.counts)}

    def rows(self, *, stage: str, epoch: int, step: int) -> list[dict]:
        """One row per bucket plus the total, with raw counts and rates."""
        rows = []
        for (target, fraction), row in zip(self.buckets.labels() + [("all", "all")], list(self.counts) + [self.counts.sum(dim=0)]):
            rows.append({
                "stage": stage, "epoch": epoch, "step": step, "target_len": target, "fraction": fraction,
                **{name: int(v) for name, v in zip(COUNTS, row.tolist())},
                **rates(row),
            })
        return rows


def write_metrics(out_dir: Path, counts: DeletionCounts, *, stage: str, epoch: int, step: int):
    """Append this epoch's rows to `<out_dir>/deletion_metrics.csv` and write the raw counts as JSON.

    When a resumed run re-evaluates an epoch, the rows of that stage from this epoch on are replaced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(counts.rows(stage=stage, epoch=epoch, step=step))
    csv_path = out_dir / "deletion_metrics.csv"
    if csv_path.is_file():
        old = pd.read_csv(csv_path)
        old = old[~((old["stage"] == stage) & (old["epoch"] >= epoch))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(csv_path, index=False)

    payload = {
        "stage": stage, "epoch": epoch, "step": step, "count_names": list(COUNTS),
        "bucket_labels": counts.buckets.labels(), "counts": counts.counts.tolist(), "summary": counts.summary(),
    }
    (out_dir / f"{stage}_epoch_{epoch:04d}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
