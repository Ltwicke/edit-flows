r"""Error analysis of sampled deletions: verdicts, clusters of errors, what the grammar says about them, and an
Euler loop that records when every letter was deleted.

Verdict of a source letter (the notation of `evaluate.render`):

    .  original, kept            x  original, deleted  (wrong deletion)
    _  inserted, kept (missed)   +  inserted, deleted  (correct deletion)

Nothing here changes the edit flow: the sampler below is an analysis-side copy of `EditFlow.sample_euler` for the
delete operation (insert and substitute rates sit at the floor of the head and are left out) that keys its random
numbers by (step, source letter), so that two runs that differ in one intervention share everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from editflows.editflows import DataBatch, Operations

DOT, WRONG, MISSED, CORRECT = 0, 1, 2, 3
SYMBOLS = np.array(list(".x_+"))


# ---- sources --------------------------------------------------------------------------------------------------


@dataclass
class Sources:
    """Sources (targets with inserted letters) as flat arrays: letters without BOS, the inserted mask, offsets."""

    letters: np.ndarray  # int8, all sources concatenated
    inserted: np.ndarray  # bool
    offsets: np.ndarray  # int64, len = number of sources + 1
    split: np.ndarray  # 0 = validation, 1 = test target
    target: np.ndarray  # index of the target in its split

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def span(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def seq(self, i: int) -> np.ndarray:
        return self.letters[self.span(i)].astype(np.int64)

    def mask(self, i: int) -> np.ndarray:
        return self.inserted[self.span(i)]

    @property
    def seq_of(self) -> np.ndarray:
        return np.repeat(np.arange(len(self)), np.diff(self.offsets))

    @property
    def rho(self) -> np.ndarray:
        """Share of inserted letters per source."""
        return np.add.reduceat(self.inserted.astype(np.int64), self.offsets[:-1]) / np.diff(self.offsets)

    def save(self, path):
        np.savez_compressed(path, letters=self.letters, inserted=self.inserted, offsets=self.offsets, split=self.split,
                            target=self.target)

    @classmethod
    def load(cls, path) -> "Sources":
        z = np.load(path)
        return cls(z["letters"], z["inserted"], z["offsets"], z["split"], z["target"])


def build_sources(datamodule, coupling, n_val: int, n_test: int, seed: int) -> Sources:
    """Couple the first `n_val` validation and `n_test` test targets with fresh insertions."""
    generator = torch.Generator().manual_seed(seed)
    letters, inserted, lengths, split, target = [], [], [], [], []
    for s, (data, n) in enumerate(((datamodule.val_data, n_val), (datamodule.test_data, n_test))):
        for start in range(0, min(n, len(data)), 256):
            ids = range(start, min(start + 256, n, len(data)))
            x_0, _ = coupling.couple(None, DataBatch.from_sequences([data[i] for i in ids]), generator=generator)
            for seq, ins in zip(torch.split(x_0.x, list(x_0.seq_lens)), torch.split(x_0.inserted, list(x_0.seq_lens))):
                letters.append(seq[1:].numpy().astype(np.int8))
                inserted.append(ins[1:].numpy())
                lengths.append(len(seq) - 1)
            split += [s] * len(ids)
            target += list(ids)
    return Sources(np.concatenate(letters), np.concatenate(inserted), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                   np.array(split, dtype=np.int8), np.array(target, dtype=np.int32))


# ---- verdicts and clusters ------------------------------------------------------------------------------------


def verdict_codes(inserted: np.ndarray, deleted: np.ndarray) -> np.ndarray:
    """0 `.`, 1 `x`, 2 `_`, 3 `+` per source letter."""
    return 2 * inserted.astype(np.uint8) + deleted.astype(np.uint8)


def verdict_string(codes: np.ndarray) -> str:
    return "".join(SYMBOLS[codes])


def cluster_ids(codes: np.ndarray, gap: int = 2) -> np.ndarray:
    """Cluster number of every error letter (`x` or `_`) of one source, -1 elsewhere.

    Two errors belong to one cluster if at most `gap` kept original letters (`.`) lie between them: distances are
    counted in the output, so correct deletions do not separate errors.
    """
    out = np.full(len(codes), -1, dtype=np.int32)
    errors = np.flatnonzero((codes == WRONG) | (codes == MISSED))
    if len(errors):
        kept_originals = np.cumsum(codes == DOT)
        out[errors] = np.concatenate([[0], np.cumsum(np.diff(kept_originals[errors]) > gap)])
    return out


def violated_windows(seq: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """For every letter from the third on: is the window ending in it forbidden by the table? (False for the first two)"""
    seq = np.asarray(seq, dtype=np.int64)
    out = np.zeros(len(seq), dtype=bool)
    if len(seq) > 2:
        out[2:] = ~eligible[seq[:-2], seq[1:-1], seq[2:]]
    return out


def clusters(letters: np.ndarray, inserted: np.ndarray, deleted: np.ndarray, eligible: np.ndarray, gap: int = 2) -> pd.DataFrame:
    """One row per error cluster of one source, typed from ground truth alone.

    Around a cluster, O is the output (kept letters) and T the target (original letters):

    * `Z` label-only: O and T are the same string;
    * `U` unresolved: a forbidden window of the output touches the cluster;
    * `R` repaired hole: grammatical, at least two wrong deletions and more wrong deletions than missed insertions;
    * `V` valid alternative: grammatical but different from the target, anything else.
    """
    codes = verdict_codes(inserted, deleted)
    ids = cluster_ids(codes, gap)
    kept = np.flatnonzero(~deleted)
    forbidden = violated_windows(letters[kept], eligible)
    bad = kept[forbidden]  # source position of the last letter of a forbidden window of the output
    first = kept[np.maximum(np.arange(len(kept)) - 2, 0)][forbidden]  # ... and of its first letter
    rows = []
    for c in range(ids.max() + 1 if len(ids) else 0):
        members = np.flatnonzero(ids == c)
        start, end = int(members[0]), int(members[-1])
        inside = slice(start, end + 1)
        output, target = letters[inside][~deleted[inside]], letters[inside][~inserted[inside]]
        n_x, n_miss = int((codes[inside] == WRONG).sum()), int((codes[inside] == MISSED).sum())
        violated = int(((first <= end) & (bad >= start)).sum())
        if len(output) == len(target) and bool((output == target).all()):
            kind = "Z"
        elif violated:
            kind = "U"
        elif n_x >= 2 and n_x > n_miss:
            kind = "R"
        else:
            kind = "V"
        rows.append({"cluster": c, "start": start, "end": end, "n_x": n_x, "n_miss": n_miss,
                     "n_plus": int((codes[inside] == CORRECT).sum()), "n_dot": int((codes[inside] == DOT).sum()),
                     "violated": violated, "kind": kind, "pattern": verdict_string(codes[inside])})
    return pd.DataFrame(rows)


# ---- what the grammar says about a removal ---------------------------------------------------------------------


def break_size(seq: np.ndarray, sites: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """How many of the two windows that straddle the gap are forbidden after removing `seq[site]` (0, 1 or 2).

    Sites need two letters on each side.
    """
    seq = np.asarray(seq, dtype=np.int64)
    l2, l1, r1, r2 = seq[sites - 2], seq[sites - 1], seq[sites + 1], seq[sites + 2]
    return (~eligible[l2, l1, r1]).astype(np.int8) + (~eligible[l1, r1, r2]).astype(np.int8)


def minimal_repair(seq: np.ndarray, lo: int, hi: int, eligible: np.ndarray) -> int:
    """Fewest deletions among `seq[lo:hi]` that leave no forbidden window between `seq[lo - 2]` and `seq[hi + 1]`
    (the two letters before and after the stretch stay). -1 if no choice of deletions works."""
    seq = np.asarray(seq, dtype=np.int64)
    best = {(int(seq[lo - 2]), int(seq[lo - 1])): 0}  # last two kept letters -> fewest deletions so far
    for c in map(int, seq[lo:hi]):
        step: dict = {}
        for (a, b), cost in best.items():
            if cost + 1 < step.get((a, b), 1 << 30):
                step[(a, b)] = cost + 1
            if eligible[a, b, c] and cost < step.get((b, c), 1 << 30):
                step[(b, c)] = cost
        best = step
    r1, r2 = int(seq[hi]), int(seq[hi + 1])
    costs = [cost for (a, b), cost in best.items() if eligible[a, b, r1] and eligible[b, r1, r2]]
    return min(costs) if costs else -1


# ---- comparisons that need no grammar --------------------------------------------------------------------------


def lcs_length(a, b) -> int:
    """Length of the longest common subsequence (bit-parallel, Allison & Dix)."""
    masks: dict = {}
    for i, ch in enumerate(a):
        masks[int(ch)] = masks.get(int(ch), 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for ch in b:
        u = v & masks.get(int(ch), 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def canonical_counts(letters: np.ndarray, inserted: np.ndarray, deleted: np.ndarray) -> dict[str, int]:
    """Errors counted on the strings instead of the labels: any set of deletions that produces the same output is
    the same answer. `wrong` = target letters missing from the output, `missed` = letters the output has too many."""
    output, target = letters[~deleted], letters[~inserted]
    common = lcs_length(output, target)
    return {"wrong": len(target) - common, "missed": len(output) - common, "deleted": int(deleted.sum()),
            "inserted": int(inserted.sum()), "equal": int(len(output) == len(target) == common)}


# ---- an Euler loop that records events -------------------------------------------------------------------------


def _batch(tokens: torch.Tensor, seq_of: torch.Tensor, alive: torch.Tensor, n_seqs: int, bos_token: int) -> DataBatch:
    """The batch `BOS + letters still present` of every sequence; `origin` is the index of the letter in `tokens`,
    BOS tokens get the ids `len(tokens) + sequence`."""
    present = torch.argwhere(alive)[:, 0]
    seq = seq_of[present]
    counts = torch.bincount(seq, minlength=n_seqs)
    starts = torch.cumsum(counts, 0) - counts + torch.arange(n_seqs, device=tokens.device)  # index of each BOS
    x = tokens.new_full((len(present) + n_seqs,), bos_token)
    origin = torch.empty(len(x), dtype=torch.int64, device=tokens.device)
    origin[starts] = len(tokens) + torch.arange(n_seqs, device=tokens.device)
    where = torch.arange(len(present), device=tokens.device) + seq + 1
    x[where], origin[where] = tokens[present], present
    return DataBatch(seq_lens=tuple((counts + 1).tolist()), x=x, origin=origin)


@torch.inference_mode()
def model_euler(
    model,
    seqs: list[np.ndarray],
    kappa,
    bos_token: int,
    *,
    device,
    n_steps: int = 100,
    seed: int = 0,
    start_step: int = 0,
    end_step: int | None = None,
    alive: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    exclusion: int = 0,
    sharpen: float = 1.0,
    floor: float = 0.0,
    log_q: bool = False,
) -> dict[str, np.ndarray]:
    """Sample deletions for the sources `seqs` (letters without BOS) and record them per source letter.

    The loop is `EditFlow.sample_euler` for the delete operation: at step k every letter still present is deleted
    with probability `h * lambda_del`. Returned arrays are flat over all letters of `seqs`:

    * `del_step`: the step of the deletion, -1 for letters that survived;
    * `q0`: the model's own probability that the letter is an insertion, `lambda_del / hazard`, at `start_step`;
    * `exposure`: the summed deletion probability the letter was exposed to while present;
    * `qlog` (if `log_q`): that probability at every step, NaN once the letter is gone.

    Interventions: `alive` (letters present at `start_step`; removing an original letter there is a forced wrong
    deletion), `keep` (letters that may never be deleted). `end_step = start_step + 1` evaluates the model once,
    which is the cheap way to read `q0` for a given state. Sampler variants: `exclusion = w` allows one deletion per
    window of +-w letters per step, `sharpen = beta` replaces q by q^beta / (q^beta + (1 - q)^beta), `floor` sets
    q below it to zero.
    """
    lengths = np.array([len(s) for s in seqs])
    tokens = torch.as_tensor(np.concatenate(seqs), dtype=torch.int64, device=device)
    seq_of = torch.as_tensor(np.repeat(np.arange(len(seqs)), lengths), device=device)
    n = len(tokens)
    present = torch.ones(n, dtype=torch.bool, device=device) if alive is None else torch.as_tensor(alive, device=device)
    locked = torch.zeros(n + len(seqs), dtype=torch.bool, device=device)
    locked[n:] = True  # BOS
    if keep is not None:
        locked[:n] = torch.as_tensor(keep, device=device)
    x_t = _batch(tokens, seq_of, present, len(seqs), bos_token)
    del_step = torch.full((n,), -1, dtype=torch.int16, device=device)
    q0 = torch.full((n,), float("nan"), device=device)
    exposure = torch.zeros(n, device=device)
    qlog = torch.full((n, n_steps), float("nan"), dtype=torch.float16, device=device) if log_q else None
    h = 1.0 / n_steps
    for k in range(start_step, n_steps if end_step is None else end_step):
        t = torch.tensor(k * h, dtype=torch.float64)
        hazard = float(kappa.loss_coefficient(t))
        rate = model(x_t, torch.full((x_t.batch_size,), k * h, device=device))
        p = (h * rate.log_lambda_del.float().exp()).clamp(max=1.0)  # the deletion probability of sample_euler_ops
        q = (p / (h * hazard)).clamp(max=1.0)
        letter = x_t.origin < n
        ids = x_t.origin[letter]
        if k == start_step:
            q0[ids] = q[letter]
        if qlog is not None:
            qlog[ids, k] = q[letter].to(torch.float16)
        exposure[ids] += p[letter]
        if sharpen != 1.0 or floor > 0:
            q = q.clamp(1e-12, 1 - 1e-6)
            q = q**sharpen / (q**sharpen + (1 - q) ** sharpen)
            p = h * hazard * torch.where(q < floor, torch.zeros_like(q), q)
        u = torch.rand(n + len(seqs), generator=torch.Generator(device).manual_seed(seed * 1_000_003 + k), device=device)
        urgency = torch.where(locked[x_t.origin], torch.full_like(p, -1.0), p - u[x_t.origin])
        delete = urgency > 0
        if exclusion > 0:
            grid = torch.full((x_t.batch_size, max(x_t.seq_lens)), -1.0, device=device)
            grid[x_t.seq_idx, x_t.token_pos] = urgency
            best = torch.nn.functional.max_pool1d(grid[:, None], 2 * exclusion + 1, stride=1, padding=exclusion)[:, 0]
            delete &= urgency >= best[x_t.seq_idx, x_t.token_pos]
        del_step[x_t.origin[delete]] = k
        ops = Operations(op_masks=torch.stack([torch.zeros_like(delete), torch.zeros_like(delete), delete]),
                         op_tokens=torch.zeros((2, len(delete)), dtype=x_t.x.dtype, device=device))
        x_t = ops.apply(x_t, bos_token=bos_token)
    out = {"del_step": del_step.cpu().numpy(), "q0": q0.cpu().numpy(), "exposure": exposure.cpu().numpy()}
    if qlog is not None:
        out["qlog"] = qlog.cpu().numpy()
    return out
