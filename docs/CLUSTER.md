# Training on a SLURM cluster

Everything here is site-neutral: replace the `<...>` placeholders with your project account, partition,
module and path names.

## 1. One-time setup on a login node

```bash
git clone https://github.com/Ltwicke/edit-flows.git && cd edit-flows
module purge
module load <Python 3.13 module>                     # or any python3.13 on PATH
python -m venv <project-area>/envs/EditFlows_env     # keep environments and checkpoints in the project area,
source <project-area>/envs/EditFlows_env/bin/activate#  not in scratch areas that are cleaned automatically
pip install -U pip
pip install -r requirements-cuda.txt                 # torch 2.13.0+cu126; use the cu128 index for GPUs that need CUDA >= 12.8
python -m pytest -q tests -m "not slow"              # CPU, a few seconds
```

Put the sequence file where the jobs can read it and point `data.path` at it (or edit
`config/data/sequences.yaml`). The first run parses the file once and caches it as `.npz` next to it.

Quick check with the placeholder data, on a login node or a short test queue:

```bash
python scripts/make_synthetic_targets.py --n 20000 --out data/synthetic/targets.xlsx
python train.py experiment=deletion-smoke trainer.accelerator=cpu     # ~1 minute on CPU
```

## 2. Submitting a run

```bash
sbatch scripts/train.sbatch <run name>                                  # 1 GPU, experiment=deletion-base
sbatch --gpus=4 --ntasks-per-node=4 scripts/train.sbatch <run name>     # 4 GPUs on one node (DDP)
sbatch scripts/train.sbatch <run name> data.path=<file> trainer.max_epochs=50   # any hydra override
```

Fill in `--account`, `--partition`, the module line and the environment path in `scripts/train.sbatch` first.
One process per GPU (`--ntasks-per-node` = `--gpus`); the script passes `gpus=N`, which sets
`trainer.devices`, and `data.num_workers = cpus-per-task - 1`. Lightning uses DDP automatically.

Per node, the memory need is dominated by the data workers and the checkpoint writes; `--mem-per-gpu=32G`
is generous for the reference configuration.

## 3. Time limits, checkpoints and resuming

A run writes to `runs/<name>/`:

| path | content |
|---|---|
| `config.yaml` | the resolved configuration |
| `train.log` | the log of every (re)start, appended |
| `logs/version_N/metrics.csv` | Lightning's CSV log; one `version_N` per (re)start (`pandas.concat` them and drop duplicate `step`s) |
| `eval/deletion_metrics.csv` | one row per epoch and bucket (plus `all`): confusion counts, precision, recall, F1, chance, ranking proxy, insertion/substitution events |
| `eval/val_epoch_NNNN.json` | the raw counts of that epoch |
| `checkpoints/best-epoch=E-step=S.ckpt` | best `val/del_f1` |
| `checkpoints/last.ckpt` | written after every validation epoch |
| `checkpoints/periodic.ckpt` | written every `checkpoint.time_interval_min` minutes of training (30 by default) |
| `slurm-<job>-<id>.out` | stdout/stderr of the job (progress bar updates are reduced when not on a terminal) |

Starting the same run name again continues from the newest of `last.ckpt` / `periodic.ckpt`; at most
`time_interval_min` minutes of work are lost when a job is killed by its time limit. `from_ckpt=<path>`
resumes from a specific checkpoint, `from_ckpt=none` forces a fresh start (use a new name instead of
overwriting a run directory).

For runs that need more than one job's wall time, submit a chain:

```bash
scripts/submit_chain.sh 5 <run name>        # 5 jobs, each starts when the previous one has ended
```

Lightning's built-in SLURM auto-requeue is disabled on purpose (`SLURMEnvironment(auto_requeue=False)`): it
writes the checkpoint inside the signal handler, which can fire in the middle of a step and is fragile
under DDP. Resubmission with periodic checkpoints is simpler and does not depend on signal timing.

## 4. Hydra's submitit launcher (alternative to the sbatch script)

`config/hydra/launcher/slurm.yaml` configures Hydra's submitit launcher, e.g. for sweeps:

```bash
python train.py -m hydra/launcher=slurm slurm.account=<project> slurm.partition=<gpu partition> \
    experiment=deletion-base name=sweep_lr_\${task.optimizer.lr} task.optimizer.lr=1e-4,3e-4
```

Each job of a sweep resumes from its own run directory like any other run; resubmit the same command to
continue after a time limit.

## 5. Multi-GPU details

* The datamodule builds its own samplers (`use_distributed_sampler: no`): a `DistributedSampler` for
  training and Marten Lienen's non-padding sampler for validation / test, so every held-out sequence is
  evaluated exactly once across processes.
* The deletion metrics are integer counts per bucket; every process evaluates its share of the fixed
  sampling subset and the counts are all-reduced before the rates are computed, so the logged values are
  identical on every rank. `val/loss` and `test/loss` are averaged across processes by Lightning.
* The coupling and the known alignment run in the data workers (`data.num_workers`), as in upstream.
  Seeds are derived per process from `seed`.
* `bf16-mixed` precision is the default of `experiment=deletion-base`. `torch.compile` stays off (see the
  note in `editflows/pretraining.py` about nested tensors).

## 6. Reading the results

`evaluate.py` scores any checkpoint on the validation or test split with annotated examples:

```bash
python evaluate.py runs/<name>/checkpoints/best-epoch=*-step=*.ckpt -n 512 --show 4 --out eval.json
python evaluate.py runs/<name>/checkpoints/last.ckpt --original-fraction 0.5 0.5 --split test
```

The walkthrough notebook (`notebooks/walkthrough.ipynb`) plots `eval/deletion_metrics.csv` and the
Lightning CSV log of a run directory; copy `runs/<name>` (without the large checkpoints if you like) to a
machine with Jupyter.
