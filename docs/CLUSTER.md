# Training on a SLURM cluster

Everything here is site-neutral: replace the `<...>` placeholders with your project account, partition,
module and path names.

## 1. One-time setup on a login node

```bash
git clone https://github.com/Ltwicke/edit-flows.git && cd edit-flows
module purge
module load <Python 3.13 module>                     # or any python3.13 on PATH; where there is no
                                                     #  module system see the appendix (uv)
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

Fill in `--partition`, the environment path and, only where the site uses SLURM accounting,
`--account` in `scripts/train.sbatch` first (already filled in for Egil, see the appendix).
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

`notebooks/on-node-training-analysis.ipynb` is the companion to this guide: it traces an `sbatch`
override through hydra's composition into a config node, explains both log surfaces of a run directory,
and plots the loss and the per-epoch precision/recall of the delete operation. It needs no GPU and no
checkpoints, so it runs wherever the run's text files are. `notebooks/walkthrough.ipynb` covers the
model and the objective instead.

To move a run's numbers to another machine, use `scripts/push_run_artifacts.sh <name>`: it pushes the
resolved config, `train.log`, the Lightning CSVs and `eval/deletion_metrics.csv` (a few hundred KiB) to
a `run-artifacts/<name>` branch, leaving the checkpoints behind. On the analysis machine,
`git restore --source origin/run-artifacts/<name> -- runs/` drops them into place without changing
branch. This is the only way to get results off a cluster that forbids `scp` and `rsync`.

---

# Appendix: the UiO CCSE "Egil" cluster (`neuroai`)

Everything above is site-neutral. This appendix is the concrete runbook for the cluster at
<https://uio-ccse.github.io/kvv-tungregning/>, where the GPU nodes are

| node | GPUs | note |
|---|---|---|
| `neuroai` | 1x RTX 3090 (24 GB) | the target of `scripts/train.sbatch`; also runs JupyterHub |
| `egil-analyze` | 1x RTX 3090 | **serves the NFS home**; runs JupyterHub |
| `hugefacet` / `greatfacet` | 4x / 6x A100-40GB | the docs say long jobs belong here |
| `bigfacet` | 8x P100-16GB | older; no bf16 |
| `bioai` | 2x RTX 2080 | runs JupyterHub |

Two facts shape the whole setup:

* **`$HOME` is one NFS filesystem**, created on `egil-analyze` and mounted on every node with the same
  UID. One clone and one environment serve every node — there is nothing to install per node. The home
  is shared by all users with no quota, and a full filesystem stops every job on the cluster, so keep
  large files out of it.
* **`lab-login.uio.no` forbids `ssh -J`, `ProxyJump`, `ProxyCommand` and `ssh -L`.** So `scp` and
  `rsync` from your laptop do not work at all, and the Jupyter port-forward the site documents cannot
  be assumed to work from off site. Both are designed around below.

## A.1 Logging in

Three hops; the last one uses your **cluster** password, not your UiO password.

```bash
ssh <uio-username>@login.uio.no
ssh <uio-username>@lab-login.uio.no
ssh <uio-username>@mn-fys-ccse-nat.lab.uio.no
```

Put this in your laptop's `~/.ssh/config` to make it one command (`ssh egil-cluster`):

```
Host egil-cluster
    HostName login.uio.no
    User <uio-username>
    RequestTTY yes
    RemoteCommand ssh -t lab-login.uio.no ssh -t mn-fys-ccse-nat.lab.uio.no
```

From `mn-fys-ccse-nat` reach a node with `ssh neuroai` (`griffith` only answers on `10.255.80.18`).

## A.2 Step 0 — the diagnostics, before installing anything

The site docs do not state the NVIDIA driver version, whether `neuroai` is managed by SLURM, its core
count, or the partition's time limit. Guessing any of them wastes a day, and the driver version decides
which `torch` wheel can be installed at all. Run this on **`neuroai` itself** and keep the output:

```bash
ssh egil-cluster
ssh neuroai

hostname; whoami
ls ~ > /dev/null && echo "HOME readable"; df -h ~ | tail -1      # is the NFS home here?
mount | grep -i nfs | head -3
nproc; free -g | head -2; ldd --version | head -1                # cores, RAM, glibc
cat /etc/os-release | head -2

nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
nvidia-smi                                                        # is anyone else on the card?

python3 -V; python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
python3 -m venv --help > /dev/null 2>&1 && echo "venv ok" || echo "NO ensurepip"
env | grep -i proxy
curl -sI https://pypi.org/simple/ | head -1                       # redsocks working?
curl -sI https://download.pytorch.org/whl/cu126/ | head -1
command -v tmux screen uv git
```

And on the gateway, for the scheduler:

```bash
sinfo -s
sinfo -N -o "%N %P %t %C %m %G"
scontrol show node neuroai
scontrol show partition
scontrol show config | grep -i -E "MaxTime|DefaultTime|GresTypes|AccountingStorageType"
squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %R %b"
```

How to read the answers:

| observation | consequence |
|---|---|
| `driver_version >= 525.60.13` | `requirements-cuda.txt` works as written. See A.6 otherwise |
| `compute_cap` is `8.6` | RTX 3090, Ampere — `bf16-mixed` is supported |
| `scontrol show node neuroai` says *not found* | `neuroai` is outside SLURM — use **A.10**, not A.8/A.9 |
| `%G` shows no `gpu:` for the node | SLURM does not know about its GPU; `--gres=gpu:1` will not work |
| `AccountingStorageType=.../none` | no `--account` is needed (the script has none) |
| `MaxTime` below `23:30:00` | lower `--time` and chain jobs (A.9) |
| `curl` to pypi.org does not return `200` | fix the proxy before A.5, nothing will install |
| `nproc` far below 8 | lower `--cpus-per-task` in `scripts/train.sbatch` |

## A.3 Where everything lives

```
~/edit-flows/                     the clone; development happens on the laptop and arrives by git pull
~/edit-flows/data/synthetic/      the generated sequence pool and its .npz cache (gitignored)
~/edit-flows/runs/<name>/         everything a run writes
~/0envs/EditFlows_env/            the virtual environment, deliberately OUTSIDE the clone
~/0envs/bootstrap/                a throwaway env that only holds uv
```

One clone, one environment, on the shared NFS home — every node sees them. The environment is outside
the clone so `git pull`, `git switch` and `git clean` can never touch it.

## A.4 The clone, and a key for pushing back

The repository is public, so cloning needs no credentials:

```bash
cd ~
git clone https://github.com/Ltwicke/edit-flows.git edit-flows
cd ~/edit-flows
git switch on-GPU-cluster-branch
```

Pushing results back (A.11) does need credentials. Generate a key **on the cluster** and add the public
half to your GitHub account under *Settings -> SSH and GPG keys*:

```bash
ssh-keygen -t ed25519 -C "egil-cluster" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub            # paste this into GitHub
git remote set-url --push origin git@github.com:Ltwicke/edit-flows.git
ssh -T git@github.com                # expect "Hi <user>! You've successfully authenticated"
```

If that hangs, the proxy is blocking port 22. GitHub also serves SSH on 443; add to `~/.ssh/config`
**on the cluster**:

```
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

## A.5 Python 3.13 and the environment

The project needs Python >= 3.13; the nodes have 3.8.10 and there is no module system. Use **uv**: it
downloads a prebuilt standalone CPython, so nothing is compiled and no system headers are needed.
Install uv from PyPI rather than with `curl | sh`, because PyPI is known to work through the proxy
while `astral.sh` may not be reachable.

```bash
python3 -m venv ~/0envs/bootstrap
~/0envs/bootstrap/bin/pip install -U pip uv
export PATH="$HOME/0envs/bootstrap/bin:$PATH"          # add to ~/.bashrc to keep it
uv --version

uv python install 3.13
uv venv --python 3.13 ~/0envs/EditFlows_env
~/0envs/EditFlows_env/bin/python -V                    # expect Python 3.13.x
~/0envs/EditFlows_env/bin/python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

Fallbacks, in order, if that fails:

1. `python3 -m pip install --user uv` — if `python3 -m venv` has no `ensurepip`. Acceptable for a
   standalone tool binary; it does not leak into the project environment.
2. `curl -LsSf https://astral.sh/uv/install.sh | sh` — only if PyPI is unreachable but `astral.sh` is.
3. **Miniforge**, which also compiles nothing:
   `bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3` then
   `~/miniforge3/bin/conda create -y -p ~/0envs/EditFlows_env python=3.13`.
4. `pyenv`, which the site docs recommend, **last**: it compiles CPython from source and needs
   `libssl-dev libffi-dev zlib1g-dev libbz2-dev libsqlite3-dev libreadline-dev liblzma-dev`, which you
   cannot install without root. A build with a missing `_ssl` produces a Python that cannot reach PyPI
   at all — and you only find out after a long compile. Verify immediately with
   `python -c "import ssl, lzma, sqlite3, ctypes"`.

If 3.13 cannot be had at all, note that the real floor across the pinned dependencies is **3.11**
(`jaxtyping`); a 3.11 or 3.12 interpreter works at the cost of one line in `pyproject.toml`.

## A.6 torch, and why `nvcc` does not matter

The site docs report `nvcc 10.1.243`. **Ignore it.** The pip wheels bundle their own CUDA runtime as
`nvidia-*-cu12` packages and never use a system CUDA toolkit. Only the **driver** matters, because CUDA
12.x wheels need a driver >= 525.60.13.

```bash
cd ~/edit-flows
uv pip install --python ~/0envs/EditFlows_env -r requirements-cuda.txt
```

| `driver_version` from A.2 | what to do |
|---|---|
| >= 525.60.13 | the command above is correct (`torch==2.13.0+cu126`) |
| 455 – 525 | CUDA 12 wheels will not load, and torch 2.13 ships no cu118 build. Pin back to the newest torch that has both a cu118 and a cp313 wheel — the other pins only require `torch>=2.4`, so nothing else moves. Test it on the laptop first |
| < 455 | an RTX 3090 cannot work; ask the CCSE admins (Henrik Andersen Sveinsson) |

Then verify, in this order — the bf16 matmul is the real gate, because `bf16-mixed` is the precision
both GPU experiments use:

```bash
P=~/0envs/EditFlows_env/bin/python
$P -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$P -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
$P -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print((a@a).float().sum().item())"
$P -m pytest -q tests -m "not slow"          # expect 51 passed
```

## A.7 The sequence pool

Generate it on the cluster — `scp` is not available, and the generator is seeded, so the same command
produces the same sequences on your laptop:

```bash
cd ~/edit-flows
~/0envs/EditFlows_env/bin/python scripts/make_synthetic_targets.py \
    --n 60000 --min-len 100 --max-len 500 --seed 0 \
    --out data/synthetic/targets-60k.txt
```

About 18 MB and a few minutes. Both GPU experiments point at this path. The train split is 90% =
54 000 sequences, so a 25 000-sequence epoch draws a different ~46% of the pool each time. The first
run parses the file once and caches it as a `.npz` beside it.

## A.8 The short job: does any of this work?

```bash
cd ~/edit-flows
mkdir -p runs                                   # ONCE: --output is resolved at submit time
EDITFLOW_EXPERIMENT=deletion-3090-check sbatch scripts/train.sbatch gpu-check
squeue -u $USER
tail -f runs/slurm-editflow-deletion-*.out
```

Two epochs of 512 sequences with the full-size encoder and `bf16-mixed` — a couple of minutes. It is
not a training result; it proves the environment, the GPU, bf16, the metrics and the run directory.

Success looks like `runs/gpu-check/eval/deletion_metrics.csv` existing with two epochs:

```bash
column -s, -t < runs/gpu-check/eval/deletion_metrics.csv | grep -E "stage|all"
grep "val epoch" runs/gpu-check/train.log
```

## A.9 The long job

First read the real epoch time off the short run and pick `max_epochs` (section 3.7 of
`notebooks/on-node-training-analysis.ipynb` does this arithmetic for you):

```bash
grep "val epoch" runs/gpu-check/train.log      # the timestamps give seconds per epoch
```

Then:

```bash
sbatch scripts/train.sbatch long-01                          # up to 24 h, max_epochs=300
sbatch scripts/train.sbatch long-01 trainer.max_epochs=<N>   # a calibrated shorter run
```

For a roughly 30-minute run of the same setup, pass the `max_epochs` the calibration gives — do **not**
add `trainer.max_time`, because lightning restores the timer cumulatively and a resumed run would stop
at once.

Submitting `long-01` again continues it from its newest checkpoint; `periodic.ckpt` is written every 30
minutes, so a job killed by the time limit loses little:

```bash
sbatch scripts/train.sbatch long-01            # continue where it stopped
scripts/submit_chain.sh 3 long-01              # three dependent jobs, one run
```

Any directive can be overridden on the command line, because `sbatch` options beat `#SBATCH` lines:

```bash
sbatch --partition=<other> -w hugefacet --gres=gpu:4 --ntasks-per-node=4 \
       scripts/train.sbatch big-01             # 4 GPUs on another node, DDP
scancel <jobid>
```

## A.10 If `neuroai` is not managed by SLURM

The site docs name only `bigfacet`, `hugefacet` and `greatfacet` as the nodes long jobs go through
SLURM on, and `neuroai` is not among them. If A.2 showed `scontrol show node neuroai` failing, run
directly — and note that **there is then no admission control**: check the card is free first, because
another user can take it and your job will run out of memory.

```bash
ssh egil-cluster && ssh neuroai
cd ~/edit-flows && mkdir -p runs
nvidia-smi                                     # is the GPU free?
df -h ~ | tail -1                              # is the shared home healthy?

tmux new -s editflow                           # if tmux exists: detach with Ctrl-b d
~/0envs/EditFlows_env/bin/python train.py experiment=deletion-3090-24h name=long-01

# without tmux - survives logout (new session, no controlling terminal, stdin closed):
setsid nohup ~/0envs/EditFlows_env/bin/python train.py \
    experiment=deletion-3090-24h name=long-01 \
    > runs/long-01.nohup.out 2>&1 < /dev/null &
echo "pid $!"
```

Monitor with `tail -f runs/long-01/train.log` and `nvidia-smi`. Resuming is unchanged: `resume_path()`
is checkpoint-based, not SLURM-based, so re-running the same `name=` continues from the newest
checkpoint.

## A.11 Getting the numbers onto your laptop

`scp` and `rsync` cannot reach this cluster, so the results travel by git. On the cluster:

```bash
cd ~/edit-flows
scripts/push_run_artifacts.sh long-01
```

That pushes only the text artifacts — the resolved config, `train.log`, the lightning CSVs and
`eval/deletion_metrics.csv`, a few hundred kilobytes — to a branch `run-artifacts/long-01`. Checkpoints
stay on the cluster; score them there with `evaluate.py`. It is safe to run while the job is still
training: it builds the commit with a throwaway index and never switches branch.

On your laptop:

```powershell
git fetch origin
git restore --source origin/run-artifacts/long-01 -- runs/
```

The files land in `runs/long-01/` and stay ignored locally, so they never pollute your working tree.
Then open `notebooks/on-node-training-analysis.ipynb`, which needs no GPU and no checkpoints.

## A.12 Jupyter on a node

JupyterHub is installed on `egil-analyze`, `neuroai` and `bioai`, on port 8000. Register the project
environment as a kernel so notebooks can import `editflows`:

```bash
mkdir -p ~/.local/share/jupyter/kernels
~/0envs/EditFlows_env/bin/python -m pip install ipykernel      # already in the dev extra
~/0envs/EditFlows_env/bin/python -m ipykernel install --user \
    --name editflows_env --display-name "Python (EditFlows_env)"
jupyter kernelspec list
```

Both notebooks expect exactly that kernel name. Then reach it, in this order:

1. **Is it even listening?** On the node: `ss -ltnp | grep 8000`. The site docs never give a command to
   start JupyterHub, so if nothing is listening, ask the admins — it is a system service.
2. **On the CCSE wired network**, the site's own one-liner works:
   `ssh -L 8000:localhost:8000 <user>@neuroai`, then open <http://localhost:8000>.
3. **From off site**, first find out whether forwarding is possible at all, since `lab-login` is
   documented as refusing it:

   ```bash
   ssh -v -N -L 18822:localhost:22 <user>@lab-login.uio.no     # terminal 1
   ssh -p 18822 <user>@localhost                               # terminal 2
   ```

   `channel 1: open failed: administratively prohibited` means forwarding is off and there is no tunnel
   through that hop. If it does work, chain `-L` hop by hop — each one opened *inside* the previous
   session, which is why the single `RemoteCommand` line cannot do it:

   ```bash
   ssh -L 8000:localhost:8000 <user>@login.uio.no
     ssh -L 8000:localhost:8000 <user>@lab-login.uio.no
       ssh -L 8000:localhost:8000 <user>@mn-fys-ccse-nat.lab.uio.no
         ssh -L 8000:localhost:8000 neuroai
   ```

   `login.uio.no` is shared, so `bind: Address already in use` there means someone else holds the port.
4. **The reliable path, and the recommended default: no tunnel.** Pull the artifact branch (A.11) and
   run the notebook on your laptop. Nothing in it needs a GPU, a checkpoint or the cluster, which is
   precisely why A.11 exists.

Note that the site also says JupyterHub is reachable only from the cabled CCSE network, so even a
working tunnel may be refused from the VPN. And do not run an analysis kernel on `neuroai` while a long
job owns the GPU.

## A.13 Picking up changes from the laptop

Development continues on your laptop. On the cluster:

```bash
cd ~/edit-flows
git pull
```

That is all that is needed for code changes: `requirements-cuda.txt` installs the project with
`-e .[dev]`, so the clone *is* the installed package. Only when the dependencies change:

```bash
uv pip install --python ~/0envs/EditFlows_env -r requirements-cuda.txt
~/0envs/EditFlows_env/bin/python -m pytest -q tests -m "not slow"
```

## A.14 When something goes wrong

| symptom | cause |
|---|---|
| `sbatch: error: ... Invalid account` | an `--account` crept back in; this site has no SLURM accounting |
| `sbatch` fails writing the output file | `mkdir -p runs` in the clone first — `--output` is resolved at submit time |
| `Batch job submission failed: Requested node configuration is not available` | `--gres=gpu:1` but SLURM does not know the node has a GPU, or `--nodelist` names a node outside the partition |
| job stuck `PD` with `ReqNodeNotAvail` | `neuroai` is down or not in the partition; check `sinfo -N` |
| `$'\r': command not found` | the script was checked out with CRLF; `.gitattributes` forces LF, so re-clone or `git config core.autocrlf input` |
| `CUDA error: no kernel image is available` | the wheel's CUDA build does not match the driver — back to A.6 |
| `torch.cuda.is_available()` is `False` inside a job but true on the node | the job got no GPU; check `--gres` reached SLURM (`scontrol show job <id>` -> `TresPerNode`) |
| `ConfigCompositionException: Multiple values for experiment` | `experiment=` was passed as an argument; use `EDITFLOW_EXPERIMENT=` instead |
| hydra `Could not override 'name'` / `name: ???` | every run needs `name=`; only `deletion-3090-check` defaults it |
| `pip` cannot reach PyPI | the proxy, or a Python built without `_ssl` (see A.5) |
| a run restarts from scratch unexpectedly | a different `name=`; the run name is the identity of the run |
| disk full, every job on the cluster stops | the shared home; `du -h --max-depth=1 ~` and delete old checkpoints |
