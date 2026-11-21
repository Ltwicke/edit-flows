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
`--account` in `scripts/train.sbatch` first. The cluster this fork is used on has **no scheduler at
all** - see the appendix for `scripts/train_detached.sh`, which replaces this script there.
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

# Appendix: the UiO CCSE "Egil" cluster

Everything above is site-neutral. This appendix is the concrete runbook for the cluster at
<https://uio-ccse.github.io/kvv-tungregning/>.

| node | GPUs | state (checked 2026-11-21) |
|---|---|---|
| **`hugefacet`** | 4x A100-PCIE-40GB | **the target.** Driver new enough for CUDA 12, and idle |
| `greatfacet` | 6x A100-PCIE-40GB | equivalent; the obvious second choice |
| `neuroai` | 1x RTX 3090 | **unusable — no working NVIDIA driver** |
| `bioai` | 2x RTX 2080 | too small |
| `bigfacet` | 8x P100-16GB | Pascal: no bf16, would need `precision: 16-mixed` or `32` |
| `egil-analyze` | 1x RTX 3090 | serves the NFS home; also runs JupyterHub |

Three facts shape the whole setup, and none of them is in the site documentation:

* **There is no batch scheduler.** `sbatch`, `squeue`, `sinfo` and `scontrol` do not exist on this
  cluster, despite what the site docs imply. Runs are started directly on a node with
  `scripts/train_detached.sh`; `scripts/train.sbatch` and `scripts/submit_chain.sh` are for SLURM sites
  and are not used here. **There is therefore no admission control** — check the GPUs are free before
  starting, because nothing else will.
* **`$HOME` is one NFS filesystem**, created on `egil-analyze` and mounted on every node with the same
  UID. One clone and one environment serve every node. The home is shared by all users with no quota,
  and filling it stops everyone's work.
* **`lab-login.uio.no` forbids `ssh -J`, `ProxyJump`, `ProxyCommand` and `ssh -L`.** So `scp` and
  `rsync` from your laptop do not work at all, and the Jupyter port-forward the site documents cannot
  be assumed to work from off site. Both are designed around below.

## A.1 Logging in

Three hops; the last uses your **cluster** password, not your UiO password.

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

From `mn-fys-ccse-nat`, reach the GPU node with `ssh hugefacet`.

## A.2 Checking a node before you use it

Run on the node you intend to use. The driver version is what decides whether the pinned `torch` can
load at all, and without a scheduler the second check is the only thing standing between you and
someone else's job:

```bash
ssh egil-cluster
ssh hugefacet

nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
nvidia-smi                                 # IS ANYONE ELSE ON THE CARDS?
nproc; free -g | head -2; df -h ~ | tail -1
```

| observation | consequence |
|---|---|
| `driver_version >= 525.60.13` | `requirements-cuda.txt` works as written |
| `compute_cap` `8.0` (A100) or `8.6` (3090) | `bf16-mixed` is supported |
| `compute_cap` `6.0` (`bigfacet`, P100) | no bf16 — add `trainer.precision=32` |
| another user's process in `nvidia-smi` | pick different GPUs with `EDITFLOW_GPUS=`, or another node |

## A.3 Where everything lives

```
~/edit-flows/                     the clone; development happens on the laptop and arrives by git pull
~/edit-flows/data/synthetic/      the generated sequence pool and its .npz cache (gitignored)
~/edit-flows/runs/<name>/         everything a run writes
~/0envs/EditFlows_env/            the virtual environment, deliberately OUTSIDE the clone
```

One clone and one environment on the shared NFS home — every node sees them, and there is nothing to
install per node. The environment is outside the clone so `git pull`, `git switch` and `git clean` can
never touch it.

## A.4 The clone, and a key for pushing back

The repository is public, so cloning needs no credentials:

```bash
cd ~
git clone https://github.com/Ltwicke/edit-flows.git edit-flows
cd ~/edit-flows
git switch on-GPU-cluster-branch
```

Pushing results back (A.10) does need credentials. Generate a key **on the cluster** and add the public
half to GitHub under *Settings -> SSH and GPG keys*:

```bash
ssh-keygen -t ed25519 -C "egil-cluster" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub            # paste into GitHub
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

## A.5 Getting through the proxy

**This is the step that fails first on this cluster.** A `pip install` typically dies with

```
ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))
ERROR: No matching distribution found for uv
```

That error means pip found an **explicit** proxy — from `https_proxy` in the environment or from a
`pip.conf` — and that proxy refused the `CONNECT` to PyPI. It is not redsocks: redsocks works
transparently at the packet level and needs no environment variables at all. Setting them routes pip
through a different, restrictive proxy instead.

Work through this in order and stop at the first that returns `200`:

```bash
# 1. What is actually configured?
env | grep -i proxy
cat /etc/environment 2>/dev/null | grep -i proxy
cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null

# 2. Which hosts are reachable, with the proxy as configured?
for u in https://github.com https://pypi.org/simple/ https://files.pythonhosted.org \
         https://download.pytorch.org/whl/cu126/; do
  printf '%-45s %s\n' "$u" "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$u" 2>&1)"
done

# 3. The same, with every proxy variable removed - this is usually the fix
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY bash -c '
for u in https://pypi.org/simple/ https://files.pythonhosted.org \
         https://download.pytorch.org/whl/cu126/; do
  printf "%-45s %s\n" "$u" "$(curl -sS -o /dev/null -w "%{http_code}" --max-time 20 "$u" 2>&1)"
done'
```

If step 3 gives `200`, unset the variables for the whole session and continue with A.6:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
# make it stick:
printf '\nunset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY\n' >> ~/.bashrc
```

`git clone` over HTTPS working while pip fails is the giveaway: git and pip read proxy settings from
different places, so one can be routed through the broken proxy while the other is not.

**If PyPI is unreachable either way**, pip needs `pypi.org` *and* `files.pythonhosted.org`, and torch
additionally needs `download.pytorch.org`. Ask the CCSE admins (Henrik Andersen Sveinsson,
h.a.sveinsson@fys.uio.no) to allow those three, or for the address of a local mirror — a cluster whose
users cannot install Python packages is broken, and the site docs do claim https works. Two things make
that wait less painful:

* `uv` itself, and the standalone CPython it installs, come from **GitHub releases**, not PyPI. If
  github.com is reachable, A.6 works even while PyPI is blocked — you just cannot install the
  dependencies yet.
* As a genuine last resort the wheels can be carried in by hand. Note that git is **not** a usable
  channel: the torch wheel alone is far over GitHub's 100 MB per-file limit. GitHub *Releases* allow
  2 GB per asset, so build the set on any Linux machine with
  `pip download -r requirements-cuda.txt -d wheels --only-binary=:all: --platform manylinux_2_28_x86_64 --python-version 3.13`,
  attach `wheels/` to a release on your own repo, then on the cluster `curl -LO` each and
  `pip install --no-index --find-links wheels -r requirements-cuda.txt`. It is roughly 4 GB.

## A.6 Python 3.13 and the environment

The project needs Python >= 3.13; the nodes have 3.8.10 and there is no module system. Use **uv**: it
downloads a prebuilt standalone CPython, so nothing is compiled and no system headers are needed.

```bash
python3 -m venv ~/0envs/bootstrap
~/0envs/bootstrap/bin/pip install -U pip uv
export PATH="$HOME/0envs/bootstrap/bin:$PATH"          # add to ~/.bashrc to keep it
uv --version
```

If PyPI is blocked but github.com is not (A.5), take uv straight from its GitHub release instead:

```bash
cd ~ && curl -LO https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar xzf uv-x86_64-unknown-linux-gnu.tar.gz
mkdir -p ~/bin && mv uv-x86_64-unknown-linux-gnu/uv ~/bin/ && export PATH="$HOME/bin:$PATH"
```

Then, either way:

```bash
uv python install 3.13                                 # prebuilt, from GitHub; nothing is compiled
uv venv --python 3.13 ~/0envs/EditFlows_env
~/0envs/EditFlows_env/bin/python -V                    # expect Python 3.13.x
~/0envs/EditFlows_env/bin/python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

Fallbacks if uv cannot be made to work at all: **Miniforge** (also compiles nothing, installer from
GitHub releases), and only then `pyenv`, which the site docs recommend but which builds CPython from
source and needs `libssl-dev libffi-dev zlib1g-dev libbz2-dev libsqlite3-dev libreadline-dev
liblzma-dev` — unavailable without root, and a build missing `_ssl` yields a Python that cannot reach
PyPI at all, discovered only after a long compile. Verify any such build with
`python -c "import ssl, lzma, sqlite3, ctypes"`.

If 3.13 proves impossible, the real floor across the pinned dependencies is **3.11** (`jaxtyping`); a
3.11 or 3.12 interpreter works at the cost of one line in `pyproject.toml`.

## A.7 torch, and why `nvcc` does not matter

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
| 455 - 525 | CUDA 12 wheels will not load, and torch 2.13 ships no cu118 build. Pin back to the newest torch with both a cu118 and a cp313 wheel; the other pins only require `torch>=2.4`, so nothing else moves. Test on the laptop first |
| < 455 | too old for an Ampere card — this is what rules `neuroai` out. An administrator's problem |

Verify in this order; the bf16 matmul is the real gate, because `bf16-mixed` is the precision both GPU
experiments use:

```bash
P=~/0envs/EditFlows_env/bin/python
$P -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$P -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
$P -c "import torch; a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); print((a@a).float().sum().item())"
$P -m pytest -q tests -m "not slow"          # expect 51 passed
```

## A.8 The sequence pool

Generate it on the cluster — `scp` is not available, and the generator is seeded, so the same command
reproduces the same sequences on your laptop:

```bash
cd ~/edit-flows
~/0envs/EditFlows_env/bin/python scripts/make_synthetic_targets.py \
    --n 60000 --min-len 100 --max-len 500 --seed 0 \
    --out data/synthetic/targets-60k.txt
```

About 18 MB and a few minutes. Both GPU experiments point at this path. The train split is 90% =
54 000 sequences, so a 25 000-sequence epoch draws a different ~46% of the pool each time. The first
run parses the file once and caches it as a `.npz` beside it.

## A.9 Running

There is no scheduler, so runs start directly on the node and detach. `scripts/train_detached.sh`
checks the GPUs are free, sets `CUDA_VISIBLE_DEVICES`, derives the worker count, and starts the run
under `setsid nohup` so it survives logout.

**The short run first** — two epochs of 512 sequences on one GPU, a couple of minutes. It is not a
training result; it proves the environment, the GPU, bf16, the metrics and the run directory:

```bash
ssh egil-cluster && ssh hugefacet
cd ~/edit-flows
EDITFLOW_EXPERIMENT=deletion-gpu-check EDITFLOW_GPUS=0 scripts/train_detached.sh gpu-check
tail -f runs/gpu-check/train.log
grep "val epoch" runs/gpu-check/train.log      # two epochs with precision and recall
```

**Then the long run.** Read the real epoch time off the short run first and choose `max_epochs`
(section 3.7 of `notebooks/on-node-training-analysis.ipynb` does the arithmetic):

```bash
scripts/train_detached.sh long-01                        # every free GPU on the node
scripts/train_detached.sh long-01 trainer.max_epochs=8   # the same setup, roughly 30 minutes
EDITFLOW_GPUS=0 scripts/train_detached.sh long-01        # a single GPU
```

Start on **one** GPU and let the measurement decide whether more help. Training scales with the GPU
count, but the per-epoch sampling evaluation is `sample_steps` *sequential* forward passes and barely
does, so four GPUs can simply mean the evaluation dominates. One A100 is a great deal of machine for a
5M-parameter model.

Useful while it runs:

```bash
tail -f runs/long-01/train.log                 # per-epoch deletion metrics
tail -f runs/long-01.detached.out              # stdout/stderr, including a crash
nvidia-smi
kill $(cat runs/long-01.pid)                   # stop it
scripts/train_detached.sh long-01              # continue from the newest checkpoint
```

Re-running the same run name **resumes** from the newest of `checkpoints/last.ckpt` and
`periodic.ckpt`; `periodic.ckpt` is written every 30 minutes, so a crash or a killed session loses
little. This is checkpoint-based, not scheduler-based, so it works exactly as it would under SLURM.

Do not add `trainer.max_time` to get a shorter run: lightning restores the timer cumulatively and a
resumed run would stop at once. Use `trainer.max_epochs=<N>`.

## A.10 Getting the numbers onto your laptop

`scp` and `rsync` cannot reach this cluster, so results travel by git. On the cluster:

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

The files land in `runs/long-01/` and stay ignored locally. Then open
`notebooks/on-node-training-analysis.ipynb`, which needs no GPU and no checkpoints.

## A.11 Jupyter on a node

JupyterHub is installed on `egil-analyze`, `neuroai` and `bioai` — note that `hugefacet` is **not** in
that list, so the analysis kernel runs on a different node from the training. That is fine: the
notebook needs neither a GPU nor the checkpoints, and `$HOME` is shared, so it can read the run
directory from `egil-analyze`.

```bash
mkdir -p ~/.local/share/jupyter/kernels
~/0envs/EditFlows_env/bin/python -m ipykernel install --user \
    --name editflows_env --display-name "Python (EditFlows_env)"
jupyter kernelspec list
```

Both notebooks expect exactly that kernel name. Then, in order:

1. **Is it listening?** On the node: `ss -ltnp | grep 8000`. The site docs never give a command to
   start JupyterHub, so if nothing is listening, ask the admins — it is a system service.
2. **On the CCSE wired network**, the site's own one-liner works:
   `ssh -L 8000:localhost:8000 <user>@egil-analyze`, then open <http://localhost:8000>.
3. **From off site**, first find out whether forwarding is possible at all, since `lab-login` is
   documented as refusing it:

   ```bash
   ssh -v -N -L 18822:localhost:22 <user>@lab-login.uio.no     # terminal 1
   ssh -p 18822 <user>@localhost                               # terminal 2
   ```

   `channel 1: open failed: administratively prohibited` means forwarding is off and no tunnel is
   possible through that hop. If it does work, chain `-L` hop by hop — each opened *inside* the
   previous session, which is why the single `RemoteCommand` line cannot do it:

   ```bash
   ssh -L 8000:localhost:8000 <user>@login.uio.no
     ssh -L 8000:localhost:8000 <user>@lab-login.uio.no
       ssh -L 8000:localhost:8000 <user>@mn-fys-ccse-nat.lab.uio.no
         ssh -L 8000:localhost:8000 egil-analyze
   ```

4. **The reliable path, and the recommended default: no tunnel.** Pull the artifact branch (A.10) and
   run the notebook on your laptop. Nothing in it needs a GPU, a checkpoint or the cluster — which is
   exactly why A.10 exists.

## A.12 Picking up changes from the laptop

```bash
cd ~/edit-flows && git pull
```

That is all that is needed for code changes: `requirements-cuda.txt` installs the project with
`-e .[dev]`, so the clone *is* the installed package. Only when the dependencies change:

```bash
uv pip install --python ~/0envs/EditFlows_env -r requirements-cuda.txt
~/0envs/EditFlows_env/bin/python -m pytest -q tests -m "not slow"
```

## A.13 When something goes wrong

| symptom | cause |
|---|---|
| `pip` fails with `Tunnel connection failed: 403 Forbidden` | an explicit proxy is configured; see A.5 |
| `sbatch: command not found` | correct — this cluster has no scheduler. Use `scripts/train_detached.sh` |
| `refusing to start; pick free GPUs` | another process holds the cards; `nvidia-smi`, then `EDITFLOW_GPUS=` or another node |
| `run '<name>' is already running as pid N` | one run directory, two processes would corrupt it. `kill $(cat runs/<name>.pid)` or use a new name |
| `CUDA error: no kernel image is available` | the wheel's CUDA build does not match the driver — back to A.7 |
| `torch.cuda.is_available()` is `False` | on `neuroai`, expected: the driver is broken. Elsewhere, check `nvidia-smi` works at all |
| `$'\r': command not found` | the script was checked out with CRLF; `.gitattributes` forces LF, so re-clone or `git config core.autocrlf input` |
| `ConfigCompositionException: Multiple values for experiment` | `experiment=` was passed as an argument; use `EDITFLOW_EXPERIMENT=` |
| hydra `name: ???` is missing | every run needs a name; only `deletion-gpu-check` defaults it |
| a run restarts from scratch unexpectedly | a different `name=`; the run name is the identity of the run |
| epochs are far smaller than 25 000 sequences | expected with several GPUs: `limit_train_batches` is per rank and the formula divides by `gpus` |
| disk full, everyone's work stops | the shared home; `du -h --max-depth=1 ~` and delete old checkpoints |
