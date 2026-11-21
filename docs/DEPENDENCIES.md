# Dependencies and environments

Single source of truth for versions: [`pyproject.toml`](../pyproject.toml). `torch` is deliberately *not* listed
there because the CPU and CUDA builds come from different package indexes; the two requirements files below pick one.

| component | version | why / notes |
|---|---|---|
| Python | 3.13 (tested with 3.13.17) | upstream pins 3.13.5; every dependency has wheels for it on Windows and Linux. The hard floor from the dependencies themselves is 3.11 (`jaxtyping`), which matters only on a machine where 3.13 cannot be installed |
| torch | 2.13.0 (`+cpu` locally, `+cu126` on GPU nodes) | same version on both sides; cu126 covers A100/H100-class GPUs (and Ampere consumer cards such as the RTX 3090), use the cu128 index for GPUs that need CUDA >= 12.8. The wheels bundle their own CUDA runtime as `nvidia-*-cu12` packages and never use a system CUDA toolkit, so the `nvcc` version on a node is irrelevant - but the **driver** must be >= 525.60.13 for any CUDA 12.x build |
| transformers | 5.14.1 | the Llama trunk used by the placeholder encoder (`editflows/llama.py`, `editflows/encoder.py`) |
| lightning | 2.6.6 (PyPI) | training loop, DDP, checkpoints. Upstream used a patched git branch for SLURM auto-requeue; this fork disables auto-requeue instead (see `docs/CLUSTER.md`) |
| hydra-core, hydra-submitit-launcher, submitit | 1.3.4, 1.2.0, 1.5.4 | configuration and optional SLURM launcher |
| numba | 0.66.0 | the Needleman-Wunsch aligner (`OptimalAligner`), only used for tests / comparisons here |
| jaxtyping, einops, rich, matplotlib, pandas, openpyxl, tqdm | see `pyproject.toml` | typing annotations, tensor ops, console output, plots, Excel/CSV reading |
| dev extra: pytest, ipykernel, ipdb | - | tests, Jupyter kernel, debugging |
| optional extra `wandb` | 0.28.1 | only if you want Weights & Biases logging (`logging=wandb`); CSV logging is the default |

Removed compared to upstream: `datasets`, `sentencepiece`, `loky`, `jupyterlab` (text8 pipeline only), the `pixi`
configuration and `pixi.lock` (Linux-only lock that no longer matched the dependency list).

## Local CPU environment (Windows, this is what the code is developed and smoke-tested with)

```powershell
py install 3.13                                   # Python install manager; skip if 3.13 is already installed
py -V:3.13 -m venv $HOME\0envs\EditFlows_env
$HOME\0envs\EditFlows_env\Scripts\python.exe -m pip install -U pip
$HOME\0envs\EditFlows_env\Scripts\python.exe -m pip install -r requirements-cpu.txt
$HOME\0envs\EditFlows_env\Scripts\python.exe -m ipykernel install --user --name EditFlows_env --display-name "Python (EditFlows_env)"
```

`requirements-cpu.txt` installs `torch==2.13.0+cpu` from the PyTorch CPU index and then the package itself in
editable mode with the `dev` extra. The Jupyter kernel is registered under the name `editflows_env` (ipykernel
lower-cases it); JupyterLab itself can live in any other environment.

## GPU environment (Linux cluster)

```bash
module purge
module load <a Python 3.13 module of your site>    # or any python3.13 on PATH
python -m venv <project-area>/envs/EditFlows_env    # not in a scratch area that is cleaned automatically
source <project-area>/envs/EditFlows_env/bin/activate
pip install -U pip
pip install -r requirements-cuda.txt
```

Do not use `pip install --user` on shared systems (packages leak into every environment). Build the environment on a
login node with internet access; compute nodes often have none.

Check the driver before installing, because it decides whether the pinned wheel can load at all:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

| driver | action |
|---|---|
| >= 525.60.13 | `requirements-cuda.txt` as written (`torch==2.13.0+cu126`) |
| 455 - 525 | CUDA 12.x wheels will not load, and torch 2.13 ships no cu118 build. Pin back to the newest torch with both a cu118 and a cp313 wheel; the other pins only ask for `torch>=2.4` (`transformers`) and `torch>=2.1,<4.0` (`lightning`), so nothing else has to move. Verify with the test suite before using it on the cluster |
| < 455 | too old for an Ampere card; a driver update is an administrator's job |

Then verify, in this order - the bf16 matmul is the real gate, since `bf16-mixed` is the precision the GPU
experiments use:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print((a@a).float().sum().item())"
python -m pytest -q tests -m "not slow"
```

### Clusters without a module system

Where there is no `module load` and the system Python is older than 3.13, use [uv](https://docs.astral.sh/uv/): it
downloads a prebuilt standalone CPython, so nothing is compiled and no system headers are needed. Install uv from
PyPI rather than with `curl | sh` - a proxied cluster often reaches PyPI while `astral.sh` is blocked.

```bash
python3 -m venv ~/0envs/bootstrap                  # a throwaway env that only holds uv
~/0envs/bootstrap/bin/pip install -U pip uv
export PATH="$HOME/0envs/bootstrap/bin:$PATH"

uv python install 3.13                             # prebuilt, no compilation
uv venv --python 3.13 ~/0envs/EditFlows_env
uv pip install --python ~/0envs/EditFlows_env -r requirements-cuda.txt
~/0envs/EditFlows_env/bin/python -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

If `pip` dies with `Tunnel connection failed: 403 Forbidden`, an **explicit** proxy is configured
(`https_proxy` in the environment, or a `pip.conf`) and is refusing the `CONNECT` to PyPI. A transparent
redsocks proxy needs no such variables, so removing them is usually the fix:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY     curl -sS -o /dev/null -w '%{http_code}
' https://pypi.org/simple/
```

pip needs both `pypi.org` and `files.pythonhosted.org`, and torch additionally `download.pytorch.org`.
Note that `uv` and the standalone CPython it installs come from GitHub releases rather than PyPI, so the
interpreter can often be built while PyPI is still blocked.

Miniforge is an equivalent fallback (also no compilation). `pyenv` is the last resort: it builds CPython from source
and needs `libssl-dev`, `libffi-dev`, `zlib1g-dev`, `libbz2-dev`, `libsqlite3-dev`, `libreadline-dev` and
`liblzma-dev`, which cannot be installed without root - and a build with a missing `_ssl` yields a Python that
cannot reach PyPI at all, which only becomes apparent after the compile. Verify any such build immediately with
`python -c "import ssl, lzma, sqlite3, ctypes"`.

Keep the environment **outside** the clone, so `git pull`, `git switch` and `git clean` can never touch it. Because
`requirements-cuda.txt` installs the project with `-e .[dev]`, the clone is the installed package: code changes need
only `git pull`, and reinstalling is for dependency changes.

`docs/CLUSTER.md` has the full worked example for the UiO CCSE cluster.

## Exact reproduction

`requirements-lock.txt` is a `pip freeze` of the local CPU environment at the time of the last commit that changed
dependencies. It pins every transitive package; use it when an exact copy is needed:

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
