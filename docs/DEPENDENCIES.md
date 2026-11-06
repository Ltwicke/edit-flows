# Dependencies and environments

Single source of truth for versions: [`pyproject.toml`](../pyproject.toml). `torch` is deliberately *not* listed
there because the CPU and CUDA builds come from different package indexes; the two requirements files below pick one.

| component | version | why / notes |
|---|---|---|
| Python | 3.13 (tested with 3.13.17) | upstream pins 3.13.5; every dependency has wheels for it on Windows and Linux |
| torch | 2.13.0 (`+cpu` locally, `+cu126` on GPU nodes) | same version on both sides; cu126 covers A100/H100-class GPUs, use the cu128 index for GPUs that need CUDA >= 12.8 |
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

## Exact reproduction

`requirements-lock.txt` is a `pip freeze` of the local CPU environment at the time of the last commit that changed
dependencies. It pins every transitive package; use it when an exact copy is needed:

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
