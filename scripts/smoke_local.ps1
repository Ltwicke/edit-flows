# Local CPU check (Windows): tests, placeholder data, a two-epoch smoke run and evaluate.py on its checkpoint.
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_local.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$python = if ($env:EDITFLOWS_PYTHON) { $env:EDITFLOWS_PYTHON } else { "$HOME\0envs\EditFlows_env\Scripts\python.exe" }

Set-Location (Join-Path $PSScriptRoot "..")

& $python -m pytest -q tests -m "not slow"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path "data\synthetic\targets.xlsx")) {
    & $python scripts\make_synthetic_targets.py --n 20000 --out data\synthetic\targets.xlsx
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $python train.py experiment=deletion-smoke from_ckpt=none
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python evaluate.py runs\smoke\checkpoints\last.ckpt -n 32 -s 20 --show 2
exit $LASTEXITCODE
