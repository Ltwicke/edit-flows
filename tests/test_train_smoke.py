"""End-to-end smoke test of train.py and evaluate.py on a tiny dataset (CPU, about a minute)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout + result.stderr


@pytest.mark.slow
def test_train_resume_and_evaluate(tmp_path):
    data = tmp_path / "targets.csv"
    run("scripts/make_synthetic_targets.py", "--n", "80", "--min-len", "20", "--max-len", "40", "--out", str(data))
    run_dir = tmp_path / "run"
    overrides = [
        "experiment=deletion-smoke",
        f"run_dir={run_dir}",
        f"data.path={data}",
        "data.max_len=40",
        "data.batch_size=8",
        "trainer.limit_train_batches=3",
        "trainer.limit_val_batches=1",
        "trainer.log_every_n_steps=1",
        "task.eval.n_sample_seqs=4",
        "task.eval.sample_steps=3",
        "task.model.encoder.hidden_size=16",
        "task.model.encoder.num_attention_heads=2",
    ]

    out = run("train.py", *overrides, "trainer.max_epochs=1")
    assert "Resuming from" not in out
    for name in ("config.yaml", "train.log", "checkpoints/last.ckpt", "logs/version_0/metrics.csv",
                 "eval/deletion_metrics.csv", "eval/val_epoch_0000.json"):
        assert (run_dir / name).is_file(), name
    metrics = pd.read_csv(run_dir / "logs/version_0/metrics.csv")
    assert {"train/loss", "val/loss", "val/del_f1", "val/rank_p_at_k"} <= set(metrics.columns)

    # The same command continues from last.ckpt into the second epoch
    out = run("train.py", *overrides, "trainer.max_epochs=2")
    assert "Resuming from" in out and "last.ckpt" in out
    assert (run_dir / "eval/val_epoch_0001.json").is_file()
    assert (run_dir / "logs/version_1/metrics.csv").is_file()
    table = pd.read_csv(run_dir / "eval/deletion_metrics.csv")
    assert sorted(table["epoch"].unique().tolist()) == [0, 1]

    out = run("evaluate.py", str(run_dir / "checkpoints/last.ckpt"), "-n", "4", "-s", "3", "--show", "1",
              "--out", str(tmp_path / "metrics.json"))
    assert "verdict" in out
    result = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "del_f1" in result["summary"]
