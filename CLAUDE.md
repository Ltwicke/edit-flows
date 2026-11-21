# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project specifics (editflows)

Project-specific context to merge with the guidelines above.

**What this is.** A fork of Marten Lienen's Edit Flows implementing a *deletion-only* edit flow: the
source is a known target sequence with `K` uniform letters inserted at random positions, and the model
must delete exactly those. Alphabet `A`-`T` (ids 0-19) plus BOS (20). Operations are fixed by tensor row
index: **insert = 0, substitute = 1, delete = 2** (`editflows/editflows.py`, `LogRate`); there is no
keep operation, it is the implicit complement.

**Entry points.** `train.py` is the only training entry point, a hydra app
(`@hydra.main(config_path="config", config_name="train")`). No `ConfigStore` and no structured configs:
plain YAML plus `hydra.utils.instantiate` on `_target_` keys. `evaluate.py` scores a checkpoint.
Configuration is composed from `config/` groups; `config/experiment/*.yaml` carry `# @package _global_`
and are applied last, so they can override anything. `OmegaConf.register_new_resolver("eval", eval)` in
`train.py` is what makes `${eval:'...'}` arithmetic work in the YAML.

**Already implemented - do not reimplement.** The delete-operation confusion matrix and
`del_precision` / `del_recall` / `del_f1` / `del_chance` / `rank_p_at_k` live in `editflows/metrics.py`,
and `editflows/deletion_task.py::_report` logs them after every validation epoch and writes
`eval/deletion_metrics.csv`. There is deliberately no precision/recall for insert or substitute: the
coupling only ever inserts, so those have no positive class and are counted as event rates whose target
is zero.

**Runs.** Every run writes to `runs/<name>/` and nowhere else; `name` is mandatory and *is* the identity
of the run. Starting the same name again **resumes** from the newest of `checkpoints/last.ckpt` and
`periodic.ckpt` by mtime (`resume_path()`); `from_ckpt=none` forces a fresh start. Do not add
`trainer.max_time` to a config meant to be resumed - lightning restores the Timer cumulatively, so the
second start stops immediately. The cosine horizon comes from `estimated_stepping_batches`, i.e.
`max_epochs x limit_train_batches`.

**Epoch convention.** With no real dataset yet, one epoch is *defined* as 25 000 synthetic training
sequences, expressed as `limit_train_batches: ${eval:'25000 // (${data.batch_size} * ${gpus})'}` so it
survives a change of batch size *or* GPU count. The world size belongs in the divisor because
`limit_train_batches` is counted per process and the `DistributedSampler` shards the split per rank.

**Dependencies.** `torch` is intentionally absent from `pyproject.toml`, because the CPU and CUDA builds
come from different indexes; `requirements-cpu.txt` and `requirements-cuda.txt` pick one. See
`docs/DEPENDENCIES.md` before changing any pin.

**Gitignored.** `/runs/`, `/data`, `*.npz`. Anything that has to leave a cluster therefore needs
`git add -f` - see `scripts/push_run_artifacts.sh`.

**Fast gate.** `python -m pytest -q tests -m "not slow"` (~10 s, CPU). The slow marker is a subprocess
end-to-end train/evaluate test.

**Launchers.** The UiO CCSE cluster this is used on has **no batch scheduler** - no `sbatch`,
`squeue` or `sinfo`. Runs start with `scripts/train_detached.sh`, which checks the GPUs are free (there
is no admission control), sets `CUDA_VISIBLE_DEVICES`, and detaches under `setsid nohup`.
`scripts/train.sbatch` and `scripts/submit_chain.sh` are kept for SLURM sites and are unused there.
`neuroai` has no working NVIDIA driver; `hugefacet` (4x A100-40GB) is the target.

**Branches.** `on-GPU-cluster-branch` holds the CUDA-enabled cluster work. `docs/CLUSTER.md` is the
submission guide, with the concrete UiO CCSE runbook in its appendix; `notebooks/` holds the method
walkthrough and the cluster/analysis notebook.
