# Error study: data, tools and conventions

The scripts in this folder analyse *how* the deletion edit flow fails. Each one takes `--run runs/<name>` and
writes tables, a `summary.json` and figures to `runs/<name>/errors/results/<topic>/`, so the same analysis can be
repeated for another checkpoint. Run them from the repository root with the project's Python.

## The task

A target is a sequence over 20 letters from an order-2 Markov chain: after every pair of letters only 5 of the 20
letters may follow (`P[a, b, c] > 0`). A source is a target with uniformly random letters inserted at uniformly
random positions; the share of the source that is original is drawn from U(0.5, 0.99). The model has to delete
exactly the inserted letters. It is sampled with 100 Euler steps; in every step each letter still present is
deleted independently with probability `h * lambda_del` (`h = 0.01`). The schedule has the hazard
`3 * exp(1.5 t)`; `q = lambda_del / hazard` is the model's own probability that a letter is a leftover insertion.

Verdict of a source letter: `.` original kept, `x` original deleted (wrong deletion), `_` inserted kept (missed),
`+` inserted deleted (correct). Codes 0, 1, 2, 3 in that order (`editflows.errors.verdict_codes`).

## Files

| file | content |
|---|---|
| `runs/error-study/sources.npz` | 6 000 held-out sources (first 3 000 from validation targets, then 3 000 test). Load with `editflows.errors.Sources.load`: `letters` (int8, flat, no BOS), `inserted` (bool), `offsets`, `split`, `target`; `sources.seq(i)`, `sources.mask(i)`, `sources.span(i)`, `sources.rho` (share inserted) |
| `runs/error-study/posterior.npz` | exact probability P(inserted) of every letter at t = 0. `q_gap`: with the coupling's own prior (K letters thrown into random gaps, so a further insertion is *less* likely after an insertion); this is the reference to use. `q_true`: the simpler hidden-Markov model of `editflows.exact` (every letter an insertion independently, prior = true share of insertions), `q_est` (the same with the share estimated from the source), `rho_hat` (that estimate) |
| `runs/error-study/g.npz`, `g-rep<k>.npz` | **the perfect sampler**: one exact joint posterior sample per source under the coupling's own prior (`scripts/errors/gap_reference.py`); same format as `ffbs.npz`; replicates `k = 0 … 31` of the first 500 sources |
| `runs/<run>/errors/m100.npz` | the trained model, standard sampler, all sources: `del_step` (int16, step of the deletion, -1 = kept), `q0` (model's q at t = 0), `exposure` (summed deletion probability while present), `first` (number of sources covered), `steps`, `seed`, `epoch` |
| `runs/<run>/errors/qlog.npz` | as above for the first 1 000 sources plus `qlog` (letters × steps, float16): the model's q at every step, NaN once the letter is gone |
| `runs/<run>/errors/rep<k>.npz` | replicates of the first 500 sources with sampler seed k (`k = 0 … 7` exist for `lap-01` and `lap-long`) |
| `runs/error-study/ffbs.npz` | the same under the simpler independent-marks prior of `editflows.exact` (the baseline for the `i2-*` samplers, which use that prior); no dynamics (`del_step == 0` means "sampled as an insertion") |
| `runs/error-study/i2-<n>.npz` | Euler sampler with exact rates and n steps; the prior is the share of leftovers a model can expect at time t given the true noise level |
| `runs/error-study/i3-100.npz` | the same with the noise level estimated from the source |
| `runs/error-study/ffbs-rep<k>.npz`, `i2-rep<k>.npz` | replicates of the first 500 sources |
| `runs/error-study/i2-leak<c>.npz` | exact rates with a deliberate defect: no letter's q is taken below `c` (`--leak`), which deletes letters without reason at a known rate |
| `runs/<run>/errors/m50/m200/m400.npz`, `v-<name>.npz` | the model with other step counts and sampler variants (see `scripts/errors/solutions.py`) |
| `runs/<dir>/results/clusters/<tag>.csv`, `<tag>-sources.csv` | cluster table and per-source counts of a corpus (`scripts/errors/cluster_tables.py`) |

All per-letter arrays are flat and aligned with `sources.letters[: sources.offsets[first]]`. Every file stores how
many sources it covers in `first`; a script must read it and use the common sources of the files it compares
(the replicates and the step sweeps cover fewer sources than the main corpora, and a run may have fewer
replicates than 32).

## Tools

- `editflows/errors.py`: `Sources`, `verdict_codes`, `verdict_string`, `cluster_ids` and `clusters` (errors grouped
  in output coordinates: at most `gap = 2` kept originals between two errors; types `Z`, `V`, `R`, `U`),
  `break_size`, `minimal_repair`, `lcs_length`, `canonical_counts`, `model_euler` (the sampler for the delete
  operation with random numbers keyed by (step, source letter); arguments `alive`, `keep`, `start_step`, `end_step`,
  `exclusion`, `sharpen`, `floor`, `log_q`).
- `editflows/exact.py` (torch, float64): `pad`, `posterior`, `posterior_sample`, `exact_euler`, `estimate_rho`,
  `leftover_share`.
- The grammar: `from make_synthetic_targets import transition_table` (in `scripts/`),
  `P = transition_table(np.random.default_rng(0), 20, 5, 1.0)`, `ELIGIBLE = P > 0`.
- Loading a model: see `cmd_model` in `scripts/build_error_corpus.py` (`load_model(checkpoint, config, ema=True)`,
  `kappa = instantiate(config.task.path).kappa`, BOS token = `config.data.vocab_size - 1`).

## Conventions

- Uncertainty: bootstrap over **sequences** (the letters of one sequence are dependent), 1 000 resamples.
- Sources 0 – 2 999 are for exploring and tuning, 3 000 – 5 999 for reported numbers of anything that was tuned.
- No new dependencies: numpy, pandas, torch, matplotlib only (no scipy).
- The edit flow (`editflows/editflows.py`, `edit_head.py`, `coupling.py`) is not modified; sampler variants live in
  analysis code.
- Figures: one message per figure, stated in the title; colour-blind safe colours
  (`#0072B2 #D55E00 #009E73 #CC79A7 #E69F00 #56B4E9 #000000 #7F7F7F`, always in this order); reference lines
  labelled on the plot.
