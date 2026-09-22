# Standalone branch B rollout demo

This directory contains an independent experiment: fit a small acceleration
correction from one observed orbit, then propagate five orbits without new
observations. Mamba and LSTM corrections are compared with the same models trained
on one-step loss, a two-body propagator, and persistence.

The existing demo is not modified or imported. All data, checkpoints, logs,
figures, and reports for this experiment stay here. See [PLAN.md](PLAN.md) for the
protocol and [STATUS.md](STATUS.md) for execution progress. Results are generated
in [REPORT.md](REPORT.md) after development has been frozen.

## Completed experiment

All 12 development cells, 14 window-0 walkthrough cells, and 56 later-window
evaluation cells completed without failures. Eight correctness tests and the
isolation audit pass. The complete result is in [REPORT.md](REPORT.md).

**Use the two-body baseline for this dataset.** Its whole-forecast RMSE is
37.98–46.21 km across windows 4–7. Median errors across seeds range from
133.63–233.59 km for the learned one-step controls and 812.95–1,538.92 km for
the rollout-trained corrections. Every learned variant loses to physics on
every later window; the rollout-trained variants also lose to their paired
one-step controls. These are full-trajectory errors, not endpoint-only scores.

The prescribed curriculum was completed, but validation selected the short
8-step stage for both rollout models. Improvements on a quarter-orbit validation
interval did not carry over to five-orbit forecasts after refitting. This is
a negative result for this particular correction/training/selection recipe;
it does not establish that all learned corrections are ineffective.

The next useful experiment would pre-register a frame-consistent J2 baseline
and validation spanning longer forecast horizons, with its additional observation
budget stated explicitly. Preserve this frozen experiment rather than retuning it
after seeing the later windows. Those extensions have not been implemented here.

## Setup and run

Use an environment with the dependencies in `requirements.txt` and an installed
`qutils.ml.mamba`. This implementation uses CPU float64 and one PyTorch thread per
process. It does not install packages, fetch remote data, or access credentials.
Source hashes, package versions, and the actual imported Mamba source hash are
recorded in each run. Commands below are run from this directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python demo.py prepare --source ../../data/real/28089/8620min-8
python -m pytest test_demo.py -q -o cache_dir=.cache/pytest
python demo.py baselines --window 0
python demo.py develop --window 0 --seeds 0 1 2
python demo.py freeze
python demo.py walkthrough --window 0 --seeds 0 1 2
python demo.py evaluate --windows 4 5 6 7 --seeds 0 1 2
python demo.py report
python demo.py audit
```

Paths are resolved relative to this directory even when `demo.py` is invoked from
another working directory. `prepare` copies only the specified NPZ and metadata,
checks their hashes, and never changes the original. Repeating it verifies the
local snapshot. Subsequent experiments use that copy.

Long matrix commands reuse completed cells with matching code, configuration,
data, and protocol hashes. Failed attempts remain saved; retrying creates a new
run. An interrupted cell can be rerun from scratch without overwriting it. Do not
start concurrent commands for the same cell. Independent seeds can run in separate
processes, for example `develop --seeds 0`, `develop --seeds 1`, and
`develop --seeds 2`. There is no background job manager.

Use `--variants mamba_rollout` (or another listed variant) to execute a subset.
Freezing requires all four learned variants and all three development seeds.
The frozen protocol cannot be overwritten, and code/config/data/dependency changes
after freezing are rejected. Preserve the first experiment before planning any
new protocol. Reports list exact run IDs and explicitly mark missing/failed cells.

## Scene 1: What is observed?

Open the generated `observation-budget.png`. Window 0 contains 8,620 states at
60-second spacing. The period is approximately 86,165 seconds. The forecast origin
is state 1,436: observations are states 0 through 1,436, inclusive (1,437 states).
The first prediction is state 1,437; five forecast periods follow the origin.

During development only, the first 75% of the observed orbit is used for fitting;
the remaining quarter is a contiguous free-running validation forecast. No future
five-orbit data select checkpoints. Each later evaluation window is fitted anew
on its own first orbit. The earlier development window is extra protocol-selection
data and is disclosed, rather than included silently in the per-window budget.

The reference is SGP4-densified space-track TLE data in GCRS, treated as approximately
J2000 by the source dataset. It is not independent ground-truth tracking. Gray
vertical lines in trajectory plots mark TLE hand-offs.

## Scene 2: What can physics already do?

Run `baselines` and inspect the two resulting `runs/<id>/run.json` files. Position
and velocity evolve together under two-body gravity using RK4, checked against
SciPy DOP853 for the complete five-orbit span. The permitted maximum numerical
disagreement is 0.01 km. This check must pass before development.

The persistence baseline deliberately holds the observed initial state fixed.
The satellite passes nearby after each orbital period, making integer-period
endpoint errors deceptively small. Compare those endpoints with whole-trajectory
RMSE and the continuous error plot. The primary metric is position RMSE across
every forecast sample, including hand-offs.

## Scene 3: What does learning change?

Both correction models receive the last eight six-component states, scaled by
fixed GEO length and speed scales. Mamba has width six and one layer; LSTM has
30 hidden units and one layer. A zero-initialized head emits a bounded acceleration
correction. Initially every learned model is exactly the two-body propagator.

The integrator evaluates physical gravity at each RK4 stage and holds the learned
correction constant within each 60-second interval. It integrates position and
velocity together. At later steps, the sliding context contains predicted states;
no true future state is fed back.

One-step controls train with horizon one. Rollout variants train with horizons
8, 32, then 128, each for 300 optimizer steps. Gradients pass through the entire
generated trajectory. Both use Adam, a batch of eight, and explicit position and
velocity loss scales. Sequences crossing training splits or TLE hand-offs are
excluded from training. Hand-offs are retained in evaluation.

Open `validation.png` to see every development checkpoint and the selected step
count. Selection minimizes median validation trajectory RMSE over three seeds;
the untrained physical baseline is also eligible. The chosen recipe is then
refitted from scratch on a full observed orbit. Equal optimizer-step budgets do
not make the training compute equal; actual costs are saved.

## Scene 4: Does it hold up without observations?

Run `walkthrough`, then `evaluate`, followed by `report`. Open the trajectory
figures linked from REPORT. They show continuous position error, orbital radius,
radial/along-track/cross-track error, energy drift, angular-momentum drift, and
velocity error. Reference energy/momentum drift is shown too: real perturbations
and TLE updates mean zero drift is not necessarily the correct target.

Window 0 is an illustrative development demo. Windows 4–7 provide later-window
evaluation on the same object. The table reports median/IQR over three seeds in
each window; a separate summary describes the distribution of window medians.
Paired ratios compare rollout training with physics and with the same architecture
trained one step at a time. Failed or missing runs suppress complete aggregates;
they are not silently discarded. Deterministic baselines appear once per window.

## Scene 5: What is the conclusion?

Prefer the simplest method supported by the later-window results. A selected
checkpoint of zero steps is a physics-only outcome. A correction that improves
one window but loses elsewhere has not established a general improvement. Report
the losses and seed spread alongside wins. This experiment cannot establish
performance on different satellites or absolute sub-km tracking accuracy.

Run `audit` last. It verifies unchanged tracked files and Git status outside this
directory, unchanged original/copy data hashes, and absence of escaping symlinks.
Generated artifact paths are checked at write time. The audit is a targeted
integrity check, not an operating-system sandbox of third-party libraries.

## Files to inspect

- `config.json`: experimental settings; freeze before evaluating future windows.
- `frozen.json`: selected steps, development run IDs, and immutable protocol hash.
- `runs/<id>/run.json`: provenance, costs, metrics, status, and any failure traceback.
- `runs/<id>/history.json`: development validation history, updated during training.
- `runs/<id>/progress.json`: most recent training progress.
- `runs/<id>/checkpoint.pt`: model weights, never loaded as pretraining for another window.
- `runs/<id>/forecast.npz`: predictions, references, elapsed times, hand-offs, and drifts.
- `figures/<protocol>/`: regenerated plots; `REPORT.md`: comparisons and run IDs.

The local tests cover integrator accuracy, zero correction, active temporal
gradients, multi-step weight updates, forecast-origin indexing, future-data
independence, hand-off-safe training sequences, snapshot hashes, output containment,
and explicit failure metrics. Training and evaluation never invoke the broader
repository's scripts, model-training helpers, or result writers.
