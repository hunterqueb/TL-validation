# Standalone rollout demo status

Implementation started 2026-09-21. All work is confined to this directory.

- Plan read; initial repository status has only this directory's PLAN.md untracked.
- Complete: local physics, correction models, CLI, snapshot and eight correctness checks.
- Complete: physics baseline; RK4 versus DOP853 maximum discrepancy 0.000015664 km.
- Complete: fixed three-seed development matrix and immutable protocol freeze.
- Complete: full first-orbit refits, window-0 walkthrough, and windows 4–7 evaluation.
- Complete: generated report, seven figures, visual inspection, and isolation audit.
- No required implementation or experiment work remains for the frozen protocol.

No experiment outcome is assumed. Failures and protocol deviations will be recorded here.

## Commands and observations

Commands executed from repository root (all outputs remain in this directory):

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/branch_b_rollout/demo.py prepare --source ../../data/real/28089/8620min-8
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=experiments/branch_b_rollout/.cache/matplotlib python -m pytest experiments/branch_b_rollout/test_demo.py -q -o cache_dir=experiments/branch_b_rollout/.cache/pytest
PYTHONDONTWRITEBYTECODE=1 python experiments/branch_b_rollout/demo.py baselines --window 0
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py develop --window 0 --seeds 0
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py develop --window 0 --seeds 1
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py develop --window 0 --seeds 2
```

Eight tests pass, including frozen-protocol selection, five-orbit numerical accuracy, and nonzero temporal
gradients after the zero-initialized output head starts learning. The reference
two-body forecast has whole-trajectory RMSE 44.72751 km, error 15.03109 km at 1T,
and 81.32957 km at 5T. Persistence RMSE is 59,610.20016 km.

Two baseline sets exist because implementation provenance changed before the
development matrix; only the matching current code/config baseline is eligible
for the development gate. Neither set overwrote previous artifacts.

Early development observation: increasing rollout horizon can worsen the reserved
quarter-orbit forecast. All prescribed stages were evaluated; selection used the
declared three-seed rule, without changing the schedule to seek a win.

## Frozen selection

All 12 development cells completed without failures. The exact median-over-seeds
selection rule picked optimizer steps:

| Variant | Selected steps | Horizon at selection |
|---|---:|---:|
| mamba_one_step | 850 | 1 |
| mamba_rollout | 300 | 8 |
| lstm_one_step | 750 | 1 |
| lstm_rollout | 100 | 8 |

Thus both selected rollout models use the short-horizon stage. The 32- and
128-step stages were executed and retained, but did not win validation. This
is a result of the declared selection rule, not a shortened development run.

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/branch_b_rollout/demo.py freeze
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py walkthrough --window 0 --seeds 0 1 2
```

## Completed evaluation and findings

Protocol: `4a25bddbb77a0ca03c7ce84498d1c07f6fccef4ec754ef75792130ba5454a437`.
Completed 12 development + 14 walkthrough + 56 evaluation cells, with no failed
or diverged cells. Two initial baseline pairs are also retained (before/after
implementation provenance changed). No hyperparameters were changed after seeing
the walkthrough or later-window results.

These four evaluation commands were run independently in separate processes:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py evaluate --windows 4 --seeds 0 1 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py evaluate --windows 5 --seeds 0 1 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py evaluate --windows 6 --seeds 0 1 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py evaluate --windows 7 --seeds 0 1 2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python experiments/branch_b_rollout/demo.py report
PYTHONDONTWRITEBYTECODE=1 python experiments/branch_b_rollout/demo.py audit
```

The two-body baseline wins on all four later windows: whole-forecast RMSE
37.98–46.21 km. Per-window three-seed median RMSE is 142.20–233.34 km for
one-step Mamba, 133.63–233.59 km for one-step LSTM, 812.95–1,538.92 km for
rollout Mamba, and 818.58–1,526.84 km for rollout LSTM. The physics-guided
implementation works, but the learned correction does not improve propagation
under this protocol. The recommended current propagator is the physical baseline.

Short-horizon validation selected checkpoints that perform poorly over five
orbits after refitting. A new, separately registered experiment could evaluate
J2 and longer validation horizons with a disclosed additional data budget.
That is a future scientific task, not an unfinished implementation step here.

Validation and walkthrough figures were visually checked for readable labels,
unclipped layout, continuous error curves, hand-offs, and reference drift. The
audit confirms unchanged outside tracked files/status, original dataset hashes,
snapshot hashes, and no escaping symlinks. No broader-repository documentation,
scripts, configuration, tests, or output files were modified.
