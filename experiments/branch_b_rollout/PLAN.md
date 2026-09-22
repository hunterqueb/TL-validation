# Standalone branch B rollout demo — agent execution plan

Status: implemented and evaluated; see local STATUS.md and REPORT.md for the negative learning result.
Created: 2026-09-21.

## 1. Assignment and isolation boundary

Implement and walk through a standalone experiment answering: **Can a physics-based
propagator with a learned acceleration correction, trained through its own predictions,
improve closed-loop prediction when trained from scratch on one observed orbit?**

This is a new branch B experiment: real-data training from scratch, one orbit of
observations per target window, then five orbits without observations. It is not a
reproduction of the existing architecture, epoch budget, or published protocol.
There is no synthetic pretraining, transfer learning, branch A/C dependency, or
two-transition DDDAS experiment in this assignment.

All new work belongs under `experiments/branch_b_rollout/` (called `DEMO` below).

- Do not edit, link to this experiment from, or register it in any file outside DEMO.
  This includes the root README, docs, task lists, configurations, requirements,
  tests, scripts, CI, results, figures, and ignore rules.
- Do not import the repository's `tlv` or `scripts` modules. Build the small amount
  of necessary loading, training, evaluation, and plotting code locally.
- The sole initial external data input is the existing real-data NPZ and metadata
  described below. Copy them into DEMO once, record their hashes, and thereafter
  run using the local snapshot. Never modify the source dataset.
- Installed scientific libraries and the installed `qutils` Mamba implementation
  may be used as dependencies. Do not edit/reinstall a sibling repository or patch
  site-packages. Record versions and the hash of the actual imported Mamba source.
- Put caches, logs, checkpoints, figures, local environment files, and test output
  under DEMO. Add a DEMO-local `.gitignore`; do not change the root ignore file.
- Resolve paths from this file's directory, not from the caller's working directory.
  Reject output destinations resolving outside DEMO, including symlink escapes.
- Keep a local `STATUS.md` with completed steps, exact commands, findings, and next
  action. Do not update the existing demo's task ledger or approval milestones.
- Do not commit, publish, or launch unrelated experiments as part of this assignment.

The user explicitly requested this independent experiment. The original demo's
fixed hyperparameters and global bookkeeping are not instructions to modify that
demo or block this one. Record decisions and deviations only here.

## 2. Evidence motivating the experiment

A diagnostic on Yamal 202, window 0, used a forecast origin at sample 1436 and
60-second steps. These are reference observations, not newly generated results
of this experiment:

| Method | Position error at 1T | Position error at 5T |
|---|---:|---:|
| Existing one-step Mamba, seed 0 | 6,228.90 km | 71,303.47 km |
| Existing one-step Mamba, three-seed median | 36,641.72 km | 663,409.03 km |
| Two-body DOP853 integration from the same observed state | 15.03 km | 81.33 km |

The seed-0 model's teacher-forced position RMSE was only 5.94 km. Existing inputs
had length one, without carried recurrent state. One checked Mamba `A_log` gradient
was zero. The new experiment must exercise actual temporal context if it uses
Mamba/LSTM, and directly optimize trajectories generated from predicted states.

Holding the initial position fixed scored 173.28 km at 1T but 59,610.20 km RMSE
across the forecast. Integer-period endpoints can hide total failure between
endpoints. Whole-trajectory metrics are mandatory.

These diagnostics used one window and do not establish generalization. The
reference is SGP4-derived TLE ephemeris, not independent tracking truth. TLE hand-off
disagreement is a consistency statistic, not a calibrated accuracy floor.

## 3. Minimal local layout

Create files when their implementation step needs them; avoid a package framework.

```text
experiments/branch_b_rollout/
  PLAN.md                 this assignment
  STATUS.md               local progress and decisions
  README.md               local setup and narrated demo commands
  .gitignore              local data/output/cache/environment exclusions
  requirements.txt        only this demo's dependencies
  config.json             frozen scientific and training settings
  demo.py                 CLI, snapshot loading, training, evaluation, reporting
  dynamics.py             physical integration and correction models
  test_demo.py            small, meaningful correctness checks
  data/                   copied NPZ, metadata, snapshot manifest
  runs/<run_id>/          config, provenance, checkpoints, predictions, metrics
  figures/<run_id>/       regenerated figures
  REPORT.md               local findings, including negative results
```

Use NumPy, SciPy, PyTorch, matplotlib, and the installed qutils Mamba. JSON and CSV
need only the standard library. Reuse local functions across the CLI subcommands.
Prefer CPU float64 initially; capture device, thread count, package versions,
source hashes, random seeds, training time, and optimizer-step counts.

## 4. Data contract and observation budget

Initial source, relative to repository root:

```text
data/real/28089/8620min-8/statesArrayReal.npz
data/real/28089/8620min-8/meta.json
```

Expected arrays: `statesArrayReal (8, 8620, 6)`, `t_sec (8, 8620)`,
`epoch_jd (8,)`, `tle_index (8, 8620)`, `handoff_jump_km (8, 8620)`.
States are `[x,y,z,vx,vy,vz]` in km and km/s. Metadata reports the frame as
`TEME -> GCRS (astropy); GCRS~J2000 bias ignored`; preserve that qualification.
Read the period from metadata, expected approximately 86165.2273284 seconds.
Record both the NPZ/metadata SHA-256 hashes and the source TLE hash.

Validate shapes, finiteness, units, positive radii, uniform 60-second sampling,
monotonic absolute epochs, non-overlapping windows, and horizon coverage. Do not
fetch data, read credentials, or rebuild the broader dataset. If input is missing
or invalid, record the exact missing path/problem in local STATUS and stop the
dependent step.

For each window define `origin = round(period_sec / dt)` (expected 1436).
Observed states are indices `0..origin`, inclusive: 1437 states / 1436 transitions.
First scored prediction is `origin + 1`. Forecast duration is measured from
`t_sec[origin]`, never from window start. Use each window's own time array.

- **Development:** window 0 only. Within its observed first orbit, let
  `cut = floor(0.75 * origin)`. Fit using states `0..cut`; validate by free-running
  from `cut` through `origin`. All normalization and fitted quantities during
  development come from the fitting prefix. Never train through the split.
- **Freeze:** select the training checkpoint/step count using that contiguous
  validation rollout. Freeze configuration before inspecting window 0's remaining
  five orbits or the final evaluation forecasts. Do not tune to the diagnostic
  numbers in section 2.
- **Walkthrough:** refit each selected recipe from scratch using window 0's entire
  observed first orbit, then forecast its remaining five orbits.
- **Final evaluation:** windows 4, 5, 6, and 7, seeds 0, 1, 2. Initialize and fit
  a fresh model on each window's own first orbit using the frozen recipe. No
  checkpoint selection or tuning on that window's future. Window 3 has previously
  been examined and is excluded. Leave windows 1 and 2 unused in this first demo.

This measures repeated low-data adaptation on later windows of one object. It
does not test transfer across objects. Development data are additional protocol
selection data and must be disclosed separately from each run's one-orbit budget.

For all training starts `s`, require `s-lookback+1 >= 0` and `s+K <= fit_end`.
Reject training samples whose context or target horizon crosses a change in
`tle_index`; SGP4 segment jumps must not become learned force impulses. Record
eligible start counts at every horizon and fail explicitly if a horizon has none.
Evaluate the complete forecast, including hand-offs; never remove difficult test
points. Mark hand-offs on plots and optionally provide a separately labeled
within-segment diagnostic.

## 5. Models and physical integration

Implement the following comparison set:

| Label | Training | Role |
|---|---|---|
| persistence | none | exposes misleading integer-period endpoints |
| two_body | none | physical baseline and default useful propagator |
| mamba_one_step | one-step loss | control for the same physics/correction architecture |
| mamba_rollout | rollout curriculum | proposed Mamba improvement |
| lstm_one_step | one-step loss | paired LSTM control |
| lstm_rollout | rollout curriculum | proposed LSTM improvement |

Use `dr/dt = v` and `dv/dt = -mu*r/||r||^3 + a_correction`, with
`mu = 398600.4418 km^3/s^2`. Integrate all six components together using
differentiable fixed-step RK4 at 60 seconds. Evaluate gravity at every RK stage.
The network emits one acceleration correction per observation step; hold that
correction constant across the RK stages of that step. State this approximation.
No independent network output for next position and next velocity.

Use a separate SciPy DOP853 two-body reference (`rtol=1e-11`, `atol=1e-12`) to check
RK4 over the full five-orbit duration. Require maximum position disagreement
below 0.01 km. If it fails, use two 30-second RK4 substeps consistently for the
baseline and learned models and rerun the check. Do not attribute integrator
error to learning.

The correction is deliberately small:

```text
R = 42164 km                    fixed characteristic radius, not fitted to test data
V = sqrt(mu/R)
A = mu/R**2
features = concatenate(r/R, v/V)
a_correction = (1e-3 * A) * tanh(head(last_context_feature))
```

The bound is per component, in km/s²; report it as an experimental prior, not an
identified property of the satellite. Log bound saturation. Zero-initialize the
final linear head so every model starts as the physical baseline.

Use a sliding context of **8 states**. Mamba: installed `Mamba/MambaConfig`, width 6,
one layer, `d_conv=4`, then a local `Linear(6,3)` head. LSTM: native PyTorch,
input width 6, hidden width 30, one layer, no dropout, then `Linear(30,3)`.
Take only the final context output. Recompute each context without a persistent
hidden cache; this makes training and rollout behavior identical. Subsequent
contexts contain predicted states, never injected future observations.

Document the bounded memory explicitly. After the head has begun updating, verify
that changing earlier context changes the correction and that a temporal parameter
can receive a nonzero gradient. A zero gradient on the first backward pass is
expected with a zero-initialized output head; do not misdiagnose that as a bug.

J2 is a follow-up physical baseline if needed, not a prerequisite. Adding it requires
a frame-consistent implementation and a matching potential for energy diagnostics;
do not casually assume the GCRS z axis is the instantaneous Earth spin axis.

## 6. Training recipe and controlled extensions

Initial development recipe: Adam, learning rate `1e-3`, batch size 8, float64,
gradient norm clipping at 1.0, fixed seeds, no weight decay, no input noise.
Use randomized valid start indices from the fitting prefix.

- One-step control: 900 optimizer steps with `K=1`.
- Rollout model: 300 steps each at `K=8`, `32`, and `128` (900 total).
- At each training start, seed from the observed context once, generate all K
  predictions autoregressively, and backpropagate through the complete K-step graph.
  Do not detach between predicted steps or substitute true target states.
- Loss is the mean over batch, time, and components of squared position error
  divided by `(1 km)^2`, plus squared velocity error divided by `(0.001 km/s)^2`.
  These fixed weights express a chosen error tradeoff; nondimensionalizing model
  inputs alone does not balance the loss.
- Every 50 optimizer steps, evaluate the entire reserved development interval
  as one free rollout. Choose the minimum whole-interval position RMSE, including
  the untrained zero-head checkpoint as a candidate. Record position and velocity
  errors, finiteness, and chosen training step.
- Select a step count per architecture/objective using median validation RMSE
  across seeds 0, 1, 2; ties choose the earlier step. Refit final models from scratch
  to that fixed step count. Never transfer a development checkpoint's weights.

Equal optimizer steps do not imply equal compute: record unrolled training steps,
force/network evaluations, wall time, and observed states. Do not claim a
compute-matched or original five-epoch comparison. Zero selected steps means the
correction did not help on validation; label it as a physics-only outcome.

If the frozen first experiment does not improve on physics, report it. Do not
silently expand the budget until a win appears. A subsequent, separately named
development experiment may test modest, unit-specified input perturbations or
additional physically simulated neighboring trajectories. Perturbation training
must define whether it teaches recovery to the observed path or propagation from
a changed initial condition; those require different targets. Synthetic training
changes the real-only branch B claim and belongs to an explicitly labeled extension.
Do not add scheduled sampling, a larger model, or extra penalties without evidence.

## 7. Metrics, saved artifacts, and claims

Compute all rollout metrics on the same forecast samples and initial conditions:

- Primary: whole-forecast position RMSE,
  `sqrt(mean(sum((pred_r - true_r)**2, axis=-1)))`, in km.
- Secondary: whole-forecast velocity RMSE, maximum position error, position error
  at 15 minutes, 1 hour, 6 hours, 0.25T, 0.5T, 1T, 2T, and 5T.
- Endpoint indices are the first samples at/after the requested horizon from the
  forecast origin. Save the actual elapsed time; unavailable horizons are missing,
  never replaced with a closer sample.
- Plot position error continuously versus time, radial/along-track/cross-track
  error using reference-state axes, and predicted/reference orbital radius.
- Save two-body specific energy and angular-momentum drift of the **rollout**,
  relative to the initial observed state. Also plot the reference trajectory's
  drift: real perturbations and segment changes mean zero drift is not the truth.
- Teacher-forced errors may be supplementary, always labeled and paired with
  rollout errors. They are not the headline and must not select checkpoints.
- Include hand-off count, median, and p95 with each window; call them consistency
  statistics. Every figure says `reference = SGP4 from space-track TLEs`.

Each immutable run directory contains the complete resolved config, data and code
hashes, seed/window/origin, environment, training/validation history, chosen step,
checkpoint, predicted states and times, and machine-readable metrics. Do not
append to or read comparison rows from the broader repository's results files.
Do not overwrite runs. Report nonfinite/diverged runs explicitly and retain the
last valid step; never drop failures from aggregates or stop a merely large but
finite trajectory to improve its score.

Aggregate in two levels: median/IQR over seeds **within each window**, then show
the distribution across windows. Report paired RMSE ratios versus two-body and
versus the same architecture's one-step control. Deterministic baselines appear
once per window, not as three fictitious independent seeds. Do not pool all seed
and window rows into one unexplained median.

Scientific outcomes and implementation completion are separate. A successful
learning claim requires lower primary error on later windows, a clear paired
comparison, and disclosure of windows/seeds where it loses. If the correction
does not beat two-body, recommend the physical baseline and explain what failed.
Never claim sub-km real accuracy from this reference or extrapolate to other objects.

## 8. Agent implementation sequence and checks

1. **Containment and snapshot.** Create the local layout/ignore rules and STATUS.
   Record initial repository status. Implement explicit snapshot import and hashes;
   validate the data contract and disjoint absolute window intervals. Only snapshot
   import and the final read-only hash audit may read the original dataset.
2. **Physics and scoring first.** Implement persistence, RK4, DOP853 reference,
   forecast indexing, and trajectory metrics. Reproduce the rough window-0 physical
   baseline from section 2; investigate timing, units, or frame mismatches before
   training. Do not hardcode expected diagnostic values into the algorithm.
3. **One correction model end to end.** Implement Mamba's local acceleration head,
   8-state context, zero initialization, and the one-step/rollout objectives.
   Add LSTM through the same train/evaluate functions, not a duplicate pipeline.
4. **Meaningful tests.** One small local test file must check: zero correction
   equals physical integration; RK4 agrees with DOP853; seeding and horizons use
   the last observed state; forecast suffix edits cannot affect training inputs or
   prediction; sampled sequences respect split/hand-off boundaries; a K>1 loss
   has nonzero finite parameter gradients and updates weights; temporal context
   is active after initial head updates; snapshot inputs remain unchanged; output
   paths cannot escape DEMO. Use a toy physical trajectory where useful.
5. **Development and freeze.** Run the fixed development matrix, save validation
   curves, select the step counts, and save a hash of the frozen protocol. If there
   is a concrete numerical/implementation failure, fix and rerun development before
   any final evaluation. Log all deviations and failed trials locally.
6. **Walkthrough.** Run window 0 from scratch for all variants and three seeds;
   produce the local figures and narration below. Treat it as a development demo,
   not independent evidence, since its prior results are already known.
7. **Later-window evaluation.** Run the frozen matrix on windows 4–7, seeds 0–2.
   Continue collecting independent cells after a run failure, retain failure rows,
   and make the command exit nonzero if any cell fails.
8. **Report and isolation audit.** Write local REPORT and README with exact commands,
   resolved protocol, results, limitations, and recommended next action. Compare
   repository status and source hashes with the initial snapshot; prove this work
   changed nothing outside DEMO. Preserve unrelated pre-existing user changes.

Do not run training in this planning task; the implementing agent executes these
steps. No success threshold should force it to manufacture an improvement.

## 9. CLI contract and narrated demo

Implement these commands; they are a target interface, not commands that exist yet.
All output defaults stay under DEMO. Run from the standalone directory:

```bash
cd experiments/branch_b_rollout
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
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

`freeze` must record the development run IDs, selection rule/results, and full
configuration hash. `walkthrough` and `evaluate` reject missing or mismatched
frozen settings. `report` identifies exact input runs; it must not silently mix
protocol revisions. `audit` checks captured source hashes and file containment.

The local README walks the audience through these five scenes:

1. **What is observed?** Show one observed orbit, the marked forecast origin, and
   five future orbits. State the exact number of observed states and time span.
2. **What can physics already do?** Show two-body and persistence on the continuous
   error curve. Explain why returning near the start after 1T flatters persistence.
3. **What does learning change?** Show the bounded acceleration correction and the
   two objectives side by side: one true-input step versus an unrolled predicted
   trajectory. Explain that position and velocity are integrated together.
4. **Does it hold up without observations?** Reveal full rollout error, radius,
   and drift plots for the window-0 demo, then paired results on later windows.
   Show seed spread and losses as well as improvements.
5. **What is the conclusion?** Choose the simplest method supported by results.
   Explicitly distinguish a successful implementation from a successful learned
   correction, and identify the next experiment only if the evidence warrants it.

## 10. Completion checklist

- [x] All implementation, documentation, caches, and generated artifacts are local.
- [x] Data snapshot and provenance are verified; no new remote data dependency.
- [x] Physical integration, metrics, gradients, splits, and containment checks pass.
- [x] Frozen development protocol and training budgets are reproducible.
- [x] One-step controls and rollout-trained variants use the same correction models.
- [x] Window-0 walkthrough and windows 4–7 evaluation are complete or failures named.
- [x] Continuous errors and paired multi-seed/multi-window comparisons are reported.
- [x] Negative results and reference-data limitations remain visible.
- [x] Local README contains working commands and the five-scene walkthrough.
- [x] Nothing outside this directory was edited or given a reference to this demo.
