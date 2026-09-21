# tl-validation — Execution Plan

Companion to `HANDOFF.md` (the spec) and `TASKS.md` (the running task list).
Read order for any agent: `HANDOFF.md` §0–§1 → this file → `TASKS.md`.

- `HANDOFF.md` is the source of truth for *what* and *why*. Do not restate it here.
- `PLAN.md` says *in what order*, *what can run in parallel*, and *how agents behave*.
- `TASKS.md` says *what is done, in progress, blocked*. Agents update it; humans read it.

> **Review annotation — 2026-09-20:** Read [HANDOFF §10](HANDOFF.md#10-plan-review-2026-09-20)
> alongside this plan. REV-01–REV-10 are open findings; the ordering below is the original
> plan, not an endorsement that the current protocol is ready for the full experiment.

---

## 1. Goal, in one paragraph

Validate the ASC 2025 transfer-learning claim and the DDDAS 2024 low-data claim on real
LEO ephemerides (space-track TLEs, SGP4-densified, J2000, 60 s). Three branches share one
target object and window: (a) zero-shot synthetic→real, (b) train-from-scratch on real
low data, (c) synthetic pretrain + 2-epoch real fine-tune. The headline result is (c) beating
both (a) and (b) at equal target budget, for Mamba, median over 3 seeds, with the SGP4
hand-off noise floor printed next to every number.

> **Review — REV-01, REV-07, REV-10:** The active target is Yamal/GEO, and hand-off
> disagreement is not a demonstrated accuracy floor. Proposed question: "Does synthetic
> pretraining improve held-out performance at a fixed target-data and adaptation budget?"
> Winning is an experimental outcome, not a completion requirement.

---

## 2. Phases and dependency graph

```
P0 env ──┬── P1 real data ──── P2 branch (b) ──┐
         │                                       ├── P6 long-term ── P7 summary
         ├── R1 qutils fix    (needed by P2+)   │
         ├── R2 GMAT config ── P3 matched synth ─┴── P4 branch (a) ── P5 branch (c)
         └── R3 pendulumRNN fix ── P8 corrected CR3BP tables ── P7 summary
```

| Phase | Entry criterion | Exit criterion (milestone in HANDOFF §6.2) |
|---|---|---|
| P0 Environment | repo cloned | `pytest tests/test_datasets.py` green after R1 |
| R1 qutils constants fix (F4) | P0 | round-trip test passes; committed to qutils with note |
| R2 GMAT config change (F5, F7) | P0 | a 10-run fixed-element LEO set exists with metadata keys |
| R3 pendulumRNN F1 fix | P0 | corrected script runs pair 1; `newModel` params change |
| P1 Real data | P0, creds | M0 |
| P2 Branch (b) | P1, R1 | M1, M1b |
| P3 Matched synthetic source | P1, R2 | synthetic-vs-real 1-period sanity plot, gap recorded |
| P4 Branch (a) zero-shot | P3, R1 | M2 |
| P5 Branch (c) transfer | P4, P2 (equal-budget rows exist) | M3 |
| P6 Long-term (7-day) | P2, P4, P5 | M4 rows |
| P7 Summary | P6, P8 | M4 figures + `results/summary.md` |
| P8 Corrected CR3BP tables | R3 | M5 |

Hard rule: P5 does not start until M1 and M2 are green. M3's comparison is meaningless
without the (a) and (b) rows it is compared against.

> **Review — REV-02–REV-07, REV-09:** Retain the requirement for comparison rows, but
> separate execution validity from M2's performance threshold. Proposed revised sequence,
> pending protocol reconciliation:
>
> 1. Settle the GEO claim, observation budget, primary metrics, and held-out windows.
> 2. Correct and test forecast origins, split boundaries, and horizon coverage.
> 3. Validate one reference window and establish physical baselines.
> 4. Produce one genuinely matched GMAT trajectory with complete metadata.
> 5. Run one paired a/b/c pilot with reproducible artifacts; then expand across windows/seeds.
> 6. Summarize the outcomes, including negative results. Keep corrected CR3BP reproduction
>    as a parallel deliverable; reconsider making it block the GEO report (REV-10).

---

## 3. Parallel lanes

Four lanes can run concurrently from the start. Each lane is owned by one agent at a time.

| Lane | Tasks (see `TASKS.md`) | Touches | Must not touch |
|---|---|---|---|
| A — data | P0-*, P1-* | `tlv/spacetrack.py`, `tlv/tle_ephem.py`, `tlv/datasets.py`, `tests/test_tle_ephem.py`, `data/` | `tlv/models.py`, `tlv/train.py` |
| B — qutils | R1-* | `../qutils/qutils/orbital.py`, `tests/test_datasets.py` | anything in `tlv/` |
| C — GMAT | R2-* | `~/git/GMAT-Thrust-Data/python/generateSpacecraftThrustOpt.py`, `configs/gmat_*.yaml` | `tlv/` |
| D — pendulumRNN | R3-*, P8-* | `pendulumRNN/scripts/conferences/ASC2025/mambaTLCR3BP.py`, `results/asc2025_corrected.md` | `tlv/` |

After lanes converge (P1 + R1 done), the branch lane is sequential: P2 → P3 → P4 → P5 → P6 → P7.
Within P2/P4/P5 the `{model} × {seed}` runs are embarrassingly parallel and may be split across
agents *after* a single seed-0 run has produced a correct plot set.

---

## 4. Agent operating rules

1. **Claim before work.** Set the task's `status` to `in_progress` and `owner` to your agent
   id in `TASKS.md` before editing any file. One task per agent at a time.
2. **Small commits, task ID in message.** `git commit -m "P1-03: segment_ephemeris + handoff jumps"`.
   Attribution lines per the session reminder.
3. **Definition of done is the DoD column.** Not "it runs". If the DoD has a command, run it
   and paste the last 5 lines of its output into the task's log.
4. **Never delete a task.** Mark `dropped` with a one-line reason.
5. **Follow-ups get new IDs** in the same phase (`P1-07`, `P1-08`, …) appended at the end of
   that phase's table. Reference the parent task.
6. **Blocked means named.** `blocked` requires `blocked_by` = a task ID or a §9.5 open item
   from `HANDOFF.md` (`H-1` … `H-4`). Do not guess around an open item; stop and log.
7. **Spec changes go to `HANDOFF.md` first.** If you must deviate (a function signature, a
   threshold), edit `HANDOFF.md`, note the change in the Decision Log below, then implement.
8. **Guardrails in `HANDOFF.md` §9.3 are non-negotiable.** In particular: no teacher-forced
   number without its rollout number; no error without the noise floor; ≥ 3 seeds for any
   headline; equal-budget rule for TL comparisons.
9. **Six modules in `tlv/`.** A seventh requires a Decision Log entry with the reason.
10. **Results are append-only CSV rows** (`tlv/metrics.py::append_result`). Never edit a row.
    Re-runs get a new row with a new `run_id`.
11. **Figures are regenerated, never hand-edited.** If a figure is wrong, fix the script.
12. **Do not tune hyper-parameters** in P2–P6. The values in `HANDOFF.md` §4 are fixed for
    comparability with the papers. Tuning is a separate, later phase if Hunter asks for it.

---

## 5. Cadence

> **Review — REV-09, REV-10:** A green smoke test is not a scientific milestone. Record
> code completion and dataset/experiment validation separately. Review findings are tracked
> in TASKS F-05–F-14; annotating them does not close them or authorize unrelated code changes.

- After each task: update `TASKS.md`, commit.
- After each phase: append a 3–6 line phase note to §7 of this file (what was produced, what
  surprised you, what the next agent must know). Commit.
- A human review is required at M0 (data quality), M3 (headline), and M5 (paper correction).
  Mark the milestone `awaiting_review` in `TASKS.md` and stop the lane.

---

## 6. Risk register (short)

> **Review — REV-01, REV-03, REV-10:** The first two risk responses retain LEO assumptions.
> For GEO, screen and document maneuvers, TLE age/cadence, coverage, initial-state alignment,
> and force-model assumptions. Do not attribute a source/reference discrepancy to drag or
> SRP before timing and initialization errors are excluded. No hand-offs is not proof of
> zero uncertainty. The original thresholds below need justification for this target.

| Risk | Signal | Response |
|---|---|---|
| TLE hand-off jumps too large for the object/window | M0 p95 > 10 km | Change window (avoid Ap > 30 days) before changing object; log both attempts |
| Synthetic-vs-real gap dominated by drag/space weather (F8) | P3 sanity plot > 20 km at 1 T | Record it; optionally do R2-04 space weather; do not tune |
| Rollout diverges within 1 period for all models | P2 rollout_pos_1T_km > 1000 | Still report it. This is a finding about the method, not a bug, unless teacher-forced MAPE is also bad |
| `captum`/`shap` import breaks `qutils.ml` | ImportError on `import qutils.ml` | `pip install captum pandas` or import submodules directly |
| GMAT Python API unavailable on the agent's machine | `load_gmat` fails | Lane C is Hunter-run; agent prepares config + patch only, marks task `awaiting_review` |
| space-track rate limit | HTTP 429 / login failures | One request per (object, window); cache is mandatory; back off 60 s |

---

## 7. Phase notes (append-only)

**P0 + R1 + P1 code (2026-09-17, claude).** Repo skeleton, configs, the six `tlv/` modules,
the six scripts and the three test files are in and `pytest tests/` is green (13 passed, the
hand-off gate skips until a real dataset exists). qutils has the F4 constants fix and the
`transferMamba` docstring note. The whole real-data path was exercised end to end on a
*fabricated* TLE history (epoch-shifted copies of an SGP4 verification TLE, kept in the
scratchpad, not committed): build -> branches a/b/c -> CSV -> summary. Numbers from that run
are meaningless; only the plumbing was checked. What the next agent must know:

- Three qutils quirks found while wiring it up, all worked around in this repo, none fixed
  upstream: `plotStatePredictions` turns its unit conversion into a no-op that returns a
  *tuple* when `states=` and `DU/TU` are passed together (so pass `DU/TU` alone -- the
  defaults already label km / km/s); `mape` needs `axis=0` for per-state numbers; and
  `transferLSTM` leaves `fc` random, so `transfer()` routes LSTM through `transferModelAll`.
- TEME->J2000 is **not** velocity-magnitude preserving to 1e-9 as HANDOFF 3.2 assumes: TEME
  rotates w.r.t. GCRS at the precession rate, adding omega x r ~ 5e-8 km/s (6e-9 relative,
  1.2e-8 on the sma). `tests/test_tle_ephem.py` uses 1e-9 on position, 1e-7 on velocity and sma.
- The synthetic sets HANDOFF 9.1 names do not all exist: `regression/leo/450min-10000` and
  `900min-10000` are **empty directories**. What is actually there: `450min-4000`,
  `90min-10000`, `30min-{2000,3000,10000}`. Use `450min-4000` until R2-05 produces the
  matched set. No metadata keys in any of them (F5), as expected.
- astropy could not reach the IERS server here, so it fell back to bundled/predicted polar
  motion (arcsec-level). Irrelevant against a km-level SGP4 truth, but it will print a warning.

**Target change to Yamal 202 (2026-09-17, claude).** NORAD 28089 everywhere, verified against
the live CelesTrak element set: sma 42 165 km, period 86 166 s (1436.1 min), ecc 4.4e-4,
inc 6.57°, B* = 0. Window sizes follow the period, so the ASC-style window is now
7181 min × 9 (a 45-day span holds 9 of them), not 505 min × 40. Four things this breaks, none
of them fixable by an id change — they are open items H-5 … H-7 in HANDOFF §9.5:

- **No source domain.** Every GMAT-Thrust-Data set is LEO. Branch (a) and (c) have nothing
  to pretrain on until a matched GEO run exists (`data/regression/geo/7181min-10`). The
  branch configs now say `orbit: geo` so they will fail loudly rather than silently train on
  a LEO orbit.
- **The two windows nearly coincide.** 5 GEO periods = 4.99 days against a 7-day long window.
  The ASC/DDDAS contrast was 20× for the LEO target; it is 1.4× now.
- **Drag was the whole OOD story (F8) and there is no drag at GEO.** B* is fitted as 0, so
  `make_gmat_config.py` clamps Cd·A/m to its floor and the number means nothing. The
  synthetic-vs-real gap at GEO will be SRP and triaxiality, not atmosphere.
- **"Passive" is an assumption here, not a fact.** Inclination 6.57° says north-south station
  keeping stopped years ago, but east-west or relocation burns are not ruled out. That has to
  be screened in the window, the way PLAN §6 risk 1 screens hand-off jumps.

Also: 1 period is now 1436 samples rather than 101, so ASC-style "train on one period" is 14×
more training data than the LEO plan assumed. The DDDAS 2-point style is unaffected.

**Documentation review (2026-09-20, codex).** Added HANDOFF §10 and inline annotations for
reference-data claims, forecast origins, matched GMAT inputs, observation budgets, temporal
generalization, baselines/metrics, neutral gates, architecture interpretation, reproducibility,
and execution drift. F-05–F-14 track the open remedies. In particular, the current Yamal "1T"
metric is a 120-second rollout and `train_size=2` exposes three states. This is a documentation
review only; no code fixes, new training results, or protocol approvals are implied.

**P1 real data on real TLEs (2026-09-21, claude).** The blocker was never the code: it was
H-2. Credentials were in `.env`, so the window was screened rather than guessed —
`scripts/screen_window.py` over 873 TLEs (2026-01-01 .. 2026-09-21) ranks spans by coverage,
largest TLE gap and |d(sma)| step. Yamal 202's sma spans 1.14 km in nine months, so it is
passive over the whole search span; the chosen window is 2026-02-18 .. 2026-04-08 (124 TLEs,
19.0 h max gap, 0.068 km max step) and the old June placeholder was the worse of the two
(48.8 h gap, 0.62 km step). Both datasets are built and the M0 data gate passes: hand-off
median 0.63 km / p95 2.13 km over 131 hand-offs (8620 min × 8) and 0.66 / 1.94 over 74
(10080 min × 4). What the next agent must know:

- **The old "1T" rollout number was a 120-second rollout** (REV-02) and the old scoring slice
  re-predicted a training target (REV-04). Both are fixed at the root: the forecast origin is
  index `train_size`, the last observed state. Any result row written before 2026-09-21 is
  not comparable to one written after.
- **The ASC window is 6 periods now (8620 min × 8), not 5.** A 5-period window physically
  cannot contain a 5T horizon measured from a 1-period-in origin — it fell 0.0002 periods
  short of even 4T. The GMAT source config moved with it (`numMinsToProp: 8620`), so R2-05
  must be run at the new length or branch (a)/(c) will compare unequal spans.
- **The first real numbers say what F3 predicted.** Branch (b) ASC-style, Mamba, seed 0:
  teacher-forced position MAPE 0.18 % (inside the M1 target) but closed-loop rollout error
  6 229 km at 1T and 71 303 km at 5T, against a hand-off consistency of ~0.5 km. Teacher
  forcing flatters this model by four orders of magnitude. Expect the same for every branch,
  and do not let a good MAPE column stand in for propagation.
- astropy still cannot reach the IERS server here and falls back to bundled polar motion
  (arcsec-level, irrelevant against km-level SGP4 truth). It prints a warning on every run.

---

## 8. Decision log (append-only)

| Date | Decision | Why | Where it changed |
|---|---|---|---|
| 2026-09-16 | Regression is the primary task; classification out of scope for pass 1 | Hunter | HANDOFF §0 |
| 2026-09-16 | Real truth = SGP4-densified TLE segments at 60 s, TEME→J2000 | Hunter | HANDOFF §0, §2 |
| 2026-09-16 | LEO passive objects; Envisat 27386 primary | Hunter; densely tracked, no maneuvers | HANDOFF §4 |
| 2026-09-17 | **Primary target changed to Yamal 202 (NORAD 28089), a GEO object.** Supersedes the row above | Hunter | HANDOFF §0, §4, §5, §9.5; `configs/targets.yaml`; `configs/branch_*.yaml`; README |
| 2026-09-17 | GMAT data is addressed through a committed `data.yaml` (one `gmat` key = the GMAT-Thrust-Data `data/` folder), not a `data/synthetic` symlink | Hunter; matches `thrust-classification/data.yaml`, keeps the big sets out of the repo | `data.yaml`, `tlv/datasets.py::data_root`, HANDOFF §2/§3/§5, README, P0-05 |
| 2026-09-16 | space-track `gp_history` with env-var creds | Hunter | HANDOFF §3.1 |
| 2026-09-16 | Branch (a) is defined as the ASC procedure *as actually run* (zero-shot, F1) | Audit finding | HANDOFF §1 F1, §4 |
| 2026-09-16 | Rollout metric added beside every teacher-forced metric | Audit finding F3 | HANDOFF §3.5, §6.1 |
| 2026-09-17 | Shared branch-script plumbing lives in `scripts/_common.py`, not a seventh `tlv` module | Keeps rule 9 (six modules) without copying the plot set into three scripts | `scripts/_common.py` |
| 2026-09-17 | `tlv/datasets.py` also exports `redim()` | `nonDim2Dim6` mutates its input; every caller needs the copying inverse of `nondim()` | `tlv/datasets.py` |
| 2026-09-17 | Frame-invariance test tolerance is 1e-7 on velocity and sma, 1e-9 on position | TEME precesses w.r.t. GCRS, so \|v\| is not invariant at 1e-9 -- the spec's tolerance is unattainable and the transform is correct | `tests/test_tle_ephem.py` |
| 2026-09-17 | qutils installed as a copy, not editable (skbuild editable install fails) | Re-run `pip install ../qutils` after each qutils edit | `README.md` |
| 2026-09-21 | TLE window is 2026-02-18 .. 2026-04-08, chosen by `scripts/screen_window.py` before any model was run | H-2 needed a screened window, not a placeholder; the criteria (coverage, TLE gap, |d(sma)| step) are recorded in `configs/targets.yaml` and reproducible from the script | `configs/targets.yaml`, HANDOFF §5 step 1, §9.6 |
| 2026-09-21 | **ASC-style window is 6 periods (8620 min × 8), not 5 (7181 × 9)** | REV-02: horizons are measured from the forecast origin, so a 5-period window ends before even a 4T horizon and `rollout_pos_5T_km` could only ever be NaN. The training span stays at one period, so the equal-budget rule is untouched; the ASC train fraction moves 1/5 → 1/6 | `configs/targets.yaml`, `configs/branch_{a,b,c}.yaml`, `configs/gmat/gmat_28089_asc.yaml`, HANDOFF §4, §5, §9.6 |
| 2026-09-21 | Forecast origin is the last *observed* state (index `train_size`); scoring starts at `train_size + 1` | REV-02/REV-04: `create_datasets` trains on transitions through x_{train_size-1} → x_{train_size}, so the old rollout seed (`train_size - 1`) re-predicted a training target and the old "1T" number was a 120 s rollout | `tlv/train.py`, `tlv/metrics.py`, `scripts/_common.py`, `tests/test_transfer.py` |
| 2026-09-21 | CSV gains `window`, `handoff_n`, `origin_sec`, `tle_sha256`; `noise_floor_*` renamed `handoff_*` | REV-01/REV-09: a row has to say which window it came from and where its forecast started, and the hand-off statistic is consistency between orbit products, not an accuracy floor | `tlv/metrics.py`, `scripts/_common.py`, `scripts/summarize.py`, HANDOFF §3.6 |
| 2026-09-21 | `scripts/screen_window.py` is a seventh script (not a seventh `tlv` module) | Rule 9 caps `tlv/`, not `scripts/`; window screening is a one-off data decision, not library code | `scripts/screen_window.py` |

