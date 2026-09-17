# tl-validation — Execution Plan

Companion to `HANDOFF.md` (the spec) and `TASKS.md` (the running task list).
Read order for any agent: `HANDOFF.md` §0–§1 → this file → `TASKS.md`.

- `HANDOFF.md` is the source of truth for *what* and *why*. Do not restate it here.
- `PLAN.md` says *in what order*, *what can run in parallel*, and *how agents behave*.
- `TASKS.md` says *what is done, in progress, blocked*. Agents update it; humans read it.

---

## 1. Goal, in one paragraph

Validate the ASC 2025 transfer-learning claim and the DDDAS 2024 low-data claim on real
LEO ephemerides (space-track TLEs, SGP4-densified, J2000, 60 s). Three branches share one
target object and window: (a) zero-shot synthetic→real, (b) train-from-scratch on real
low data, (c) synthetic pretrain + 2-epoch real fine-tune. The headline result is (c) beating
both (a) and (b) at equal target budget, for Mamba, median over 3 seeds, with the SGP4
hand-off noise floor printed next to every number.

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

- After each task: update `TASKS.md`, commit.
- After each phase: append a 3–6 line phase note to §7 of this file (what was produced, what
  surprised you, what the next agent must know). Commit.
- A human review is required at M0 (data quality), M3 (headline), and M5 (paper correction).
  Mark the milestone `awaiting_review` in `TASKS.md` and stop the lane.

---

## 6. Risk register (short)

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

---

## 8. Decision log (append-only)

| Date | Decision | Why | Where it changed |
|---|---|---|---|
| 2026-09-16 | Regression is the primary task; classification out of scope for pass 1 | Hunter | HANDOFF §0 |
| 2026-09-16 | Real truth = SGP4-densified TLE segments at 60 s, TEME→J2000 | Hunter | HANDOFF §0, §2 |
| 2026-09-16 | LEO passive objects; Envisat 27386 primary | Hunter; densely tracked, no maneuvers | HANDOFF §4 |
| 2026-09-16 | space-track `gp_history` with env-var creds | Hunter | HANDOFF §3.1 |
| 2026-09-16 | Branch (a) is defined as the ASC procedure *as actually run* (zero-shot, F1) | Audit finding | HANDOFF §1 F1, §4 |
| 2026-09-16 | Rollout metric added beside every teacher-forced metric | Audit finding F3 | HANDOFF §3.5, §6.1 |
| 2026-09-17 | Shared branch-script plumbing lives in `scripts/_common.py`, not a seventh `tlv` module | Keeps rule 9 (six modules) without copying the plot set into three scripts | `scripts/_common.py` |
| 2026-09-17 | `tlv/datasets.py` also exports `redim()` | `nonDim2Dim6` mutates its input; every caller needs the copying inverse of `nondim()` | `tlv/datasets.py` |
| 2026-09-17 | Frame-invariance test tolerance is 1e-7 on velocity and sma, 1e-9 on position | TEME precesses w.r.t. GCRS, so \|v\| is not invariant at 1e-9 -- the spec's tolerance is unattainable and the transform is correct | `tests/test_tle_ephem.py` |
| 2026-09-17 | qutils installed as a copy, not editable (skbuild editable install fails) | Re-run `pip install ../qutils` after each qutils edit | `README.md` |
