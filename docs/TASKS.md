# tl-validation — Running Task List

Agents edit this file. Rules are in `PLAN.md` §4. Specs are in `HANDOFF.md` (section refs below).

**Status values:** `todo` · `in_progress` · `blocked` · `awaiting_review` · `done` · `dropped`
**Open items owned by Hunter** (from `HANDOFF.md` §9.5): `H-1` secondary object · `H-2` TLE window
· `H-3` ISS case in/out · `H-4` publish corrected CR3BP tables

Last updated: 2026-09-17 (claude: P0, R1, P1 code, P2-01..03, P5-01, R2-03)

---

## Milestone board

| Milestone | Gate (HANDOFF §6.2) | Status | Reviewed by |
|---|---|---|---|
| M0 real dataset validated | tests green; hand-off med < 2 km, p95 < 10 km; TLE sha256 in meta | blocked (H-2: creds + confirmed window; all code ready) | |
| M1 branch (b) ASC-style | Mamba+LSTM × 3 seeds; plot set + CSV rows | todo | |
| M1b branch (b) DDDAS-style 7-day | same, 7-day window | todo | |
| M2 branch (a) zero-shot | source MAPE ≤ 0.1 % synthetic; real rows with noise floor | todo | |
| M3 branch (c) transfer | full+feature, Mamba+LSTM, 3 seeds; F1 unit test passes | todo | |
| M4 long-term + summary | summary table + figures | todo | |
| M5 corrected CR3BP tables | pairs 1–3 re-run | todo | |

---

## P0 — Environment

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P0-01 | Create repo skeleton: `tlv/` (6 empty modules + `__init__`), `scripts/`, `tests/`, `configs/`, `data/.gitkeep`, `results/.gitkeep`, `figures/.gitkeep`; extend `.gitignore` with `data/`, `results/*.csv`, `figures/*.pdf`, `figures/*.png` | done | claude | — | `tree -L 2` matches HANDOFF §3; `git status` clean after commit |2026-09-17 claude: code: tlv/ 6 modules, scripts/, tests/, configs/, .gitignore extended |
| P0-02 | `requirements.txt` per HANDOFF §3 + `captum pandas pytest` (see PLAN §6); `README.md` one screen: install, env vars, the §9.6 command list | done | claude | P0-01 | `pip install -r requirements.txt` succeeds on a fresh venv; `python -c "import qutils.ml.regression, sgp4, astropy"` |2026-09-17 claude: requirements.txt + README.md written; qutils installs as a copy (skbuild editable install fails) -- re-run `pip install ../qutils` after qutils edits |
| P0-03 | `configs/targets.yaml`: Envisat 27386 (name, norad, period_sec placeholder → filled by P1-04), windows `asc: 505 min / 40`, `long: 10080 min / 4`, `dt: 60`; secondary object entry left `null` pending H-1 | done | claude | P0-01 | file parses; loader in P1-05 reads it |2026-09-17 claude: configs/targets.yaml written; period_sec stays null, it is written to the dataset's meta.json by build_real_dataset.py (see deviation note in P1-04) |
| P0-04 | `configs/branch_{a,b,c}.yaml` with the fixed hyper-parameters of HANDOFF §4 (Mamba: d_model 6, n_layers 1, d_conv 4 / 16; LSTM hidden 30; Adam 1e-3; Huber; epochs 5/2; DDDAS 50; batch 8; seeds [0,1,2]) | done | claude | P0-01 | values match HANDOFF §4 table exactly; no extra keys |2026-09-17 claude: configs/branch_{a,b,c}.yaml written; branch_b gains a third style `asc2` = the 2-epoch equal-budget row of P2-07 |
| P0-05 | Symlink `data/synthetic -> ~/git/GMAT-Thrust-Data/data` (document in README; not committed) | done | claude | P0-01 | `ls data/synthetic/regression/leo/450min-10000/statesArrayNoThrust.npz` |2026-09-17 claude: symlink made; NOTE 450min-10000 and 900min-10000 are EMPTY dirs -- the usable sets are 450min-4000, 90min-10000, 30min-* |

## R1 — qutils constants fix (HANDOFF §8.2, F4) — lane B

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| R1-01 | In `qutils/orbital.py` define `MU_EARTH = 398600.4415`, `DU_EARTH = 6378.1`, `TU_EARTH = (DU_EARTH**3/MU_EARTH)**0.5`; use as defaults in `dim2NonDim4/6`, `nonDim2Dim4/6`, `orbitalEnergy`. Reuse existing `MU_EARTH_JGM2` | done | claude | P0-02 | `python -c "from qutils.orbital import *; import numpy as np; x=np.random.rand(5,6); assert np.allclose(nonDim2Dim6(dim2NonDim6(x.copy())), x, atol=1e-12)"` |2026-09-17 claude: MU_EARTH/DU_EARTH/TU_EARTH added, used as defaults in dim2NonDim4/6, nonDim2Dim4/6, orbitalEnergy; roundtrip assert passes at 1e-12 |
| R1-02 | Docstring on `transferMamba`: state that `in_proj` and `A_log` stay frozen; `transferModelAll` is the full-fine-tune path (F2). No code change | done | claude | — | diff touches docstring only |2026-09-17 claude: docstring only |
| R1-03 | Commit to qutils with message noting the 0.23 % velocity change in default outputs of old scripts | done | claude | R1-01, R1-02 | commit hash recorded here |2026-09-17 claude: committed in ~/git/qutils as a510e83 (not pushed) |
| R1-04 | `tests/test_datasets.py::test_nondim_roundtrip` in tl-validation (1e-12) | done | claude | R1-01, P1-05 | `pytest tests/test_datasets.py -k roundtrip` green |2026-09-17 claude: tests/test_datasets.py::test_nondim_roundtrip green (+ test_nondim_copies for guardrail 9.3.1) |

## R2 — GMAT-Thrust-Data config change (HANDOFF §8.1, F5, F7) — lane C

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| R2-01 | Patch `python/generateSpacecraftThrustOpt.py`: `fixedElements` branch (SMA/ECC/INC/RAAN/AOP/TA + `_spread`, `dryMass/dragArea/Cd/srpArea/Cr`), `epoch` field with the old default. Keep the `else` path byte-identical | todo | | P0-05 | running the old `config.yaml` (no new keys) with `--no-save` produces identical first-run state to before the patch (compare 6 numbers printed) | |
| R2-02 | Add metadata keys to every `np.savez` in that file: `elements (N,6)`, `phys (N,5)`, `epoch (str)`, `dt` | todo | | R2-01 | `np.load(...).files` shows the 5 keys; `convertToOE.py` and `combineDatasets.py` still run on the new file | |
| R2-03 | Write `configs/gmat_27386.yaml` generator: `scripts/make_gmat_config.py --norad 27386` reads the first real state of the ASC window, `ECI2OE` → fixed elements (deg), epoch string in GMAT `UTCGregorian` format, `Cd*A/m = 2*Bstar/0.1570` clamped to [0.005, 0.05], `DryMass 80`, `Cd 2.2`, `DragArea = CdAm*80/2.2`; `num_runs: 10`, spreads: SMA 5 km, ECC 0.0005, INC/RAAN/AOP/TA 0.1 deg for runs ≥ 1 (run 0 exact — implement as `_spread` applied only when `i > 0`) | blocked | claude | P1-06, R2-01 | config file written; raw and clamped B* logged in `meta` comment line |2026-09-17 claude: scripts/make_gmat_config.py written (ECI2OE -> fixed elements, B* -> Cd*A/m clamped to [0.005, 0.05]); needs the real dataset to read the first state -- blocked_by H-2 (window) / P1-06 |
| R2-04 | *(optional, M4)* Space-weather fields `F107`, `F107A`, `MagneticIndex` on `Atmos` from CelesTrak SW-All for the epoch | todo | | R2-01 | flag `--space-weather` in the generate script; default off | |
| R2-05 | **Hunter-run:** `cp configs/gmat_27386.yaml ~/git/GMAT-Thrust-Data/config.yaml && ./generateGMATNoThrustDataset.sh` → `data/regression/leo/505min-10` | todo | | R2-02, R2-03 | npz with 5 keys exists; `awaiting_review` until Hunter confirms the run | |

## R3 — pendulumRNN F1 fix (HANDOFF §8.3) — lane D

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| R3-01 | In `scripts/conferences/ASC2025/mambaTLCR3BP.py`: target phase uses `trainModel(newModel, …)`, `plotStatePredictions(newModel, …)`; `plotGen` block and MAPE printout use the target model's prediction | todo | | — | diff is ≤ 6 lines; a `print` of `sum(p.abs().sum() for p in newModel.parameters())` before/after target training differs | |
| R3-02 | Re-run orbit pairs 1, 2, 3 (`python mambaTLCR3BP.py <pair>`) with the fix, `compareLSTM=True`, `plotGen=True` | todo | | R3-01 | figures written to `figures/`; MAPE printouts captured to `results/asc2025_corrected_raw.txt` | |
| R3-03 | `results/asc2025_corrected.md`: Tables 1–3 layout, columns "paper (zero-shot as run)" vs "corrected (fine-tuned 2 epochs)" for Mamba and LSTM | todo | | R3-02 | table complete; one paragraph stating whether fine-tuning helped, hurt, or did nothing per pair | |
| R3-04 | Milestone M5 → `awaiting_review` (H-4) | todo | | R3-03 | | |

## P1 — Real data (HANDOFF §3.1, §3.2, §2) — lane A

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P1-01 | `tlv/spacetrack.py::fetch_tle_history` with disk cache, env-var creds, single query per (norad, start, stop); `scripts/fetch_tles.py` CLI | done | claude | P0-02 | second call returns from cache with no HTTP; cache file sha256 printed |2026-09-17 claude: tlv/spacetrack.py + scripts/fetch_tles.py; cache-hit path exercised offline, 429 backs off 60 s once |
| P1-02 | Choose the 45-day window (placeholder `2026-06-01 … 2026-07-15`): check CelesTrak SW-All, require Ap < 30 every day; record the window and max Ap in `configs/targets.yaml` | blocked | | P1-01 | blocked_by H-2 until Hunter confirms; agent proposes a window and logs it | |
| P1-03 | `tlv/tle_ephem.py::parse_tles`, `segment_ephemeris` (propagate each TLE epoch→next epoch at 60 s; `tle_index`, `jump_km` at hand-offs) | done | claude | P0-02 | unit test P1-08 green |2026-09-17 claude: segment_ephemeris uses one uniform 60 s grid and the most recent TLE per sample; epochs-jd0 cancellation needs a 0.1 ms tie-break nudge, see comment |
| P1-04 | `tlv/tle_ephem.py::teme_to_j2000` via astropy `TEME→GCRS`; compute `period_sec` from mean motion of the first TLE and write it to `configs/targets.yaml` | done | claude | P1-03 | invariance test P1-08 green; `period_sec` ≈ 6040 s for Envisat (**VERIFY** against TLE) |2026-09-17 claude: teme_to_j2000 via astropy TEME->GCRS; period_sec from no_kozai. DEVIATION: period_sec is written to the dataset meta.json, not configs/targets.yaml -- it belongs to the TLE set, and targets.yaml is a hand-edited file |
| P1-05 | `tlv/datasets.py`: `load_synthetic`, `load_real`, `nondim` (copies; explicit DU/TU from qutils constants), `split_periods`, `nearest_synthetic` | done | claude | P1-04, R1-01 | `tests/test_datasets.py` green (shapes, roundtrip, split) |2026-09-17 claude: load_synthetic/load_real/nondim/redim/split_periods/nearest_synthetic; tests green |
| P1-06 | `tlv/tle_ephem.py::to_windows`, `write_real_npz` with all keys of HANDOFF §2 + `meta.json`; `scripts/build_real_dataset.py` CLI | blocked | claude | P1-03, P1-04 | `data/real/27386/505min-40/statesArrayReal.npz` and `10080min-4/` exist with the 6 keys |2026-09-17 claude: code written (to_windows, write_real_npz, scripts/build_real_dataset.py); artifact needs space-track creds and the confirmed window -- blocked_by H-2 (window) / P1-06 |
| P1-07 | Hand-off histogram figure `figures/27386-HandoffJump.png` + median/p95 in `meta.json` | blocked | claude | P1-06 | median < 2 km, p95 < 10 km; if not, log and go to PLAN §6 risk 1 |2026-09-17 claude: code written inside build_real_dataset.py (histogram + median/p95 in meta.json + M0 gate warning); artifact needs the dataset -- blocked_by H-2 (window) / P1-06 |
| P1-08 | `tests/test_tle_ephem.py`: SGP4 self-check vs bundled `SGP4-VER.TLE`/`tcppver.out` (1e-6 km); `| done | claude |v|,a` invariance under frame rotation (1e-9 rel); hand-off statistics gate | todo | | P1-03, P1-04 | `pytest tests/test_tle_ephem.py` green |2026-09-17 claude: pytest tests/test_tle_ephem.py: 4 passed, 1 skipped. TOLERANCE CHANGED: velocity and sma at 1e-7, not 1e-9 -- TEME precesses w.r.t. GCRS (omega x r ~ 5e-8 km/s). Position stays 1e-9 |
| P1-09 | Milestone M0 → `awaiting_review` (human data-quality check of P1-07 figure) | todo | | P1-06, P1-07, P1-08 | | |

## P2 — Branch (b): real low-data (HANDOFF §4, §5 step 2)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P2-01 | `tlv/models.py::return_model`, `transfer(mode)` per HANDOFF §3.4 | done | claude | P0-04 | `torchinfo.summary` param counts: Mamba ≈ 906 (d_conv 4) / ≈ 1050 (d_conv 16), LSTM 4746 — matches ASC/DDDAS tables |2026-09-17 claude: torchinfo param counts match the papers exactly: Mamba 906 (d_conv 4) / 1050 (d_conv 16), LSTM 4746 |
| P2-02 | `tlv/train.py::fit` (wraps `qutils.ml.regression.trainModel`, Adam, returns seconds), `predict_teacher_forced`, `rollout` (closed loop from `train_size`, NaN before) | done | claude | P2-01 | `rollout` on a constant sequence returns that constant; shape `(T,6)` |2026-09-17 claude: fit()/predict_teacher_forced()/rollout(); rollout on an identity model reproduces the last training state (tests/test_transfer.py) |
| P2-03 | `tlv/metrics.py`: `energy_drift` (`orbitalEnergyJ2`), `position_error_at`, `noise_floor`, `append_result` with the fixed column list of HANDOFF §3.6 + `run_id`, `branch`, `style` | done | claude | P1-05 | CSV header matches HANDOFF §3.6 exactly; a dummy row appends |2026-09-17 claude: CSV header = HANDOFF 3.6 columns + run_id/branch/style; append_result rejects unknown columns; plot_rollout_error draws the noise-floor band |
| P2-04 | `scripts/branch_b_real_lowdata.py --style asc`: port of ASC source half on real data (train 1 period, 5 epochs, test 4 periods); standard plot set (HANDOFF §6.1) incl. rollout-vs-noise-floor figure; CSV row | blocked | claude | P2-02, P2-03, P1-09 | seed 0 Mamba run completes; 5 figures in `figures/27386-b-*`; row in `results/branch_b.csv` |2026-09-17 claude: scripts/branch_b_real_lowdata.py written and exercised end to end on a fabricated TLE history; needs the real dataset for a real row -- blocked_by H-2 (window) / P1-06 |
| P2-05 | Run matrix `{mamba, lstm} × {0,1,2}` for `--style asc` | todo | | P2-04 | 6 rows; median Mamba position MAPE logged here | |
| P2-06 | `--style dddas` (train_size 2, 50 epochs, d_conv 16) on the 7-day window; run matrix | todo | | P2-04 | 6 rows in `results/branch_b.csv` with `style=dddas`; energy-drift figure over 7 days | |
| P2-07 | Equal-budget rows for M3: `--style asc --epochs 2` (from-scratch, 2 epochs on 1 period), matrix | todo | | P2-05 | 6 rows with `epochs_tgt=2`, `tl_mode=none` | |
| P2-08 | Milestones M1, M1b → status per gate; phase note in `PLAN.md` §7 | todo | | P2-05, P2-06, P2-07 | | |

## P3 — Matched synthetic source (HANDOFF §5 step 3)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P3-01 | `scripts/check_matched_source.py`: load `data/synthetic/regression/leo/505min-10` run 0 and real window 0; align `t_j` (GMAT `(j+1)dt` vs real `j·dt`, HANDOFF F5); plot position difference vs time for 1 period; print gap at 1 T | todo | | R2-05, P1-06 | figure `figures/27386-SynthVsReal-1T.png`; gap logged; if > 20 km write the PLAN §6 risk-2 note, do not tune | |
| P3-02 | *(optional)* Multi-trajectory bank: `nearest_synthetic` over `regression/leo/450min-10000` to pick 9 neighbours of run 0 by `[a,e,i]`; save index list to `configs/source_bank_27386.yaml` | todo | | P1-05, P0-05 | list of 9 indices with distances | |

## P4 — Branch (a): zero-shot (HANDOFF §5 step 4)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P4-01 | `scripts/branch_a_zero_shot.py`: train on synthetic run 0, 1 period, 5 epochs; source plot set (`figures/27386-a-SourceDomain*`); then evaluate on real 5-period window with **no** training: teacher-forced + rollout; CSV rows `branch=a` | blocked | claude | P3-01, P2-02, P2-03 | seed 0 Mamba: source position MAPE ≤ 0.1 %; real rows written with noise floor |2026-09-17 claude: scripts/branch_a_zero_shot.py written and exercised end to end; needs the real dataset and the matched synthetic set (R2-05) -- blocked_by H-2 (window) / P1-06 |
| P4-02 | Run matrix `{mamba, lstm} × {0,1,2}` | todo | | P4-01 | 6 rows in `results/branch_a.csv` | |
| P4-03 | Save the source-trained state dicts to `results/models/27386-a-<model>-s<seed>.pt` for reuse in P5 (`qutils.ml.utils.saveModel` or `torch.save`) | todo | | P4-02 | 6 files; reload test in P5-01 | |
| P4-04 | Milestone M2 → status; phase note | todo | | P4-02, P4-03 | | |

## P5 — Branch (c): transfer (HANDOFF §5 step 5) — requires M1 and M2 green

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P5-01 | `tests/test_transfer.py`: after `transfer(src,tgt,'full')` + one `fit` step on random data, ≥ 1 parameter differs; after `'feature'`, `tgt.layers[0].mixer.in_proj` weights equal `src` (guards F1) | done | claude | P2-01, P2-02 | pytest green |2026-09-17 claude: tests/test_transfer.py: 4 tests, green -- full transfer changes the target and leaves the source alone; feature keeps in_proj and A_log frozen; LSTM copies every layer |
| P5-02 | `scripts/branch_c_transfer.py --tl {full,feature}`: load P4-03 model, `transfer`, `fit(newModel, real, train 1 period, 2 epochs)`, evaluate on 4 periods (teacher-forced + rollout); target plot set `figures/27386-c-TargetDomain*`; CSV rows `branch=c` | blocked | claude | P5-01, P4-03, P2-07 | seed 0 Mamba `full` completes; the equal-budget row from P2-07 with the same seed exists |2026-09-17 claude: scripts/branch_c_transfer.py written and exercised for tl=full and tl=feature; needs M1 and M2 green first (PLAN hard rule) -- blocked_by H-2 (window) / P1-06 |
| P5-03 | Run matrix `{full, feature} × {mamba, lstm} × {0,1,2}` | todo | | P5-02 | 12 rows in `results/branch_c.csv` | |
| P5-04 | *(optional)* `--tl full-multi`: pretrain on P3-02 bank (concatenated `create_datasets`), then fine-tune; matrix for Mamba only | todo | | P3-02, P5-02 | 3 rows | |
| P5-05 | Headline check (HANDOFF §6.2 M3 target): median over seeds, Mamba `full` vs P2-07 from-scratch-2-epoch vs P4 zero-shot, on teacher-forced MAPE **and** rollout 1 T. Write the three numbers and the noise floor here | todo | | P5-03 | numbers in this log; M3 → `awaiting_review` | |

## P6 — Long-term 7-day variants (HANDOFF §5 step 6)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P6-01 | Synthetic 7-day matched run: `configs/gmat_27386_long.yaml` (`numMinsToProp: 10080`, `num_runs: 3`); **Hunter-run** GMAT | todo | | R2-03 | `data/regression/leo/10080min-3` exists | |
| P6-02 | Branch (a) and (c) on the 7-day real window (train 1 day, test 6), Mamba + LSTM, 3 seeds; `style=long` rows | todo | | P6-01, P5-03, P2-06 | rows in `results/branch_{a,c}.csv` with `style=long` | |
| P6-03 | Phase note; M4 rows complete | todo | | P6-02 | | |

## P7 — Summary (HANDOFF §5 step 7)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P7-01 | `scripts/summarize.py`: read `results/branch_*.csv`; per (branch, model, tl_mode, style) median ± IQR over seeds; noise-floor row; write `results/summary.md` in the ASC Table 1–3 layout (columns x, y, z, ẋ, ẏ, ż MAPE) plus rollout and energy columns | blocked | claude | P6-03 | table renders; every row has ≥ 3 seeds or is flagged |2026-09-17 claude: scripts/summarize.py written and exercised; needs result rows -- blocked_by H-2 (window) / P1-06 |
| P7-02 | `figures/Summary-MAPE.pdf` and `figures/Summary-Rollout.pdf`: grouped bars a/b/c × Mamba/LSTM, error bars = IQR, dashed line = noise floor; title carries "truth = SGP4 from space-track TLEs" | todo | | P7-01 | figures exist; read the `dataviz` skill before plotting | |
| P7-03 | One-page `results/README.md`: what was claimed (ASC, DDDAS), what was measured, the three headline numbers, the F1 correction and its effect (link `results/asc2025_corrected.md`) | todo | | P7-01, R3-03 | ≤ 1 page; no number without seed count and noise floor | |
| P7-04 | Final phase note; M4 → `awaiting_review` | todo | | P7-03 | | |

## Follow-ups (append new IDs here if they do not belong to a phase above)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| F-01 | *(H-3)* ISS 25544 contamination case: run branches a/b/c on a window containing a known reboost; expected: all three fail at the burn; one figure | blocked | | H-3, P5-03 | | |
| F-02 | *(H-1)* Secondary lower-LEO passive object: satcat query `OBJECT_TYPE=ROCKET BODY, DECAY=null, PERIGEE 450--550`; propose 3 candidates with TLE count over the window; Hunter picks | blocked | | H-1, P1-01 | candidates logged here | |
| F-03 | After M3 green: move `teme_to_j2000`, `segment_ephemeris` into `qutils/orbital.py` (HANDOFF §8.2 item 3) | todo | | P5-05 | qutils commit; `tlv/tle_ephem.py` imports from qutils; tests still green | |

---

## Log format for agents

Append to the task's Log cell, newest last, pipe-safe (use `<br>` for line breaks):

```
2026-09-18 agentX: claimed<br>2026-09-18 agentX: pytest 3 passed<br>2026-09-18 agentX: done, commit ab12cd3
```
