# tl-validation — Running Task List

Agents edit this file. Rules are in `PLAN.md` §4. Specs are in `HANDOFF.md` (section refs below).

**Status values:** `todo` · `in_progress` · `blocked` · `awaiting_review` · `done` · `dropped`
**Open items owned by Hunter** (from `HANDOFF.md` §9.5): `H-1` secondary object · `H-2` TLE window
· `H-3` ISS case in/out · `H-4` publish corrected CR3BP tables

Last updated: 2026-09-21 (claude: P1-02/06/07/08 on real space-track data, M0 data gate green;
forecast-origin and hand-off-statistic fixes for F-06/F-08/F-05/F-13/F-14; ASC window 7181->8620)

Review annotations added: 2026-09-20 (codex). Existing completion logs are historical;
they do not establish resolution of [HANDOFF §10](HANDOFF.md#10-plan-review-2026-09-20).
F-05–F-14 below are open review follow-ups, not approved implementation changes.

---

## Milestone board

| Milestone | Gate (HANDOFF §6.2) | Status | Reviewed by |
|---|---|---|---|
| M0 real dataset validated | tests green; hand-off med < 2 km, p95 < 10 km; TLE sha256 in meta | **done** (gate green; window confirmed by Hunter 2026-09-21) | Hunter |
| M1 branch (b) ASC-style | Mamba+LSTM × 3 seeds; plot set + CSV rows | **done** (gate met; targets met: Mamba 0.321 % ≤ 1 %, below LSTM on all 6 states, rollout reported) | |
| M1b branch (b) DDDAS-style 7-day | same, 7-day window | **done** (gate met; energy-drift target missed: 16.4 % vs < 1 % — a finding, not a failure, per HANDOFF §6.2) | |
| M2 branch (a) zero-shot | source MAPE ≤ 0.1 % synthetic; real rows with noise floor | todo | |
| M3 branch (c) transfer | full+feature, Mamba+LSTM, 3 seeds; F1 unit test passes | todo | |
| M4 long-term + summary | summary table + figures | todo | |
| M5 corrected CR3BP tables | pairs 1–3 re-run | todo | |

> **Review impact:** M0 needs reference-quality/coverage reconciliation (REV-01/REV-10);
> M1/M1b need split, baseline, and window validation (REV-02/REV-04–REV-06); M2 needs a
> matched source and outcome-independent gates (REV-03/REV-07); M3 needs all of those plus
> verifiable paired budgets; M4 needs reliable artifact identity and aggregation (REV-09).
> The historical "all code ready" wording in M0 is not supported by this review. No milestone
> is marked complete by adding these annotations. M5's dependency on the GEO report is a
> proposed scheduling change (REV-10).

---

## P0 — Environment

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P0-01 | Create repo skeleton: `tlv/` (6 empty modules + `__init__`), `scripts/`, `tests/`, `configs/`, `data/.gitkeep`, `results/.gitkeep`, `figures/.gitkeep`; extend `.gitignore` with `data/`, `results/*.csv`, `figures/*.pdf`, `figures/*.png` | done | claude | — | `tree -L 2` matches HANDOFF §3; `git status` clean after commit |2026-09-17 claude: code: tlv/ 6 modules, scripts/, tests/, configs/, .gitignore extended |
| P0-02 | `requirements.txt` per HANDOFF §3 + `captum pandas pytest` (see PLAN §6); `README.md` one screen: install, env vars, the §9.6 command list | done | claude | P0-01 | `pip install -r requirements.txt` succeeds on a fresh venv; `python -c "import qutils.ml.regression, sgp4, astropy"` |2026-09-17 claude: requirements.txt + README.md written; qutils installs as a copy (skbuild editable install fails) -- re-run `pip install ../qutils` after qutils edits |
| P0-03 | `configs/targets.yaml`: Yamal 202 28089 (name, norad, period_sec placeholder → filled by P1-04), windows `asc: 7181 min / 9`, `long: 10080 min / 4`, `dt: 60`; secondary object entry left `null` pending H-1 | done | claude | P0-01 | file parses; loader in P1-05 reads it |2026-09-17 claude: configs/targets.yaml written; period_sec stays null, it is written to the dataset's meta.json by build_real_dataset.py (see deviation note in P1-04)<br>2026-09-17 claude: target changed to Yamal 202 (28089, GEO); windows resized to the 1436.1 min period |
| P0-04 | `configs/branch_{a,b,c}.yaml` with the fixed hyper-parameters of HANDOFF §4 (Mamba: d_model 6, n_layers 1, d_conv 4 / 16; LSTM hidden 30; Adam 1e-3; Huber; epochs 5/2; DDDAS 50; batch 8; seeds [0,1,2]) | done | claude | P0-01 | values match HANDOFF §4 table exactly; no extra keys |2026-09-17 claude: configs/branch_{a,b,c}.yaml written; branch_b gains a third style `asc2` = the 2-epoch equal-budget row of P2-07 |
| P0-05 | ~~Symlink `data/synthetic -> ~/git/GMAT-Thrust-Data/data`~~ **superseded:** committed `data.yaml` with one `gmat` key holding the absolute path to the GMAT-Thrust-Data data folder, same pattern as `thrust-classification/data.yaml`. `tlv.datasets.data_root()` reads it; nothing is symlinked or copied in | done | claude | P0-01 | `python -c "import tlv.datasets as d; print(d.data_root())"` resolves and `load_synthetic` finds a set under it |2026-09-17 claude: symlink made; NOTE 450min-10000 and 900min-10000 are EMPTY dirs -- the usable sets are 450min-4000, 90min-10000, 30min-*<br>2026-09-17 claude: symlink replaced by data.yaml; load_synthetic(root=None) resolves through it and raises a FileNotFoundError naming data.yaml when a set is missing |

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
| R2-03 | Write `configs/gmat/gmat_28089*.yaml` generator: `scripts/make_gmat_config.py --norad 28089` reads the first real state of the ASC window, `ECI2OE` → fixed elements (deg), epoch string in GMAT `UTCGregorian` format, `Cd*A/m = 2*Bstar/0.1570` clamped to [0.005, 0.05], `DryMass 80`, `Cd 2.2`, `DragArea = CdAm*80/2.2`; `num_runs: 10`, spreads: SMA 5 km, ECC 0.0005, INC/RAAN/AOP/TA 0.1 deg for runs ≥ 1 (run 0 exact — implement as `_spread` applied only when `i > 0`) | blocked | claude | P1-06, R2-01 | config file written; raw and clamped B* logged in `meta` comment line |2026-09-17 claude: scripts/make_gmat_config.py written (ECI2OE -> fixed elements, B* -> Cd*A/m clamped to [0.005, 0.05]); needs the real dataset to read the first state -- blocked_by H-2 (window) / P1-06<br>2026-09-17 claude: DONE differently -- emits the schema the generator reads today (lowerAlt/upperAlt pin SMA, existing *_spread keys); section 8.1 keys are a commented block. Both example configs hand-written and committed under configs/gmat/, verified byte-comparable to the generator's output on a fabricated 28089 window |
| R2-04 | *(optional, M4)* Space-weather fields `F107`, `F107A`, `MagneticIndex` on `Atmos` from CelesTrak SW-All for the epoch | todo | | R2-01 | flag `--space-weather` in the generate script; default off | |
| R2-05 | **Hunter-run:** `cp configs/gmat/gmat_28089_asc.yaml ~/git/GMAT-Thrust-Data/config.yaml && ./generateGMATNoThrustDataset.sh` → `data/regression/geo/8620min-10`. Config is committed, so this no longer waits on R2-03 | todo | | R2-02, R2-03 | npz with 5 keys exists; `awaiting_review` until Hunter confirms the run | |

## R3 — pendulumRNN F1 fix (HANDOFF §8.3) — lane D

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| R3-01 | In `scripts/conferences/ASC2025/mambaTLCR3BP.py`: target phase uses `trainModel(newModel, …)`, `plotStatePredictions(newModel, …)`; `plotGen` block and MAPE printout use the target model's prediction | awaiting_review | claude | — | diff is ≤ 6 lines; a `print` of `sum(p.abs().sum() for p in newModel.parameters())` before/after target training differs |2026-09-21 claude: 2 functional lines changed (+ the weight-sum print and a comment). Everything downstream already flowed from `networkPrediction_target`, so the plotGen block and the MAPE printout are fixed by the same two lines. Pair 1 verified: Mamba target weight sum 560.027550 -> 558.792336 (delta -1.235), LSTM 516.527171 -> 564.402949 (delta +47.88); before the fix the delta was exactly 0. **Left uncommitted in ~/git/pendulumRNN for Hunter** -- that repo is on `main` and this session was not asked to commit there. NOTE the script needs a `figures/` dir in the CWD or it raises FileNotFoundError at the first savefig |
| R3-02 | Re-run orbit pairs 1, 2, 3 (`python mambaTLCR3BP.py <pair>`) with the fix, `compareLSTM=True`, `plotGen=True` | todo | | R3-01 | figures written to `figures/`; MAPE printouts captured to `results/asc2025_corrected_raw.txt` | |
| R3-03 | `results/asc2025_corrected.md`: Tables 1–3 layout, columns "paper (zero-shot as run)" vs "corrected (fine-tuned 2 epochs)" for Mamba and LSTM | todo | | R3-02 | table complete; one paragraph stating whether fine-tuning helped, hurt, or did nothing per pair | |
| R3-04 | Milestone M5 → `awaiting_review` (H-4) | todo | | R3-03 | | |

## P1 — Real data (HANDOFF §3.1, §3.2, §2) — lane A

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P1-01 | `tlv/spacetrack.py::fetch_tle_history` with disk cache, env-var creds, single query per (norad, start, stop); `scripts/fetch_tles.py` CLI | done | claude | P0-02 | second call returns from cache with no HTTP; cache file sha256 printed |2026-09-17 claude: tlv/spacetrack.py + scripts/fetch_tles.py; cache-hit path exercised offline, 429 backs off 60 s once<br>2026-09-18 claude: credentials come from the environment or a gitignored .env (.env.example committed); load_env + credentials() in spacetrack.py, tests/test_spacetrack.py covers parsing, env precedence, the unset message and the cache-hit-no-network rule |
| P1-02 | (H-2 closed 2026-09-21) Choose the 45-day window. **Criteria changed for GEO:** Ap is irrelevant at 35 800 km; screen on TLE coverage, largest TLE gap and |d(sma)| step instead (`scripts/screen_window.py`) | awaiting_review | claude | P1-01 | agent proposes a window and logs it; Hunter confirms (H-2) |2026-09-21 claude: screened 873 TLEs over 2026-01-01..2026-09-21. sma spans 1.14 km in 9 months -> object is passive, no burn signature. Chosen: **2026-02-18 .. 2026-04-08**, 124 TLEs, max gap 19.0 h, max \|d(sma)\| 0.068 km. The old 2026-06-01 placeholder is worse on both (48.8 h gap, 0.62 km step). Criteria recorded in configs/targets.yaml; `confirmed: false` until Hunter signs off |
| P1-03 | `tlv/tle_ephem.py::parse_tles`, `segment_ephemeris` (propagate each TLE epoch→next epoch at 60 s; `tle_index`, `jump_km` at hand-offs) | done | claude | P0-02 | unit test P1-08 green |2026-09-17 claude: segment_ephemeris uses one uniform 60 s grid and the most recent TLE per sample; epochs-jd0 cancellation needs a 0.1 ms tie-break nudge, see comment |
| P1-04 | `tlv/tle_ephem.py::teme_to_j2000` via astropy `TEME→GCRS`; compute `period_sec` from mean motion of the first TLE and write it to `configs/targets.yaml` | done | claude | P1-03 | invariance test P1-08 green; `period_sec` ≈ 86166 s for Yamal 202 (checked against the 2026-09-17 TLE) |2026-09-17 claude: teme_to_j2000 via astropy TEME->GCRS; period_sec from no_kozai. DEVIATION: period_sec is written to the dataset meta.json, not configs/targets.yaml -- it belongs to the TLE set, and targets.yaml is a hand-edited file |
| P1-05 | `tlv/datasets.py`: `load_synthetic`, `load_real`, `nondim` (copies; explicit DU/TU from qutils constants), `split_periods`, `nearest_synthetic` | done | claude | P1-04, R1-01 | `tests/test_datasets.py` green (shapes, roundtrip, split) |2026-09-17 claude: load_synthetic/load_real/nondim/redim/split_periods/nearest_synthetic; tests green |
| P1-06 | `tlv/tle_ephem.py::to_windows`, `write_real_npz` with all keys of HANDOFF §2 + `meta.json`; `scripts/build_real_dataset.py` CLI | done | claude | P1-03, P1-04 | `data/real/28089/8620min-8/statesArrayReal.npz` and `10080min-4/` exist with the 6 keys |2026-09-21 claude: both built from the screened window. 8620min-8 [8,8620,6], 10080min-4 [4,10080,6], 134 TLEs, 69622 samples. NOTE the ASC set is 8620min-8, not 7181min-9 -- see the decision log<br>2026-09-17 claude: code written (to_windows, write_real_npz, scripts/build_real_dataset.py); artifact needs space-track creds and the confirmed window -- blocked_by H-2 (window) / P1-06 |
| P1-07 | Hand-off histogram figure `figures/28089-<minutes>min-HandoffJump.png` + median/p95/count in `meta.json` | done | claude | P1-06 | median < 2 km, p95 < 10 km; if not, log and go to PLAN §6 risk 1 |2026-09-21 claude: GATE PASSES. 8620min-8: median 0.629 km, p95 2.127 km, 131 hand-offs. 10080min-4: 0.657 / 1.943 over 74. One figure per dataset now (the two builds used to overwrite each other)<br>2026-09-17 claude: code written inside build_real_dataset.py (histogram + median/p95 in meta.json + M0 gate warning); artifact needs the dataset -- blocked_by H-2 (window) / P1-06 |
| P1-08 | `tests/test_tle_ephem.py`: SGP4 self-check vs bundled `SGP4-VER.TLE`/`tcppver.out` (1e-6 km); \|v\|, a invariance under frame rotation (1e-7 rel, see decision log); hand-off statistics gate | done | claude | P1-03, P1-04 | `pytest tests/test_tle_ephem.py` green |2026-09-17 claude: pytest tests/test_tle_ephem.py: 4 passed, 1 skipped. TOLERANCE CHANGED: velocity and sma at 1e-7, not 1e-9 -- TEME precesses w.r.t. GCRS (omega x r ~ 5e-8 km/s). Position stays 1e-9 |
| P1-09 | Milestone M0 → `awaiting_review` (human data-quality check of P1-07 figure) | done | claude | P1-06, P1-07, P1-08 | |2026-09-21 claude: `pytest tests/` 20 passed (the hand-off gate no longer skips -- it runs against both real datasets). Hunter: check figures/28089-8620min-HandoffJump.png and confirm the window in configs/targets.yaml |

## P2 — Branch (b): real low-data (HANDOFF §4, §5 step 2)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P2-01 | `tlv/models.py::return_model`, `transfer(mode)` per HANDOFF §3.4 | done | claude | P0-04 | `torchinfo.summary` param counts: Mamba ≈ 906 (d_conv 4) / ≈ 1050 (d_conv 16), LSTM 4746 — matches ASC/DDDAS tables |2026-09-17 claude: torchinfo param counts match the papers exactly: Mamba 906 (d_conv 4) / 1050 (d_conv 16), LSTM 4746 |
| P2-02 | `tlv/train.py::fit` (wraps `qutils.ml.regression.trainModel`, Adam, returns seconds), `predict_teacher_forced`, `rollout` (closed loop from `train_size`, NaN before) | done | claude | P2-01 | `rollout` on a constant sequence returns that constant; shape `(T,6)` |2026-09-17 claude: fit()/predict_teacher_forced()/rollout(); rollout on an identity model reproduces the last training state (tests/test_transfer.py) |
| P2-03 | `tlv/metrics.py`: `energy_drift` (`orbitalEnergyJ2`), `position_error_at`, `noise_floor`, `append_result` with the fixed column list of HANDOFF §3.6 + `run_id`, `branch`, `style` | done | claude | P1-05 | CSV header matches HANDOFF §3.6 exactly; a dummy row appends |2026-09-17 claude: CSV header = HANDOFF 3.6 columns + run_id/branch/style; append_result rejects unknown columns; plot_rollout_error draws the noise-floor band |
| P2-04 | `scripts/branch_b_real_lowdata.py --style asc`: port of ASC source half on real data (train 1 period, 5 epochs, test the remaining 5); standard plot set (HANDOFF §6.1) incl. rollout-vs-hand-off figure; CSV row | done | claude | P2-02, P2-03, P1-09 | seed 0 Mamba run completes; 5 figures in `figures/28089-b-*`; row in `results/branch_b.csv` |2026-09-21 claude: seed 0 Mamba on real data: teacher-forced position MAPE 0.178 %, rollout 6229 km at 1T / 71303 km at 5T, hand-off med 0.53 / p95 1.16 km (n=14). 5 figures + 1 row written. The rollout/teacher-forced gap is 4 orders of magnitude -- F3 exactly<br>2026-09-17 claude: scripts/branch_b_real_lowdata.py written and exercised end to end on a fabricated TLE history; needs the real dataset for a real row -- blocked_by H-2 (window) / P1-06 |
| P2-05 | Run matrix `{mamba, lstm} × {0,1,2}` for `--style asc` | done | claude | P2-04 | 6 rows; median Mamba position MAPE logged here |2026-09-21 claude: 18 runs, no failures. Medians over 3 seeds, teacher-forced position MAPE (mean of x,y,z): asc mamba **0.321 %** vs lstm 10.8 %; asc2 (2 epochs) mamba **0.465 %** vs lstm 52.9 %; dddas mamba 5.23 % vs lstm 193 %. Mamba's median MAPE is below LSTM's on **all six** states in all three styles. Rollout at 1T (median): asc mamba 3.66e4 km, lstm 4.50e4 km -- i.e. **no model propagates**; the orbit radius is 4.2e4 km and the hand-off consistency is 0.53 km. LSTM rollouts are near-identical at 1T and 5T, so they collapse to a fixed point rather than diverging |
| P2-06 | `--style dddas` (train_size 2, 50 epochs, d_conv 16) on the 7-day window; run matrix | done | claude | P2-04 | 6 rows in `results/branch_b.csv` with `style=dddas`; energy-drift figure over 7 days |2026-09-21 claude: 6 rows. **M1b energy target missed and that is the finding:** Mamba teacher-forced J2 energy drift over 7 days is 16.4 % median (7.7 / 29.9 / 16.4 by seed), not < 1 %. LSTM is 981 %. Energy figures in figures/28089-b-dddas-s*-RealConservedQuantity-*.pdf |
| P2-07 | Equal-budget rows for M3: `--style asc2` (from-scratch, 2 epochs on 1 period), matrix. **The CLI option is `--style asc2`, not `--epochs 2`** (F-14/REV-10) | done | claude | P2-05 | 6 rows with `epochs_tgt=2`, `tl_mode=none` |2026-09-21 claude: 6 rows, epochs_tgt=2, tl_mode=none. **This is the number branch (c) has to beat:** Mamba median teacher-forced position MAPE 0.465 %, rollout 1T 4.96e4 km, on window 0 of 8620min-8, seeds 0/1/2 |
| P2-08 | Milestones M1, M1b → status per gate; phase note in `PLAN.md` §7 | done | claude | P2-05, P2-06, P2-07 | |2026-09-21 claude: M1 gate and targets met; M1b gate met, energy target missed (logged as a finding per HANDOFF §6.2). Phase note appended |

## P3 — Matched synthetic source (HANDOFF §5 step 3)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P3-01 | `scripts/check_matched_source.py`: load `<gmat>/regression/geo/8620min-10` run 0 (`<gmat>` = the data.yaml path) and real window 0; align `t_j` (GMAT `(j+1)dt` vs real `j·dt`, HANDOFF F5); plot position difference vs time for 1 period; print gap at 1 T | todo | | R2-05, P1-06 | figure `figures/28089-SynthVsReal-1T.png`; gap logged; if > 20 km write the PLAN §6 risk-2 note, do not tune | |
| P3-02 | *(optional)* Multi-trajectory bank: `nearest_synthetic` over a GEO bank to pick 9 neighbours of run 0 by `[a,e,i]`; save index list to `configs/source_bank_28089.yaml`. **Was `regression/leo/450min-10000`; there is no GEO bank, so this needs a GMAT run of its own now (H-5)** | blocked | | P1-05, P0-05 | list of 9 indices with distances | |

## P4 — Branch (a): zero-shot (HANDOFF §5 step 4)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P4-01 | `scripts/branch_a_zero_shot.py`: train on synthetic run 0, 1 period, 5 epochs; source plot set (`figures/28089-a-SourceDomain*`); then evaluate on real 5-period window with **no** training: teacher-forced; CSV rows `branch=a` | blocked | claude | P3-01, P2-02, P2-03 | seed 0 Mamba: source position MAPE ≤ 0.1 %; real rows written with noise floor |2026-09-17 claude: scripts/branch_a_zero_shot.py written and exercised end to end; needs the real dataset and the matched synthetic set (R2-05) -- blocked_by H-2 (window) / P1-06 |
| P4-02 | Run matrix `{mamba, lstm} × {0,1,2}` | todo | | P4-01 | 6 rows in `results/branch_a.csv` | |
| P4-03 | Save the source-trained state dicts to `results/models/28089-a-<model>-s<seed>.pt` for reuse in P5 (`qutils.ml.utils.saveModel` or `torch.save`) | todo | | P4-02 | 6 files; reload test in P5-01 | |
| P4-04 | Milestone M2 → status; phase note | todo | | P4-02, P4-03 | | |

## P5 — Branch (c): transfer (HANDOFF §5 step 5) — requires M1 and M2 green

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P5-01 | `tests/test_transfer.py`: after `transfer(src,tgt,'full')` + one `fit` step on random data, ≥ 1 parameter differs; after `'feature'`, `tgt.layers[0].mixer.in_proj` weights equal `src` (guards F1) | done | claude | P2-01, P2-02 | pytest green |2026-09-17 claude: tests/test_transfer.py: 4 tests, green -- full transfer changes the target and leaves the source alone; feature keeps in_proj and A_log frozen; LSTM copies every layer |
| P5-02 | `scripts/branch_c_transfer.py --tl {full,feature}`: load P4-03 model, `transfer`, `fit(newModel, real, train 1 period, 2 epochs)`, evaluate on 4 periods (teacher-forced); target plot set `figures/28089-c-TargetDomain*`; CSV rows `branch=c` | blocked | claude | P5-01, P4-03, P2-07 | seed 0 Mamba `full` completes; the equal-budget row from P2-07 with the same seed exists |2026-09-17 claude: scripts/branch_c_transfer.py written and exercised for tl=full and tl=feature; needs M1 and M2 green first (PLAN hard rule) -- blocked_by H-2 (window) / P1-06 |
| P5-03 | Run matrix `{full, feature} × {mamba, lstm} × {0,1,2}` | todo | | P5-02 | 12 rows in `results/branch_c.csv` | |
| P5-04 | *(optional)* `--tl full-multi`: pretrain on P3-02 bank (concatenated `create_datasets`), then fine-tune; matrix for Mamba only | todo | | P3-02, P5-02 | 3 rows | |
| P5-05 | Headline check (HANDOFF §6.2 M3 target): median over seeds, Mamba `full` vs P2-07 from-scratch-2-epoch vs P4 zero-shot, on teacher-forced MAPE. Write the three numbers and the noise floor here | todo | | P5-03 | numbers in this log; M3 → `awaiting_review` | |

## P6 — Long-term 7-day variants (HANDOFF §5 step 6)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P6-01 | Synthetic 7-day matched run: `configs/gmat/gmat_28089_long.yaml` (`numMinsToProp: 10080`, `num_runs: 3`); **Hunter-run** GMAT. Config is committed | todo | | R2-03 | `data/regression/geo/10080min-3` exists | |
| P6-02 | Branch (a) and (c) on the 7-day real window (train 1 day, test 6), Mamba + LSTM, 3 seeds; `style=long` rows | todo | | P6-01, P5-03, P2-06 | rows in `results/branch_{a,c}.csv` with `style=long` | |
| P6-03 | Phase note; M4 rows complete | todo | | P6-02 | | |

## P7 — Summary (HANDOFF §5 step 7)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| P7-01 | `scripts/summarize.py`: read `results/branch_*.csv`; per (branch, model, tl_mode, style) median ± IQR over seeds; noise-floor row; write `results/summary.md` in the ASC Table 1–3 layout (columns x, y, z, ẋ, ẏ, ż MAPE) plus energy column | blocked | claude | P6-03 | table renders; every row has ≥ 3 seeds or is flagged |2026-09-17 claude: scripts/summarize.py written and exercised; needs result rows -- blocked_by H-2 (window) / P1-06 |
| P7-02 | `figures/Summary-MAPE.pdf`: grouped bars a/b/c × Mamba/LSTM, error bars = IQR, dashed line = noise floor; title carries "truth = SGP4 from space-track TLEs" | todo | | P7-01 | figures exist; read the `dataviz` skill before plotting | |
| P7-03 | One-page `results/README.md`: what was claimed (ASC, DDDAS), what was measured, the three headline numbers, the F1 correction and its effect (link `results/asc2025_corrected.md`) | todo | | P7-01, R3-03 | ≤ 1 page; no number without seed count and noise floor | |
| P7-04 | Final phase note; M4 → `awaiting_review` | todo | | P7-03 | | |

## Follow-ups (append new IDs here if they do not belong to a phase above)

| ID | Task | Status | Owner | Depends on | DoD | Log |
|---|---|---|---|---|---|---|
| F-01 | *(H-3)* ISS 25544 contamination case: run branches a/b/c on a window containing a known reboost; expected: all three fail at the burn; one figure | blocked | | H-3, P5-03 | | |
| F-02 | *(H-1)* Secondary lower-LEO passive object: satcat query `OBJECT_TYPE=ROCKET BODY, DECAY=null, PERIGEE 450--550`; propose 3 candidates with TLE count over the window; Hunter picks | blocked | | H-1, P1-01 | candidates logged here | |
| F-03 | After M3 green: move `teme_to_j2000`, `segment_ephemeris` into `qutils/orbital.py` (HANDOFF §8.2 item 3) | todo | | P5-05 | qutils commit; `tlv/tle_ephem.py` imports from qutils; tests still green | |
| F-04 | Annotate the docs with the 2026-09-20 plan review; distinguish findings and recommendations from approved design changes | done | codex | — | All review findings have stable IDs, evidence, and links from affected sections; follow-ups remain open | 2026-09-20 codex: claimed; documentation only<br>2026-09-20 codex: added REV-01–REV-10 and F-05–F-14; git diff --check passed; review IDs, follow-up rows, code fences, and local file links validated |

## Review follow-ups (2026-09-20)

These entries record recommendations from [HANDOFF §10](HANDOFF.md#10-plan-review-2026-09-20).
`todo` means unresolved review work, not authorization to adopt every proposed design change.
Reconcile the protocol in HANDOFF before implementing a changed experimental condition.

| ID | Review / affected tasks | Proposed work | Status | Owner | Closure evidence |
|---|---|---|---|---|---|
| F-05 | REV-01; P1-07, P2-03, M0 | Correct reference-data and uncertainty claims; track hand-off count/TLE age; distinguish absent from zero-valued hand-offs | in_progress | claude | Consistent terminology, justified quality criteria, and validated missing-evidence handling |
| F-06 | REV-02; P2-02/P2-03, P5-05, P6 | Define common forecast origin and elapsed horizons; fix coverage and endpoint handling | in_progress | claude | Timestamp tests show 1T is a full period after the final observation, with all requested endpoints present |
| F-07 | REV-03; R2-01–R2-03, P3-01 | Produce exact initial state/shared epoch and correct anomaly/force metadata before accepting matched GMAT data | todo | | Run-zero state/time alignment verified and discrepancy components documented |
| F-08 | REV-04; P2-06/P2-07, P5 | Specify observed states versus transitions, remove training-target overlap, and declare target information used for source construction | in_progress | claude | Auditable sample budget and tests that scoring starts strictly after the final observed state |
| F-09 | REV-05; P2-05, P4-02, P5-03, P7 | Preselect development/held-out windows and paired seed/window comparisons | todo | | Frozen window protocol and separate reporting of seed/window variability and overlap |
| F-10 | REV-06; P2-03, P7 | Add physical/simple baselines and prioritize position/velocity errors over MAPE and energy diagnostics | todo | | Baselines share forecast origins and available information; primary metrics are specified before the matrix |
| F-11 | REV-07; M2/M3, PLAN §1–§2 | Replace success-conditioned gates with validity gates and a neutral research question | todo | | Protocol permits complete reporting of poor fits and negative transfer |
| F-12 | REV-08; P2-01, HANDOFF §4 | Clarify length-one state-map interpretation and evaluate a feed-forward comparator | todo | | Architecture claims match the actual context/state policy; comparator budget documented |
| F-13 | REV-09; P4-03, P7-01/P7-02 | Add run provenance, unique artifacts, compatible grouping, and a deterministic rerun policy | in_progress | claude | Object/window/config identities survive to summary; duplicate runs cannot silently reweight results |
| F-14 | REV-10; P0, P1-02/P1-08, P2-07, P6, HANDOFF §9.6 | Reconcile GEO instructions, actual coverage, CLI examples, task status/table formatting, and proposed dependency changes | in_progress | claude | Active commands/requirements agree; code-written and data-validated states are distinct |

---

## Log format for agents

Append to the task's Log cell, newest last, pipe-safe (use `<br>` for line breaks):

```
2026-09-18 agentX: claimed<br>2026-09-18 agentX: pytest 3 passed<br>2026-09-18 agentX: done, commit ab12cd3
```

**2026-09-21 (claude), F-05/F-06/F-08/F-13/F-14 partial closure.** What is implemented and
tested is listed in [HANDOFF §10](HANDOFF.md#10-plan-review-2026-09-20) under *Implementation
status*. In short: forecast origin and horizons fixed and guarded by a test, training-target
overlap removed, hand-off statistic reports NaN and a count instead of a silent 0.0, rows
carry window/origin/TLE identity, figures carry seed and transfer mode, `summarize.py` no
longer lets a rerun reweight a median, and the CLI examples match the code. Still open in
each: paired cross-branch origins, the observed-state budget in the CSV, TLE age, GEO
threshold justification, a run manifest, and dependency/checkpoint identity.

