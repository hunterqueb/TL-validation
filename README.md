# tl-validation

Real-world validation of Mamba transfer learning (ASC 2025, AAS 25-543) and low-data
training (DDDAS 2024) on real ephemerides derived from space-track TLEs.

Primary target: **Yamal 202** (NORAD 28089), a GEO comsat — period 1436.09 min, so the
6-period window is 8620 samples: one training period plus five test periods. Window
**2026-02-18 .. 2026-04-08**, screened for maneuvers and TLE cadence by
`scripts/screen_window.py` before any model was run. The original plan targeted a LEO
object; see the Decision Log in [docs/PLAN.md](docs/PLAN.md) for what that change costs.

Spec: [docs/HANDOFF.md](docs/HANDOFF.md) · order of work: [docs/PLAN.md](docs/PLAN.md) ·
state: [docs/TASKS.md](docs/TASKS.md).

Three branches share one target object and window:

| branch | source | target | what it shows |
|---|---|---|---|
| a | GMAT synthetic, 1 period | real, no training | zero-shot gap (the ASC procedure as it actually runs, F1) |
| b | — | real, 1 period (ASC) or 2 points (DDDAS) | from scratch on very little real data |
| c | GMAT synthetic | real, 1 period, 2-epoch fine-tune | the ASC claim, on real data |

Truth is SGP4-quality (km-level). Every error is reported next to the TLE hand-off
consistency statistic — successive TLEs disagree by ~0.6 km here, and that is a consistency
figure, not a calibrated accuracy floor.

## Install

```bash
pip install -e ../qutils          # skbuild: if the editable install fails, `pip install ../qutils`
pip install -r requirements.txt
$EDITOR data.yaml                 # point `gmat` at your GMAT-Thrust-Data/data folder
cp .env.example .env              # then fill in your space-track credentials
pytest tests/                     # 18 pass (the hand-off gate needs a built dataset)
```

`qutils` is installed as a copy, not in editable mode: **re-run `pip install ../qutils`
after any qutils edit**, or the fix will not be picked up.

## Run

Every branch has a run script that executes its whole matrix and writes the standard plot
set (HANDOFF §6.1) for each run. Figure names carry style, model and seed, so nothing
overwrites anything. Run them from the repo root.

### 1. Real data (once)

```bash
python scripts/screen_window.py --norad 28089 --start 2026-01-01 --stop 2026-09-21
python scripts/fetch_tles.py --norad 28089 --start 2026-02-18 --stop 2026-04-08
python scripts/build_real_dataset.py --norad 28089 --minutes 8620  --windows 8
python scripts/build_real_dataset.py --norad 28089 --minutes 10080 --windows 4
pytest tests/
```

Secondary target, BLOCK DM R/B (NORAD 26769, HEO, period 771.3 min), branch (b) only:

```bash
python scripts/screen_window.py --norad 26769 --start 2026-01-01 --stop 2026-10-19 \
    --minutes 4630 --windows 8 --max-step 1.0
python scripts/build_real_dataset.py --norad 26769 --minutes 4630  --windows 8
python scripts/build_real_dataset.py --norad 26769 --minutes 10080 --windows 3
NORAD=26769 ./scripts/run_branch_b.sh
```

Once the windows are in `configs/targets.yaml`, `./scripts/build_datasets.sh` runs all four
`build_real_dataset.py` commands above, for both objects.

`screen_window.py` ranks candidate spans by TLE coverage, largest TLE gap and |d(sma)| step
before any model is run; the chosen window and its criteria are recorded in
`configs/targets.yaml`.

### 2. Matched synthetic source (GMAT, run on Hunter's machine)

Branches (a) and (c) need it; branch (b) does not.

```bash
cp configs/gmat/gmat_28089_asc.yaml ~/git/GMAT-Thrust-Data/config.yaml
( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )   # -> regression/geo/8620min-10
cp configs/gmat/gmat_28089_long.yaml ~/git/GMAT-Thrust-Data/config.yaml
( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )   # -> regression/geo/10080min-3
```

### 3. Branch matrices

```bash
./scripts/run_branch_b.sh      # M1 + M1b: {asc, asc2, dddas} x {mamba, lstm} x 3 seeds = 18 runs
./scripts/run_branch_a.sh      # M2:       {mamba, lstm} x 3 seeds = 6 runs
./scripts/run_branch_c.sh      # M3:       {full, feature} x {mamba, lstm} x 3 seeds = 12 runs
python scripts/summarize.py    # results/branch_*.csv -> results/summary.md + results/branch_b_cost.tex
```

| script | styles / modes | needs |
|---|---|---|
| [run_branch_b.sh](scripts/run_branch_b.sh) | `asc` (5 epochs), `asc2` (2 epochs, the equal-budget row for M3), `dddas` (2 points, 50 epochs, 7-day window) | the real dataset |
| [run_branch_a.sh](scripts/run_branch_a.sh) | `asc`, `long` | the real dataset **and** the GMAT source |
| [run_branch_c.sh](scripts/run_branch_c.sh) | `full`, `feature` × `asc`, `long` | both, **and** branch (a) run first |

**Order matters.** Branch (a) saves its source weights to `results/models/`; branch (c)
loads them, so the two share one source model. Run (a) before (c) or the comparison is not
a transfer-learning comparison — (c) warns per-cell when those weights are missing. PLAN's
hard rule is that (c) does not start until M1 and M2 are green.

Narrow a re-run with arguments or the environment — settings live in
[scripts/_common.sh](scripts/_common.sh) and are all overridable:

```bash
./scripts/run_branch_b.sh asc asc2               # only these styles
./scripts/run_branch_c.sh full                   # only this transfer mode
./scripts/run_branch_a.sh long                   # the 7-day window instead
SEEDS=0 MODELS=mamba ./scripts/run_branch_b.sh dddas
WINDOW=3 ./scripts/run_branch_b.sh asc           # a different window of the set
SUMMARIZE=0 ./scripts/run_branch_b.sh            # skip the summary at the end
NORAD=26769 ./scripts/run_branch_b.sh            # the Block DM R/B (HEO) secondary target
```

| variable | default | |
|---|---|---|
| `NORAD` | `28089` | target object (`26769`: Block DM R/B, HEO) |
| `ASC_MIN` / `ASC_WIN` | `8620` / `8` (26769: `4630` / `8`) | ASC-style window: 6 periods, 8 of them |
| `LONG_MIN` / `LONG_WIN` | `10080` / `4` (26769: `10080` / `3`) | 7-day window |
| `WINDOW` | `0` | which window of the set to use |
| `SEEDS` | `0 1 2` | three is the floor for any headline number |
| `MODELS` | `mamba lstm` | |
| `SUMMARIZE` | `1` | run `summarize.py` at the end |

Each script checks its inputs first and stops with the command that produces a missing one.
A run that fails does not abort the matrix: it is counted, listed at the end, and the script
exits 1.

### 4. A single run

```bash
python scripts/branch_b_real_lowdata.py --norad 28089 --minutes 8620 --windows 8 \
    --style asc --model mamba --seed 0
python scripts/branch_a_zero_shot.py --norad 28089 --minutes 8620 --windows 8 \
    --model mamba --seed 0
python scripts/branch_c_transfer.py  --norad 28089 --minutes 8620 --windows 8 \
    --model mamba --tl full --seed 0
```

`--windows` is required: together with `--minutes` it selects the dataset directory.
Add `--no-plots` to skip the plot set.

## Configs

One file per example run, all of them committed:

| file | drives |
|---|---|
| [.env](.env.example) | space-track credentials (copy `.env.example`; `.env` is gitignored) |
| [data.yaml](data.yaml) | where the GMAT data folder is |
| [configs/targets.yaml](configs/targets.yaml) | object, window, `fetch_tles` + `build_real_dataset` |
| [configs/branch_a.yaml](configs/branch_a.yaml) | zero-shot: synthetic source, no target training |
| [configs/branch_b.yaml](configs/branch_b.yaml) | from scratch on real data, styles `asc` / `asc2` / `dddas` |
| [configs/branch_c.yaml](configs/branch_c.yaml) | transfer: 2-epoch fine-tune, modes `full` / `feature` |
| [configs/gmat/gmat_28089_asc.yaml](configs/gmat/gmat_28089_asc.yaml) | the GMAT run that makes `regression/geo/8620min-10` |
| [configs/gmat/gmat_28089_long.yaml](configs/gmat/gmat_28089_long.yaml) | the GMAT run that makes `regression/geo/10080min-3` |

The two `configs/gmat/` files are in GMAT-Thrust-Data's own schema (see its `config.yaml`),
so they are copied to that repo's `config.yaml` and run there. `scripts/make_gmat_config.py`
regenerates them from a real window and emits the same schema.

## Layout

`tlv/` is six modules and stays six (PLAN.md rule 9): `spacetrack`, `tle_ephem`, `datasets`,
`models`, `train`, `metrics`. Shared plumbing lives in `scripts/_common.py` (window
loading, the plot set, result rows) and `scripts/_common.sh` (run-matrix settings).
Results are append-only CSV rows in `results/`; figures are regenerated, never hand-edited.

GMAT data stays where it is: [data.yaml](data.yaml) holds one absolute path (`gmat`) to the
GMAT-Thrust-Data `data/` folder, and `tlv.datasets.load_synthetic` addresses
`<gmat>/<dataset>/<orbit>/<T>min-<N>/` from it. Nothing is symlinked or copied into this repo.
Edit that path per machine; it is committed, like the same file in `thrust-classification`.
