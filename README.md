# tl-validation

Real-world validation of Mamba transfer learning (ASC 2025, AAS 25-543) and low-data
training (DDDAS 2024) on real ephemerides derived from space-track TLEs.

Primary target: **Yamal 202** (NORAD 28089), a GEO comsat — period 1436.09 min, so the
6-period window is 8620 samples: one training period plus the full five-period forecast
horizon. Window **2026-02-18 .. 2026-04-08**, screened for maneuvers and TLE cadence by
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
figure, not a calibrated accuracy floor — and every teacher-forced number next to its
closed-loop rollout number. Rollout horizons are measured from the forecast origin, the last
observed state.

## Install

```bash
pip install -e ../qutils          # skbuild: if the editable install fails, `pip install ../qutils`
pip install -r requirements.txt
$EDITOR data.yaml                 # point `gmat` at your GMAT-Thrust-Data/data folder
cp .env.example .env              # then fill in your space-track credentials
pytest tests/                     # 13 pass, hand-off gate skips until a dataset exists
```

`qutils` is installed as a copy, not in editable mode: **re-run `pip install ../qutils`
after any qutils edit**, or the fix will not be picked up.

## Run

```bash
# real data
python scripts/screen_window.py --norad 28089 --start 2026-01-01 --stop 2026-09-21
python scripts/fetch_tles.py --norad 28089 --start 2026-02-18 --stop 2026-04-08
python scripts/build_real_dataset.py --norad 28089 --minutes 8620  --windows 8
python scripts/build_real_dataset.py --norad 28089 --minutes 10080 --windows 4

# branch (b): from scratch on real data
python scripts/branch_b_real_lowdata.py --norad 28089 --minutes 8620 --windows 8 \
    --style asc --model mamba --seed 0
python scripts/branch_b_real_lowdata.py --norad 28089 --minutes 10080 --windows 4 \
    --style dddas --model mamba --seed 0
python scripts/branch_b_real_lowdata.py --norad 28089 --minutes 8620 --windows 8 \
    --style asc2 --model mamba --seed 0     # equal-budget row for the M3 comparison

# matched synthetic source (GMAT runs on Hunter's machine)
# configs/gmat/ already holds both runs; regenerate from a real window with
#   python scripts/make_gmat_config.py --norad 28089 --minutes 8620 --windows 8
cp configs/gmat/gmat_28089_asc.yaml ~/git/GMAT-Thrust-Data/config.yaml
( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )   # -> regression/geo/8620min-10
cp configs/gmat/gmat_28089_long.yaml ~/git/GMAT-Thrust-Data/config.yaml
( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )   # -> regression/geo/10080min-3

# branches (a) and (c)
python scripts/branch_a_zero_shot.py --norad 28089 --minutes 8620 --windows 8 --model mamba --seed 0
python scripts/branch_c_transfer.py  --norad 28089 --minutes 8620 --windows 8 --model mamba --tl full --seed 0

python scripts/summarize.py
```

Branch (a) saves its source weights to `results/models/`; branch (c) loads them, so the two
branches share one source model. Run (a) first.

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
`models`, `train`, `metrics`. Shared script plumbing lives in `scripts/_common.py`.
Results are append-only CSV rows in `results/`; figures are regenerated, never hand-edited.

GMAT data stays where it is: [data.yaml](data.yaml) holds one absolute path (`gmat`) to the
GMAT-Thrust-Data `data/` folder, and `tlv.datasets.load_synthetic` addresses
`<gmat>/<dataset>/<orbit>/<T>min-<N>/` from it. Nothing is symlinked or copied into this repo.
Edit that path per machine; it is committed, like the same file in `thrust-classification`.
