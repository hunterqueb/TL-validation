# tl-validation

Real-world validation of Mamba transfer learning (ASC 2025, AAS 25-543) and low-data
training (DDDAS 2024) on real LEO ephemerides derived from space-track TLEs.

Spec: [docs/HANDOFF.md](docs/HANDOFF.md) · order of work: [docs/PLAN.md](docs/PLAN.md) ·
state: [docs/TASKS.md](docs/TASKS.md).

Three branches share one target object and window:

| branch | source | target | what it shows |
|---|---|---|---|
| a | GMAT synthetic, 1 period | real, no training | zero-shot gap (the ASC procedure as it actually runs, F1) |
| b | — | real, 1 period (ASC) or 2 points (DDDAS) | from scratch on very little real data |
| c | GMAT synthetic | real, 1 period, 2-epoch fine-tune | the ASC claim, on real data |

Truth is SGP4-quality (km-level). Every error is reported next to the TLE hand-off noise
floor, and every teacher-forced number next to its closed-loop rollout number.

## Install

```bash
pip install -e ../qutils          # skbuild: if the editable install fails, `pip install ../qutils`
pip install -r requirements.txt
ln -s ~/git/GMAT-Thrust-Data/data data/synthetic
export SPACETRACK_USER=... SPACETRACK_PASS=...
pytest tests/                     # 13 pass, hand-off gate skips until a dataset exists
```

`qutils` is installed as a copy, not in editable mode: **re-run `pip install ../qutils`
after any qutils edit**, or the fix will not be picked up.

## Run

```bash
# real data
python scripts/fetch_tles.py --norad 27386 --start 2026-06-01 --stop 2026-07-15
python scripts/build_real_dataset.py --norad 27386 --minutes 505   --windows 40
python scripts/build_real_dataset.py --norad 27386 --minutes 10080 --windows 4

# branch (b): from scratch on real data
python scripts/branch_b_real_lowdata.py --norad 27386 --minutes 505 --windows 40 \
    --style asc --model mamba --seed 0
python scripts/branch_b_real_lowdata.py --norad 27386 --minutes 10080 --windows 4 \
    --style dddas --model mamba --seed 0
python scripts/branch_b_real_lowdata.py --norad 27386 --minutes 505 --windows 40 \
    --style asc2 --model mamba --seed 0     # equal-budget row for the M3 comparison

# matched synthetic source (GMAT runs on Hunter's machine)
python scripts/make_gmat_config.py --norad 27386
cp configs/gmat_27386.yaml ~/git/GMAT-Thrust-Data/config.yaml
( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )

# branches (a) and (c)
python scripts/branch_a_zero_shot.py --norad 27386 --minutes 505 --windows 40 --model mamba --seed 0
python scripts/branch_c_transfer.py  --norad 27386 --minutes 505 --windows 40 --model mamba --tl full --seed 0

python scripts/summarize.py
```

Branch (a) saves its source weights to `results/models/`; branch (c) loads them, so the two
branches share one source model. Run (a) first.

## Layout

`tlv/` is six modules and stays six (PLAN.md rule 9): `spacetrack`, `tle_ephem`, `datasets`,
`models`, `train`, `metrics`. Shared script plumbing lives in `scripts/_common.py`.
Results are append-only CSV rows in `results/`; figures are regenerated, never hand-edited.
