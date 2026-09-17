# tl-validation — Implementation Handoff

Real-world validation of Mamba transfer learning (ASC 2025, AAS 25-543) and low-data
training (DDDAS 2024) on real satellite ephemerides derived from space-track TLEs.

Written 2026-09-16 from an audit of: `hunterqueb/pendulumRNN` (`scripts/conferences/ASC2025`,
`scripts/conferences/dddas`, `scripts/journals/orbitProp/p2bp`), `hunterqueb/qutils` (v0.11),
and `~/git/GMAT-Thrust-Data` (local). Every file, function and array name below was checked
against those sources. Anything not checked is marked **VERIFY**.

Design rules for this repo (from the project instructions): simple beats complex; surgical
changes only. Reuse `qutils` functions. Do not fork them into this repo unless a bug blocks you.

---

## 0. Decisions already made

| Decision | Value |
|---|---|
| Primary task | State-vector regression (6-state ECI), as in ASC 2025 / DDDAS. Classification is out of scope for the first pass. |
| Real-data representation | SGP4-densified TLE segments: each TLE propagated only inside its own validity window, sampled at 60 s, TEME → J2000. Truth quality is SGP4-quality (km-level). Every plot must say so. |
| Regime | LEO, passive (non-maneuvering) objects. Synthetic source domain comes from GMAT-Thrust-Data LEO runs. |
| Historical TLE source | space-track.org `gp_history`, credentials in `SPACETRACK_USER` / `SPACETRACK_PASS`. CelesTrak is a fallback for current-epoch sanity checks only. |
| Sampling | `dt = 60 s`, identical to GMAT-Thrust-Data. |
| Frames / units | km, km/s, J2000 equatorial (GMAT `EarthMJ2000Eq`). Non-dimensionalize with `qutils.orbital.dim2NonDim6` before training (see finding F4). |

---

## 1. Audit findings — read before writing code

These change what the demo can honestly claim. They are ordered by impact.

**F1. The ASC 2025 script never fine-tunes the target model.**
`pendulumRNN/scripts/conferences/ASC2025/mambaTLCR3BP.py`, target phase:

```python
newModel = transferMamba(model, newModel, [True, True, False])
optimizer = torch.optim.Adam(newModel.parameters(), lr=lr)
trainModel(model, n_epochs, [...], criterion, optimizer, ...)      # trains `model`, steps `newModel`
networkPrediction_target = plotStatePredictions(model, ...)         # evaluates `model`
```

`loss.backward()` puts gradients on `model`; `optimizer.step()` updates `newModel`, whose
gradients are `None`. Neither model changes. The "target domain" numbers in Tables 1–3 of the
paper are therefore **zero-shot** evaluations of the source-trained model on the target orbit.
Consequences for this demo: branch (a) *is* the ASC procedure as actually run; branch (c) must use
the corrected loop (`trainModel(newModel, ...)`, `plotStatePredictions(newModel, ...)`). The same
correction must be applied to `pendulumRNN` and the three ASC orbit pairs re-run so the CR3BP
baseline numbers are known before real-data numbers are compared to them.

**F2. `transferMamba` is feature-extraction TL, not full fine-tuning.**
`qutils/ml/regression.py::transferMamba` freezes every parameter, then unfreezes only
`conv1d`, `x_proj`, `dt_proj` and (optionally) `out_proj`. `in_proj` and `A_log` stay frozen.
The paper text says "all parameters still trainable". Use two explicit modes in this repo:
`full` → `transferModelAll`; `feature` → `transferMamba(..., [True, True, True])`. Report both.

**F3. "Long-term propagation" in past work is teacher-forced one-step prediction.**
`qutils.ml.regression.genPlotPrediction` computes `model(test_in)`, where `test_in` is the
*true* previous state at every step. There is no closed-loop rollout. This demo keeps the
teacher-forced metric (for comparability with the papers) and adds a closed-loop rollout metric
(`tlv/train.py::rollout`). Both are reported on every plot set. Expect rollout error to be much
larger; that is the honest number for "propagation".

**F4. `qutils.orbital` non-dimensionalization does not round-trip.**
`dim2NonDim6` default `TU = sqrt(6378.1**3 / 3.968e5)` = 808.63 s (μ typo: 396800).
`nonDim2Dim6` default `TU = 806.80415` s (μ = 398600). Round trip scales velocities by 1.00227
(0.23 %). `orbitalEnergy(Y, mu=396800)` carries the same typo. Errors *relative* to truth in
past plots are unaffected (truth and prediction are scaled identically); absolute velocity axes
and energy values are off by 0.23 % / 0.45 %. Fix in qutils (section 8.2) and always pass
`DU`, `TU` explicitly in this repo.

**F5. GMAT per-run parameters are not saved.**
`GMAT-Thrust-Data/python/generateSpacecraftThrustOpt.py` draws `SMA, ECC, INC, RAAN, AOP, TA`,
`SRPArea, Cr, DragArea, Cd, DryMass` per run with `rng = default_rng(123)` and saves only
`statesArray*.npz` `(N, T, 6)`. Epoch is fixed at `20 Jul 2020 12:00:00.000`. Sample `j` is the
state after the `(j+1)`-th 60 s step, so `t_j = (j+1)·60 s`; the initial condition is not in the
array. To build a source domain that matches a real object you need fixed elements, a matching
epoch, a matching ballistic coefficient and the metadata in the file. See section 8.1.

**F6. `convertToTLE.py` does not make TLE mean elements.**
`ECI2TLE` returns osculating Keplerian elements relabelled as TLE fields, with `Bstar = 0`. Its
own docstring asks "is this right?". Do not compare `TLEArray*` to real TLE lines. All real-vs-
synthetic comparison in this demo is in Cartesian J2000.

**F7. Inclination sampling.** `INC[i] = config['inclination'] * rng.random()` with
`inclination: 180` gives uniform `[0, 180)` deg, i.e. half retrograde. The existing LEO
regression sets (`data/regression/leo/*`) are therefore not "LEO-like" in inclination. For the
matched source domain use fixed elements (section 8.1); for the multi-trajectory source use
nearest-neighbour selection on elements (section 3.3).

**F8. Real LEO drag is not in the synthetic model's training distribution the way it is in
reality.** GMAT runs use Jacchia-Roberts with default space-weather. Real 2025–2026 TLEs
reflect actual F10.7/Ap. This is the intended out-of-distribution gap for branch (a); it must
be named as such in the results, not hidden.

---

## 2. Dataset contract

All arrays that enter a model follow the GMAT-Thrust-Data convention so that one loader serves
synthetic and real data.

```
key         : statesArray<Tag>            Tag ∈ {NoThrust, Chemical, Electric, ImpBurn, Real}
shape       : (N, T, 6)  float64          [x, y, z, vx, vy, vz]  km, km/s, J2000 / EarthMJ2000Eq
dt          : 60.0 s
time        : synthetic  t_j = (j+1)*dt   (GMAT, F5)
              real       t_j = j*dt       from segment start; explicit `t_sec` key
directory   : data/<dataset>/<orbit>/<T>min-<N>/          e.g. data/regression/leo/450min-10000
```

Real-data file (written by `tlv/tle_ephem.py`), one object per file:

```
data/real/<norad>/<T>min-<N>/statesArrayReal.npz
    statesArrayReal   (N, T, 6)   N contiguous windows of T minutes
    t_sec             (N, T)      seconds from window start
    epoch_jd          (N,)        UTC Julian date of window start
    tle_index         (N, T)      index into the TLE list used at each sample
    handoff_jump_km   (N, T)      |r_new - r_old| at TLE hand-off samples, 0 elsewhere
    oe                (N, T, 7)   qutils.orbital.ECI2OE output [a, e, i, Ω, ω, M0, P]
    meta.json         norad, name, tle_count, first/last epoch, sgp4 version, frame chain
```

`handoff_jump_km` is the noise floor of the "truth". It is reported next to every error metric.

---

## 3. Modules (item 1 of the request)

Package name `tlv`. Six modules. Nothing else until a milestone needs it.

```
tl-validation/
├── HANDOFF.md                  this file
├── README.md                   one-screen usage
├── requirements.txt            numpy scipy matplotlib torch torchinfo pyyaml requests sgp4 astropy
│                               qutils (pip install -e ../qutils)
├── configs/
│   ├── targets.yaml            NORAD ids, windows, dt
│   ├── branch_a.yaml  branch_b.yaml  branch_c.yaml
├── tlv/
│   ├── __init__.py
│   ├── spacetrack.py           TLE history download + on-disk cache
│   ├── tle_ephem.py            TLE list -> SGP4 segments -> J2000 -> statesArrayReal.npz
│   ├── datasets.py             load synthetic/real npz, non-dim, windows, nearest-neighbour source pick
│   ├── models.py               returnModel(), transfer(mode)
│   ├── train.py                fit(), predict_teacher_forced(), rollout()
│   └── metrics.py              mape, rmse, energy drift, noise floor, results CSV
├── scripts/
│   ├── fetch_tles.py
│   ├── build_real_dataset.py
│   ├── branch_a_zero_shot.py
│   ├── branch_b_real_lowdata.py
│   ├── branch_c_transfer.py
│   └── summarize.py            results/*.csv -> comparison table + bar chart
├── tests/
│   ├── test_tle_ephem.py
│   └── test_datasets.py
├── data/      (gitignored)   tle/  real/  synthetic/ (symlink to ~/git/GMAT-Thrust-Data/data)
├── results/   (gitignored)   csv per run
└── figures/                  pdf/png, same naming scheme as ASC2025 (`<problem>-<Domain><Plot>.pdf`)
```

### 3.1 `tlv/spacetrack.py`

```python
"""Download TLE history from space-track.org and cache it on disk."""
import os
import pathlib
import requests

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
QUERY_URL = ("https://www.space-track.org/basicspacedata/query/class/gp_history/"
             "NORAD_CAT_ID/{norad}/EPOCH/{start}--{stop}/orderby/EPOCH asc/format/tle")


def fetch_tle_history(norad, start, stop, cache_dir="data/tle"):
    """Return the TLE text for one object in [start, stop].

    Read the cache first. Query space-track only when the cache file is absent.
    Dates are ISO strings, for example "2026-01-01".
    """
    path = pathlib.Path(cache_dir) / f"{norad}_{start}_{stop}.tle"
    if path.exists():
        return path.read_text()
    session = requests.Session()
    session.post(LOGIN_URL, data={"identity": os.environ["SPACETRACK_USER"],
                                  "password": os.environ["SPACETRACK_PASS"]})
    text = session.get(QUERY_URL.format(norad=norad, start=start, stop=stop)).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
```

Rules: rate limit is 30 requests/min and 300/hour; one query per object per window; never
loop over days. Cache file is the unit of reproducibility — commit its sha256 in `meta.json`.

### 3.2 `tlv/tle_ephem.py`

Core algorithm. This is the only place where "real" data is created.

```python
"""Convert a TLE list to a 60 s J2000 ephemeris with the GMAT array layout."""
import numpy as np
from sgp4.api import Satrec, jday
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
from qutils.orbital import ECI2OE

DT = 60.0


def parse_tles(text):
    """Return a list of (epoch_jd, Satrec) sorted by epoch, duplicates removed."""


def segment_ephemeris(sats, dt=DT):
    """Propagate each TLE from its epoch to the next TLE epoch.

    Return r (M, 3) km, v (M, 3) km/s in TEME, t_jd (M,), tle_index (M,), jump_km (M,).
    jump_km[k] is |r_new - r_old| at the first sample of a new TLE, else 0.
    """


def teme_to_j2000(r_teme, v_teme, t_jd):
    """Rotate TEME to GCRS with astropy. GCRS-to-J2000 frame bias (~23 mas) is ignored."""
    t = Time(t_jd, format="jd", scale="utc")
    teme = TEME(CartesianRepresentation(r_teme.T * u.km,
                differentials=CartesianDifferential(v_teme.T * u.km / u.s)), obstime=t)
    gcrs = teme.transform_to(GCRS(obstime=t))
    return gcrs.cartesian.xyz.to_value(u.km).T, gcrs.velocity.d_xyz.to_value(u.km / u.s).T


def to_windows(states, t_jd, tle_index, jump_km, minutes):
    """Cut a continuous ephemeris into N windows of T = minutes samples. Drop the remainder."""


def write_real_npz(path, windows):
    """Write statesArrayReal.npz with the keys listed in HANDOFF.md section 2."""
```

Validation tests (`tests/test_tle_ephem.py`):

- SGP4 self-check: propagate the sgp4 package's reference TLE and compare to the values in the
  package's own test file (`sgp4/tests`), tolerance 1e-6 km.
- Frame check: `|r|` is invariant under `teme_to_j2000` to 1e-9 relative. `|v|` and the
  semi-major axis are **not**: TEME rotates with respect to GCRS at the precession rate
  (~7.7e-12 rad/s), so the transform correctly adds an ω × r term of ~5e-8 km/s, i.e. 6e-9
  relative on velocity and twice that on `a` through v². Tolerance is 1e-7 for those two.
  (Corrected 2026-09-17; the 1e-9 velocity tolerance written at audit time is unattainable.)
- Hand-off statistics: for the chosen object, median `handoff_jump_km` < 2 km and 95th
  percentile < 10 km over the window. If not, the object or the window is unsuitable.

### 3.3 `tlv/datasets.py`

```python
def load_synthetic(root, dataset, orbit, minutes, n_sys, tag="NoThrust"):
    """Load statesArray<tag>.npz from root/data/<dataset>/<orbit>/<minutes>min-<n_sys>."""


def load_real(norad, minutes, n_windows):
    """Load statesArrayReal.npz and its side keys."""


def nondim(states, DU=6378.1, TU=806.80455):
    """Apply qutils.orbital.dim2NonDim6 with explicit DU, TU (see F4). Copy first."""


def split_periods(seq, period_sec, train_periods, dt=60.0):
    """Return train_size for 'train on train_periods, test on the rest' (ASC style)."""


def nearest_synthetic(oe_target, oe_bank, k=1):
    """Pick k synthetic trajectories whose initial [a, e, i] are nearest to oe_target.

    Distance: normalized by (100 km, 0.01, 5 deg). Used only in branch (c)-multi.
    """
```

`create_datasets` comes from `qutils.ml.regression` unchanged (`seq_length = 1`).

### 3.4 `tlv/models.py`

```python
from qutils.ml.mamba import Mamba, MambaConfig
from qutils.ml.regression import LSTM, transferMamba, transferModelAll


def return_model(kind, device, d_model=6, n_layers=1, d_conv=4, lstm_hidden=30):
    """Build Mamba or LSTM exactly as in the ASC 2025 script (d_conv=16 for DDDAS-style)."""


def transfer(source, target, mode):
    """mode 'full' -> transferModelAll; 'feature' -> transferMamba(..., [True, True, True])."""
```

### 3.5 `tlv/train.py`

```python
from qutils.ml.regression import trainModel, create_datasets
from qutils.plot import plotStatePredictions


def fit(model, seq, train_size, epochs, lr, criterion, device):
    """Wrap qutils trainModel. Return (model, seconds). Optimizer: torch.optim.Adam."""


def predict_teacher_forced(model, t, seq, train_in, test_in, train_size, test_size, **plot_kw):
    """Return the (T, 6) prediction array from qutils.plot.plotStatePredictions."""


def rollout(model, seq, train_size, device):
    """Closed-loop prediction. Feed the model its own output from index train_size onward.

    Return (T, 6) with NaN for j < train_size. This is the honest propagation metric (F3).
    """
```

### 3.6 `tlv/metrics.py`

`mape`, `rmse` → import from `qutils.ml.utils`. Add:

- `energy_drift(seq_dim, mu, Re)` using `qutils.orbital.orbitalEnergyJ2` (two-body energy
  oscillates ~0.1 % under J2; the J2-corrected energy is the right conserved-ish quantity
  for LEO).
- `position_error_at(pred, truth, t, periods)` → km at 1, 2, 5 periods.
- `noise_floor(handoff_jump_km)` → median, p95.
- `append_result(csv_path, row)` with fixed columns:
  `branch, object, model, tl_mode, train_periods, epochs_src, epochs_tgt, mape_x..mape_vz,
  rmse_pos_km, rollout_pos_1T_km, rollout_pos_5T_km, energy_drift_pct, noise_floor_med_km,
  noise_floor_p95_km, train_s, seed`.

---

## 4. Branches (item 2)

All three branches share one target object and one target window so the numbers are comparable.

| Branch | Source (train) | Target (test) | Fine-tune | Skeleton | What it demonstrates |
|---|---|---|---|---|---|
| (a) zero-shot OOD | GMAT synthetic, matched elements, 5 periods, train on 1 period | Real, 5 periods, all 5 tested | none | `mambaTLCR3BP.py` as it actually runs (F1) | How far the synthetic model is from reality. Baseline that (c) must beat. |
| (b) real low-data | Real, first 1 period (ASC style) **and** first 2 points (DDDAS style) | Real, remaining periods / 7 days | n/a | `mambaCR3BP6d_DDDAS.py`, `mambaGMATTest.py` | Training from scratch on very little real data. Baseline that (c) must beat at equal target epochs. |
| (c) transfer | Same as (a) | Real, train on 1 period, test on 4 | yes, 2 epochs, modes `full` and `feature` | corrected `mambaTLCR3BP.py` | The ASC 2025 claim on real data. |

Model matrix per branch: `{Mamba, LSTM} × {seed 0, 1, 2}`. Three seeds is the minimum to
report a spread; single-seed numbers from the papers cannot be compared to anything.

Target objects (`configs/targets.yaml`):

| Role | Object | NORAD | Why |
|---|---|---|---|
| Primary passive LEO | Envisat | 27386 | Defunct since 2012, ~765 km SSO, densely tracked, no maneuvers. |
| Secondary passive LEO, lower | rocket body, perigee 450–550 km | **VERIFY** via satcat query `OBJECT_TYPE=ROCKET BODY`, `DECAY=null`, `PERIGEE 450--550` | Stronger drag; stresses F8. |
| Contaminated (optional) | ISS | 25544 | Known reboosts. Use only to show that a maneuver breaks all three branches — not for the headline numbers. |

Windows: ASC-style = 5 orbital periods (~8.4 h for Envisat, ≈ 505 samples). DDDAS-style
long-term = 7 days = 10080 samples (matches the existing `data/classification/leo/10080min-10`
synthetic set). Both cut from the same TLE history so the epoch is identical.

Hyper-parameters (copied from the scripts, not re-tuned):

| | Mamba | LSTM |
|---|---|---|
| Layers / hidden | `MambaConfig(d_model=6, n_layers=1)` (d_state 16, expand 2, d_conv 4; d_conv 16 for DDDAS-style) | `LSTM(6, 30, 6, 1, 0)` |
| Optimizer | Adam, lr 1e-3 (`Adam_mini` in DDDAS script; keep Adam) | same |
| Loss | `torch.nn.HuberLoss()` source, `F.smooth_l1_loss` target (as in ASC) — unify to Huber | same |
| Epochs | 5 source / 2 target (ASC); 50 for 2-point DDDAS | same |
| Batch / lookback | 8 / 1 | same |
| dtype | `.double()` | same |

---

## 5. Implementation steps (item 3)

Order is strict. Each step ends with a runnable command and an artifact.

**Step 0 — environment**
1. `git clone` qutils next to this repo; `pip install -e ../qutils`; `pip install -r requirements.txt`.
2. `ln -s ~/git/GMAT-Thrust-Data/data data/synthetic`.
3. Export `SPACETRACK_USER`, `SPACETRACK_PASS`.
4. Apply the qutils fix in 8.2 (F4) first; every later number depends on it.

**Step 1 — real data**
1. `python scripts/fetch_tles.py --norad 27386 --start 2026-06-01 --stop 2026-07-15`
   → `data/tle/27386_*.tle`. Window dates are placeholders; pick a 45-day span with no
   space-weather storm (check CelesTrak SW file, Ap < 30 throughout). **VERIFY** the span.
2. `python scripts/build_real_dataset.py --norad 27386 --minutes 505 --windows 40`
   and `--minutes 10080 --windows 4` → two `statesArrayReal.npz` files.
3. `pytest tests/test_tle_ephem.py`. Attach the hand-off histogram (`figures/27386-HandoffJump.png`).

**Step 2 — branch (b), real low-data** (first because it needs no synthetic data)
1. Port `mambaCR3BP6d_DDDAS.py`: replace `ode87(...)` with `load_real(...)`, keep the training
   loop (`train_size = 2`, 50 epochs) → DDDAS-style run.
2. Port the ASC source-domain half (`p_motion_knowledge = 1/5`, 5 epochs) → ASC-style run.
3. Both write the standard plot set (section 6.1) and a results row.

**Step 3 — matched synthetic source**
1. Take the first real state of the target window, `ECI2OE` it, write a GMAT config with
   `fixedElements: true`, `epoch` = window start, ballistic coefficient from B* (section 8.1).
2. `./generateGMATNoThrustDataset.sh` with that config → `data/regression/leo/505min-<N>` where
   N ≥ 10 (one matched trajectory + 9 with small spreads for the multi-trajectory variant).
3. Sanity plot: synthetic vs real for the first period, position difference < 20 km at 1 period.
   If larger, drag/space-weather mismatch dominates; record it, do not tune it away.

**Step 4 — branch (a), zero-shot**
1. Port `mambaTLCR3BP.py` source half; train on 1 synthetic period, plot source domain.
2. Evaluate the same model on the real 5-period window without training (teacher-forced and
   rollout). This reproduces the ASC procedure as run (F1).

**Step 5 — branch (c), transfer**
1. Same source model as step 4, then `transfer(mode)` and `fit(newModel, real, train_size=1 period,
   epochs=2)`. Evaluate on the remaining 4 periods.
2. Run `mode ∈ {full, feature}`, `{Mamba, LSTM}`, 3 seeds.
3. Optional multi-trajectory source: pretrain on the N-trajectory synthetic set (concatenate
   `create_datasets` outputs), then repeat 1–2. Keep as a separate `tl_mode` value `full-multi`.

**Step 6 — long-term variant**
Repeat (a), (b), (c) on the 7-day window. This is the "long term propagation" deliverable of (b).

**Step 7 — summary**
`python scripts/summarize.py` → `figures/Summary-MAPE.pdf`, `figures/Summary-Rollout.pdf`,
`results/summary.md` (MAPE table in the Table 1–3 format of the ASC paper, plus noise floor row).

**Step 8 — pendulumRNN back-port**
Fix F1 in `mambaTLCR3BP.py`, re-run orbit pairs 1–3, record corrected CR3BP MAPE tables in
`results/asc2025_corrected.md`. Without this the real-data results have no honest baseline.

---

## 6. Milestones and pass criteria (item 4)

### 6.1 Standard plot set (every run, every branch)

Reproduce the ASC 2025 figure kinds with the same qutils calls so figures are directly comparable:

| Figure | qutils call | ASC/DDDAS analogue |
|---|---|---|
| State prediction vs time, train region marked | `plotStatePredictions(..., DU, TU, timeLabel='Periods')` | DDDAS Fig. 3 |
| 3-D orbit, truth / Mamba / LSTM | `plot3dOrbitPredictions(..., networkLabel='Mamba')` + LSTM line | ASC Figs. 6, 7 (`...-SourceDomainPrediction.pdf`) |
| Percent error per state | `newPlotSolutionErrors(..., percentError=True)` | ASC Figs. 10–13 (`...-PredictionError<Model>.png`) |
| Conserved quantity | `plotEnergy(..., orbitalEnergyJ2, xLabel='Number of Periods (T)')` | ASC Figs. 8, 9 (`...-ConservedQuantity.pdf`) |
| **New:** rollout position error vs periods, with the hand-off noise floor as a shaded band | matplotlib in `tlv/metrics.py` | none — this is the honest addition |

File names: `figures/<object>-<branch>-<Domain><Plot>-<model>.pdf`, e.g.
`27386-c-TargetDomainConservedQuantity-Mamba.pdf`.

### 6.2 Milestones

Gates are pass/fail. Targets are what a good result looks like; missing a target is a finding,
not a failure.

| # | Milestone | Gate | Target |
|---|---|---|---|
| M0 | Real dataset built and validated | `pytest tests/` green; hand-off median < 2 km, p95 < 10 km; `meta.json` has TLE sha256 | 45-day span, Ap < 30 |
| M1 | Branch (b) ASC-style (train 1 period, test 4) | Runs end-to-end for Mamba and LSTM, 3 seeds; plot set + CSV rows written | Mamba teacher-forced position MAPE ≤ 1 %; Mamba < LSTM on every state; rollout position error at 1 T reported |
| M1b | Branch (b) DDDAS-style (2 points, 7 days) | Same as M1 on the 7-day window | Mamba energy drift < 1 % over 7 days (teacher-forced); rollout reported |
| M2 | Branch (a) zero-shot | Source-domain plots match ASC source quality (Mamba position MAPE ≤ 0.1 % on synthetic); real-target numbers written with noise floor | Zero-shot degradation quantified (expected large); Mamba < LSTM |
| M3 | Branch (c) transfer | `full` and `feature`, Mamba and LSTM, 3 seeds; corrected training loop verified by a unit test that asserts `newModel` parameters change | **Headline:** Mamba TL (2 target epochs) beats Mamba from-scratch (branch b, 2 epochs on the same period) and beats zero-shot (a), on teacher-forced MAPE *and* rollout at 1 T, median over seeds |
| M4 | Long-term (7-day) versions of a/b/c + summary figures + `results/summary.md` | Summary table has one row per (branch, model, tl_mode) with median ± spread over seeds and the noise-floor row | TL advantage persists on the 7-day window |
| M5 | pendulumRNN F1 fix and corrected CR3BP tables | Pairs 1–3 re-run, tables in repo | Corrected fine-tuned numbers ≤ the paper's zero-shot numbers (if not, say so) |

Equal-budget rule for M3: branch (b)-2-epoch and branch (c) must see the same real samples, the
same number of target epochs, the same seed and the same batch size. Only the initial weights
differ. Anything else is not a transfer-learning comparison.

---

## 7. Baseline reuse map (item 5)

| Past script | Reuse as | Take | Leave |
|---|---|---|---|
| `ASC2025/mambaTLCR3BP.py` | skeleton for `branch_a_zero_shot.py`, `branch_c_transfer.py` | model construction, hyper-parameters, plot sequence, figure naming, `mape` printout, `plotGen` block | the target-phase training bug (F1); `sys.argv` orbit-pair selection; CR3BP `system()` |
| `dddas/mambaCR3BP6d_DDDAS.py` | skeleton for `branch_b_real_lowdata.py` (2-point variant) | `train_size = 2`, `d_conv=16`, 50 epochs, LSTM comparison block | inline training loop → use `qutils.ml.regression.trainModel`; `Adam_mini` → Adam |
| `journals/orbitProp/p2bp/mambaGMATTest.py` | pattern for reading an Earth-orbit ephemeris and nondim/dim handling | `dim2NonDim6` → train → `plotStatePredictions(..., DU, TU)` → `nonDim2Dim6`; `plotOrbitPhasePredictions`, `plot3dOrbitPredictions`, `plotEnergy(orbitalEnergy)` | `readGMATReport` (replaced by `load_real`); `qutils.mlExtras` import (module does not exist in qutils main — use `qutils.ml.utils.rmse`) |
| `GMAT-Thrust-Data/python/testProcessDatasetsCart.py` | `apply_noise`, `orbitalEnergy` plotting, FFT spectrum plot (useful synthetic-vs-real spectral comparison) | `fft_amplitude_spectrum` | everything else |
| `GMAT-Thrust-Data/python/convertToOE.py` | none — use `qutils.orbital.ECI2OE` (identical function) | | `ECI2TLE` (F6) |
| `qutils.ml.regression` | as-is | `create_datasets`, `trainModel`, `LSTM`, `transferMamba`, `transferModelAll` | `transferLSTM` (loads only `lstm`, leaves `fc` random — use `transferModelAll` for LSTM too) |

---

## 8. Changes to supporting repos (item 6)

### 8.1 GMAT-Thrust-Data (surgical, three edits)

1. **Fixed elements + epoch + ballistic parameters in config.** In
   `python/generateSpacecraftThrustOpt.py`, inside the sampling loop, add a branch:

```python
if config.get("fixedElements", False):
    # Use the exact elements from the config. Apply the optional spreads.
    SMA[i] = config["SMA"] + rng.uniform(-1, 1) * config.get("SMA_spread", 0.0)
    ECC[i] = config["ECC"] + rng.uniform(-1, 1) * config.get("ECC_spread", 0.0)
    INC[i] = config["INC"] + rng.uniform(-1, 1) * config.get("INC_spread", 0.0)
    RAAN[i] = config["RAAN"] + rng.uniform(-1, 1) * config.get("RAAN_spread", 0.0)
    AOP[i] = config["AOP"] + rng.uniform(-1, 1) * config.get("AOP_spread", 0.0)
    TA[i] = config["TA"] + rng.uniform(-1, 1) * config.get("TA_spread", 0.0)
    DryMasses[i] = config.get("dryMass", DryMasses[i])
    DragAreas[i] = config.get("dragArea", DragAreas[i])
    Cds[i] = config.get("Cd", Cds[i])
    SRPAreas[i] = config.get("srpArea", SRPAreas[i])
    Crs[i] = config.get("Cr", Crs[i])
else:
    ...existing sampling...
```

   and before the spacecraft is built: `earthorb.SetField("Epoch", config.get("epoch", "20 Jul 2020 12:00:00.000"))`.
   Ballistic coefficient from a TLE: `Cd*A/m [m^2/kg] = 2 * Bstar / 0.1570` (`Bstar` in 1/Earth-radii,
   ρ0 = 0.1570 kg m⁻² ER⁻¹). B* is a fit parameter and often unphysical; clamp to
   `[0.005, 0.05] m²/kg` and log the raw value. Keep `DryMass = 80`, set `DragArea = (Cd*A/m) * 80 / 2.2`, `Cd = 2.2`.

2. **Save metadata in the npz.** In each `np.savez(...)` call add
   `elements=np.stack([SMA, ECC, INC, RAAN, AOP, TA], 1)`, `phys=np.stack([SRPAreas, Crs, DragAreas, Cds, DryMasses], 1)`,
   `epoch=str(...)`, `dt=dt`. `combineDatasets.py` and `convertToOE.py` read only the named
   key, so extra keys are backwards-compatible (checked).

3. **Space weather (optional, M4).** Set `Atmos` fields `F107`, `F107A`, `MagneticIndex`
   from the CelesTrak SW-All file for the config epoch. Without this, F8 stays as-is; with it,
   the OOD gap in branch (a) shrinks and the demo can show *which* physics causes the gap.

Do not touch `convertToTLE.py`; do not use its output.

### 8.2 qutils (two edits, one addition)

1. **Constants (F4).** In `qutils/orbital.py` define once
   `MU_EARTH = 398600.4415`, `DU_EARTH = 6378.1`, `TU_EARTH = (DU_EARTH**3 / MU_EARTH) ** 0.5`
   and use them as the defaults of `dim2NonDim4/6`, `nonDim2Dim4/6`, `orbitalEnergy`.
   This changes default outputs of old scripts by 0.23 % in velocity; note it in the commit.
   (`MU_EARTH_JGM2` already exists further down the file — reuse it.)
2. **`transferMamba`.** No code change. Add a docstring line stating that `in_proj` and
   `A_log` remain frozen and that `transferModelAll` is the full-fine-tune path.
3. **Addition (only after M3 is green):** move `tlv/tle_ephem.py::teme_to_j2000` and
   `segment_ephemeris` into `qutils/orbital.py`. Not before — keep the demo self-contained
   until the interface is stable.

### 8.3 pendulumRNN (one fix)

`scripts/conferences/ASC2025/mambaTLCR3BP.py`: `trainModel(newModel, ...)` and
`plotStatePredictions(newModel, ...)` in the target phase; also pass `newModel` to the
`plotGen` block. Re-run pairs 1–3 (M5).

---

## 9. Handoff for coding models (item 7)

### 9.1 Environment facts

- Python ≥ 3.10, PyTorch CPU is sufficient (models have < 5 k parameters; ASC runs take minutes).
- `qutils` is installed from a sibling checkout. Its `ml/__init__.py` star-imports every
  submodule; importing `qutils.ml` pulls in `shap` on Python ≥ 3.7. If `shap` is missing,
  `pip install shap` or import `qutils.ml.regression` directly.
- GMAT is needed only for step 3 (matched synthetic source). It lives in `~/git/GMAT-Thrust-Data`
  and is driven by `./generateGMATNoThrustDataset.sh` with `config.yaml` at the repo root
  (`python -m python.generateSpacecraftThrustOpt --propType none`). The GMAT Python API is loaded
  by `python/load_gmat.py`. Run it from that repo root.
- Existing synthetic sets you can use before GMAT runs (**checked 2026-09-17**, this list
  replaces the one written at audit time): `data/regression/leo/450min-4000`,
  `90min-10000`, `30min-{2000,3000,10000}`. `regression/leo/450min-10000` and
  `900min-10000` are **empty directories** — the files were never there. Inclination is
  uniform 0–180° in all of them (F7), and none carry metadata keys (F5).

### 9.2 Order of work and definition of done

Follow section 5 in order. A step is done when its command runs from a clean checkout, its
artifact exists, and its test (if any) passes. Do not start branch (c) before M1 and M2 are green.

### 9.3 Guardrails

1. Never modify arrays in place before nondimensionalizing (`dim2NonDim6` mutates its input).
   `nondim()` copies.
2. Never compare real TLE mean elements to `TLEArray*` (F6).
3. Never report a teacher-forced number without the rollout number next to it (F3).
4. Never report an error metric without the hand-off noise floor next to it (section 2).
5. Every result row has a seed; every headline number is a median over ≥ 3 seeds.
6. Equal-budget rule for TL comparisons (section 6.2).
7. Any deviation from the ASC hyper-parameters goes into the CSV as a column, not a comment.
8. Keep `tlv/` at six modules. If a seventh seems necessary, write the reason in this file first.
9. Real-data plots carry the text "truth = SGP4 from space-track TLEs" in the figure title or caption.

### 9.4 Tests that must exist

- `tests/test_tle_ephem.py` — the three checks in section 3.2.
- `tests/test_datasets.py` — `nondim(nonDim2Dim6(x)) == x` to 1e-12 after the qutils fix;
  `load_synthetic` shape `(N, T, 6)`; `split_periods` returns `train_size` such that
  `train_size * 60 / period_sec ≈ train_periods`.
- `tests/test_transfer.py` — after `transfer(src, tgt, 'full')` and one `fit` step on random
  data, at least one parameter tensor of `tgt` differs from `src` (guards against F1
  recurring); after `'feature'`, `tgt.layers[0].mixer.in_proj` equals `src`'s.

### 9.5 Open items for Hunter (not for the coding model)

1. Confirm the primary object (Envisat 27386) and pick the secondary lower LEO object.
2. Confirm the 45-day TLE window after checking space weather.
3. Decide whether the ISS maneuver-contamination case is in the demo or not.
4. Decide whether M5 (pendulumRNN corrected tables) is published alongside the demo. It changes the
   interpretation of the ASC 2025 target-domain results.

### 9.6 Quick command reference

```bash
# Step 0
pip install -e ../qutils && pip install -r requirements.txt
ln -s ~/git/GMAT-Thrust-Data/data data/synthetic
# Step 1
python scripts/fetch_tles.py --norad 27386 --start 2026-06-01 --stop 2026-07-15
python scripts/build_real_dataset.py --norad 27386 --minutes 505 --windows 40
python scripts/build_real_dataset.py --norad 27386 --minutes 10080 --windows 4
pytest tests/test_tle_ephem.py
# Step 2
python scripts/branch_b_real_lowdata.py --norad 27386 --minutes 505 --style asc   --model mamba --seed 0
python scripts/branch_b_real_lowdata.py --norad 27386 --minutes 10080 --style dddas --model mamba --seed 0
# Step 3 (in ~/git/GMAT-Thrust-Data)
cp <tl-validation>/configs/gmat_27386.yaml config.yaml && ./generateGMATNoThrustDataset.sh
# Step 4/5
python scripts/branch_a_zero_shot.py --norad 27386 --minutes 505 --model mamba --seed 0
python scripts/branch_c_transfer.py  --norad 27386 --minutes 505 --model mamba --tl full --seed 0
# Step 7
python scripts/summarize.py
```
