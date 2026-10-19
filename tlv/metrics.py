"""Metrics and the append-only results CSV."""
import csv
import pathlib
import uuid

import numpy as np

from qutils.ml.utils import mape, rmse  # noqa: F401  re-exported
from qutils.orbital import orbitalEnergyJ2

COLUMNS = ["run_id", "branch", "style", "object", "window", "model", "tl_mode",
           "train_periods", "epochs_src", "epochs_tgt", "mape_x", "mape_y", "mape_z",
           "mape_vx", "mape_vy", "mape_vz", "rmse_pos_km", "energy_drift_pct", "handoff_med_km", "handoff_p95_km",
           "handoff_n", "origin_sec", "tle_sha256", "train_s", "test_s", "n_params", "seed"]


def energy_drift(seq_dim, mu=None, Re=None):
    """Percent drift of the J2 energy over the sequence."""
    kw = {k: v for k, v in (("mu", mu), ("Re", Re)) if v is not None}
    E = orbitalEnergyJ2(np.asarray(seq_dim, dtype=float), **kw)
    return float(100.0 * (E.max() - E.min()) / abs(E[0]))


def noise_floor(handoff_jump_km):
    """(median, p95) of the TLE hand-off jumps, NaN if there are none.

    A consistency statistic between successive TLEs, not an accuracy floor.
    """
    j = np.asarray(handoff_jump_km)
    nz = j[j > 0]
    if nz.size == 0:
        return float("nan"), float("nan")
    return float(np.median(nz)), float(np.percentile(nz, 95))


def handoff_count(handoff_jump_km):
    """Number of hand-offs behind noise_floor()."""
    return int(np.count_nonzero(np.asarray(handoff_jump_km) > 0))


def append_result(csv_path, row):
    """Append one result row. Append-only: never edit a row, re-runs get a new run_id."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown result columns: {sorted(unknown)}")
    path = pathlib.Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("run_id", uuid.uuid4().hex[:8])
    new = not path.exists()
    # an older file keeps its own header, so dropped columns stay aligned (left blank)
    fields = COLUMNS if new else next(csv.reader(path.open()))
    added = [c for c in COLUMNS if c in row and c not in fields]
    if added:  # widen an older header; existing rows keep their values, blank in the new columns
        rows = list(csv.DictReader(path.open()))
        fields = fields + added
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in fields})
    return row["run_id"]

