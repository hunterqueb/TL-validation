"""Metrics and the append-only results CSV (HANDOFF section 3.6)."""
import csv
import pathlib
import uuid

import numpy as np

from qutils.ml.utils import mape, rmse  # noqa: F401  (re-exported; HANDOFF 3.6)
from qutils.orbital import orbitalEnergyJ2

COLUMNS = ["run_id", "branch", "style", "object", "window", "model", "tl_mode",
           "train_periods", "epochs_src", "epochs_tgt", "mape_x", "mape_y", "mape_z",
           "mape_vx", "mape_vy", "mape_vz", "rmse_pos_km", "rollout_pos_1T_km",
           "rollout_pos_5T_km", "energy_drift_pct", "handoff_med_km", "handoff_p95_km",
           "handoff_n", "origin_sec", "tle_sha256", "train_s", "seed"]


def energy_drift(seq_dim, mu=None, Re=None):
    """Percent drift of the J2 energy over the sequence (two-body energy oscillates ~0.1 %
    under J2; the J2-corrected energy is the right conserved-ish quantity for LEO)."""
    kw = {k: v for k, v in (("mu", mu), ("Re", Re)) if v is not None}
    E = orbitalEnergyJ2(np.asarray(seq_dim, dtype=float), **kw)
    return float(100.0 * (E.max() - E.min()) / abs(E[0]))


def position_error_at(pred, truth, t_sec, period_sec, periods=(1, 2, 5), origin=0):
    """Position error (km) at each horizon, measured from the forecast origin.

    `origin` is the index of the last observed state. Horizon p is the first sample at or
    after t_sec[origin] + p * period_sec -- NOT p periods from the window start, which is
    what made the old "1T" number a 120-second rollout (HANDOFF REV-02). NaN when the
    window ends before the horizon: an unavailable number, not a good one.
    """
    out = {}
    t0 = float(np.asarray(t_sec)[origin])
    for p in periods:
        j = int(np.searchsorted(t_sec, t0 + p * period_sec))
        if j >= len(truth) or np.any(np.isnan(pred[j, :3])):
            out[p] = float("nan")
        else:
            out[p] = float(np.linalg.norm(pred[j, :3] - truth[j, :3]))
    return out


def noise_floor(handoff_jump_km):
    """(median, p95) of the TLE hand-off position jumps in this window.

    This is a *consistency* statistic between successive orbit products, not a calibrated
    accuracy floor: SGP4-densified TLE states are model-derived, and errors shared by two
    consecutive TLEs are invisible to it (HANDOFF REV-01). Returns NaN when the window
    contains no hand-off -- absent evidence, which is not the same as zero uncertainty.
    """
    j = np.asarray(handoff_jump_km)
    nz = j[j > 0]
    if nz.size == 0:
        return float("nan"), float("nan")
    return float(np.median(nz)), float(np.percentile(nz, 95))


def handoff_count(handoff_jump_km):
    """How many TLE hand-offs the statistic above is based on (HANDOFF REV-01)."""
    return int(np.count_nonzero(np.asarray(handoff_jump_km) > 0))


def append_result(csv_path, row):
    """Append one result row. Append-only: never edit a row, re-runs get a new run_id."""
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"columns not in HANDOFF 3.6: {sorted(unknown)}")
    path = pathlib.Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("run_id", uuid.uuid4().hex[:8])
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})
    return row["run_id"]


def plot_rollout_error(pred, truth, t_sec, period_sec, handoff_jump_km, path=None,
                       title="", label="Mamba", origin=0):
    """Rollout position error vs periods *since the forecast origin*, over the hand-off band.

    The honest propagation figure (HANDOFF 6.1). No error is shown without the hand-off
    statistic beside it (guardrail 9.3.4), and that band is consistency, not accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    med, p95 = noise_floor(handoff_jump_km)
    n = handoff_count(handoff_jump_km)
    err = np.linalg.norm(np.asarray(pred)[:, :3] - np.asarray(truth)[:, :3], axis=1)
    periods = (np.asarray(t_sec) - np.asarray(t_sec)[origin]) / period_sec

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if np.isfinite(p95):
        ax.axhspan(0, p95, color="0.8",
                   label=f"TLE hand-off consistency, p95 {p95:.2f} km (n={n})")
        ax.axhline(med, color="0.5", ls="--", lw=1, label=f"hand-off median {med:.2f} km")
    else:
        ax.text(0.02, 0.95, "no TLE hand-off in this window:\nconsistency unmeasured",
                transform=ax.transAxes, va="top", fontsize=8, color="0.4")
    ax.semilogy(periods, err, lw=1.2, label=f"{label} rollout")
    ax.set_xlabel("Periods since the forecast origin (T)")
    ax.set_ylabel("Position error (km)")
    ax.set_title(f"{title}\ntruth = SGP4 from space-track TLEs", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return med, p95
