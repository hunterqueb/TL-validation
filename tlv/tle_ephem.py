"""Convert a TLE list to a 60 s J2000 ephemeris with the GMAT array layout (SGP4-quality truth)."""
import json
import pathlib

import numpy as np
import astropy.units as u
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from sgp4.api import Satrec
from sgp4 import __version__ as sgp4_version

from qutils.orbital import ECI2OE

DT = 60.0


def parse_tles(text):
    """Return a list of (epoch_jd, Satrec) sorted by epoch, duplicates removed."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    by_epoch = {}
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            sat = Satrec.twoline2rv(lines[i], lines[i + 1])
            by_epoch[round(sat.jdsatepoch + sat.jdsatepochF, 9)] = sat
    return sorted(by_epoch.items())


def period_sec(sat):
    """Orbital period from the TLE mean motion (no_kozai is rad/min)."""
    return 2 * np.pi / sat.no_kozai * 60.0


def segment_ephemeris(sats, dt=DT):
    """Propagate each TLE on a uniform grid from its own epoch to the next TLE's epoch.

    Returns r, v (M, 3) TEME km, km/s; t_jd; tle_index; jump_km, which is |r_new - r_old|
    at the first sample of each new TLE and 0 elsewhere.
    """
    epochs = np.array([jd for jd, _ in sats])
    jd0 = epochs[0]
    n = int((epochs[-1] - jd0) * 86400.0 / dt) + 1
    fr = np.arange(n) * dt / 86400.0
    # epochs - jd0 loses ~7 digits, so an epoch on a sample can land a few us late;
    # shift the boundaries 0.1 ms early so the new TLE wins the tie.
    tle_index = np.searchsorted(epochs - jd0 - 1e-9, fr, side="right") - 1

    r = np.empty((n, 3))
    v = np.empty((n, 3))
    jump_km = np.zeros(n)
    jd_col = np.full(n, jd0)

    for k in np.unique(tle_index):
        mask = tle_index == k
        e, rk, vk = sats[k][1].sgp4_array(jd_col[mask], fr[mask])
        if np.any(e):
            bad = np.flatnonzero(e)[0]
            raise RuntimeError(f"SGP4 error code {e[bad]} for TLE {k} at sample {bad}")
        r[mask], v[mask] = rk, vk
        if k > 0:  # same instant, previous TLE -> hand-off jump
            first = np.flatnonzero(mask)[0]
            e_old, r_old, _ = sats[k - 1][1].sgp4(jd0, fr[first])
            if e_old == 0:
                jump_km[first] = np.linalg.norm(r[first] - np.asarray(r_old))

    return r, v, jd0 + fr, tle_index, jump_km


def teme_to_j2000(r_teme, v_teme, t_jd):
    """Rotate TEME to GCRS with astropy. GCRS-to-J2000 frame bias (~23 mas) is ignored."""
    t = Time(t_jd, format="jd", scale="utc")
    teme = TEME(CartesianRepresentation(r_teme.T * u.km,
                differentials=CartesianDifferential(v_teme.T * u.km / u.s)), obstime=t)
    gcrs = teme.transform_to(GCRS(obstime=t))
    return (gcrs.cartesian.xyz.to_value(u.km).T,
            gcrs.velocity.d_xyz.to_value(u.km / u.s).T)


def to_windows(states, t_jd, tle_index, jump_km, minutes, n_windows=None, dt=DT):
    """Cut a continuous ephemeris into N windows of T = minutes samples. Drop the remainder."""
    T = int(round(minutes * 60.0 / dt))
    N = len(states) // T
    if n_windows is not None:
        if N < n_windows:
            raise ValueError(f"ephemeris holds {N} windows of {minutes} min, asked for {n_windows}")
        N = n_windows
    cut = N * T

    states = states[:cut].reshape(N, T, 6)
    oe = np.array([[ECI2OE(s[:3], s[3:]) for s in w] for w in states])
    return {
        "statesArrayReal": states,
        "t_sec": np.tile(np.arange(T) * dt, (N, 1)),
        "epoch_jd": t_jd[:cut].reshape(N, T)[:, 0],
        "tle_index": tle_index[:cut].reshape(N, T),
        "handoff_jump_km": jump_km[:cut].reshape(N, T),
        "oe": oe,
    }


def write_real_npz(path, windows, meta):
    """Write statesArrayReal.npz + meta.json."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "statesArrayReal.npz", **windows)
    jump = windows["handoff_jump_km"]
    nz = jump[jump > 0]
    meta = dict(meta,
                sgp4_version=sgp4_version,
                frame_chain="TEME -> GCRS (astropy); GCRS~J2000 bias ignored",
                dt=DT,
                shape=list(windows["statesArrayReal"].shape),
                handoff_count=int(nz.size),
                handoff_median_km=float(np.median(nz)) if nz.size else 0.0,
                handoff_p95_km=float(np.percentile(nz, 95)) if nz.size else 0.0)
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
