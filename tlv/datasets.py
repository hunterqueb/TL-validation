"""Load synthetic and real state arrays, non-dimensionalize, split, match source orbits."""
import json
import pathlib

import numpy as np

from qutils.orbital import dim2NonDim6, nonDim2Dim6, DU_EARTH, TU_EARTH

DT = 60.0


def load_synthetic(root="data/synthetic", dataset="regression", orbit="leo",
                   minutes=450, n_sys=10000, tag="NoThrust"):
    """Load statesArray<tag>.npz from root/<dataset>/<orbit>/<minutes>min-<n_sys>.

    GMAT sample j is the state after the (j+1)-th step, so t_j = (j+1)*dt (HANDOFF F5).
    Returns (states (N, T, 6), t_sec (T,), extras dict of any metadata keys present).
    """
    path = pathlib.Path(root) / dataset / orbit / f"{minutes}min-{n_sys}" / f"statesArray{tag}.npz"
    with np.load(path, allow_pickle=True) as f:
        states = f[f"statesArray{tag}"]
        extras = {k: f[k] for k in f.files if k != f"statesArray{tag}"}
    t_sec = (np.arange(states.shape[1]) + 1) * DT
    return states, t_sec, extras


def load_real(norad, minutes, n_windows, root="data/real"):
    """Load statesArrayReal.npz and its side keys. Returns (states, t_sec, extras, meta)."""
    path = pathlib.Path(root) / str(norad) / f"{minutes}min-{n_windows}"
    with np.load(path / "statesArrayReal.npz") as f:
        states = f["statesArrayReal"]
        extras = {k: f[k] for k in f.files if k != "statesArrayReal"}
    meta = json.loads((path / "meta.json").read_text())
    return states, extras["t_sec"][0], extras, meta


def nondim(states, DU=DU_EARTH, TU=TU_EARTH):
    """dim2NonDim6 with explicit DU, TU (HANDOFF F4). Copies -- dim2NonDim6 mutates."""
    return dim2NonDim6(np.array(states, dtype=float, copy=True), DU, TU)


def redim(states, DU=DU_EARTH, TU=TU_EARTH):
    """Inverse of nondim. Copies."""
    return nonDim2Dim6(np.array(states, dtype=float, copy=True), DU, TU)


def split_periods(seq, period_sec, train_periods, dt=DT):
    """train_size for 'train on train_periods, test on the rest' (ASC style)."""
    return int(round(train_periods * period_sec / dt))


def nearest_synthetic(oe_target, oe_bank, k=1):
    """Indices of the k synthetic trajectories nearest oe_target in [a (km), e, i (rad)].

    Distance is normalized by (100 km, 0.01, 5 deg). Used only in branch (c)-multi.
    """
    scale = np.array([100.0, 0.01, np.deg2rad(5.0)])
    d = np.linalg.norm((np.asarray(oe_bank)[:, :3] - np.asarray(oe_target)[:3]) / scale, axis=1)
    idx = np.argsort(d)[:k]
    return idx, d[idx]
