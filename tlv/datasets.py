"""Load synthetic and real state arrays, non-dimensionalize, split, match source orbits."""
import json
import pathlib

import numpy as np
import yaml

from qutils.orbital import dim2NonDim6, nonDim2Dim6, DU_EARTH, TU_EARTH

DT = 60.0
DATA_YAML = pathlib.Path(__file__).resolve().parents[1] / "data.yaml"


def data_root(key="gmat"):
    """Path for `key` from data.yaml."""
    cfg = yaml.safe_load(DATA_YAML.read_text())
    if key not in cfg:
        raise KeyError(f"{key!r} not in {DATA_YAML}: has {sorted(cfg)}")
    return pathlib.Path(cfg[key]).expanduser()


def load_synthetic(root=None, dataset="regression", orbit="leo",
                   minutes=450, n_sys=10000, tag="NoThrust"):
    """Load <root>/<dataset>/<orbit>/<minutes>min-<n_sys>/statesArray<tag>.npz.

    Returns (states (N, T, 6), t_sec (T,), extras). t_j = (j+1)*dt: GMAT sample j follows step j+1.
    """
    root = data_root() if root is None else pathlib.Path(root)
    path = root / dataset / orbit / f"{minutes}min-{n_sys}" / f"statesArray{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- check the `gmat` path in {DATA_YAML}")
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
    """dim2NonDim6 with explicit DU, TU (the qutils defaults once disagreed). Copies -- dim2NonDim6 mutates."""
    return dim2NonDim6(np.array(states, dtype=float, copy=True), DU, TU)


def redim(states, DU=DU_EARTH, TU=TU_EARTH):
    """Inverse of nondim. Copies."""
    return nonDim2Dim6(np.array(states, dtype=float, copy=True), DU, TU)


def split_periods(seq, period_sec, train_periods, dt=DT):
    """train_size for 'train on train_periods, test on the rest' (ASC style)."""
    return int(round(train_periods * period_sec / dt))


def nearest_synthetic(oe_target, oe_bank, k=1):
    """(indices, distances) of the k bank rows nearest oe_target in scaled [a, e, i]."""
    scale = np.array([100.0, 0.01, np.deg2rad(5.0)])
    d = np.linalg.norm((np.asarray(oe_bank)[:, :3] - np.asarray(oe_target)[:3]) / scale, axis=1)
    idx = np.argsort(d)[:k]
    return idx, d[idx]
