"""Shared plumbing for the branch scripts: window loading, the standard plot set, result rows.

Lives in scripts/ rather than tlv/ so tlv stays at six modules (PLAN.md rule 9).
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from qutils.ml.regression import create_datasets
from qutils.ml.utils import mape, rmse
from qutils.orbital import orbitalEnergyJ2
from qutils.plot import plot3dOrbitPredictions, newPlotSolutionErrors, plotEnergy

from tlv import datasets as ds
from tlv import metrics as mt
from tlv.train import predict_teacher_forced, rollout, STATES, UNITS

FIGDIR = pathlib.Path("figures")
TRUTH_NOTE = "truth = SGP4 from space-track TLEs"


def load_targets(path="configs/targets.yaml"):
    return yaml.safe_load(open(path))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_window(norad, minutes, n_windows, index=0, period_sec=None):
    """One real window: dimensional states, time base, hand-off jumps, period."""
    states, t_sec, extras, meta = ds.load_real(norad, minutes, n_windows)
    period_sec = period_sec or meta.get("period_sec")
    if period_sec is None:
        raise ValueError("period_sec missing from meta.json -- rebuild the dataset (P1-04)")
    return dict(seq=states[index], t_sec=np.asarray(t_sec, dtype=float),
                handoff=extras["handoff_jump_km"][index], period_sec=float(period_sec),
                meta=meta, index=index)


def provenance(w):
    """Row fields that say which data a result came from (HANDOFF REV-09)."""
    return dict(window=w["index"], tle_sha256=w["meta"].get("tle_sha256", "")[:12])


def datasets_for(seq_dim, train_size, device):
    """Non-dimensionalize (copies) and cut into the qutils lookback-1 tensors."""
    if train_size >= len(seq_dim) - 1:
        raise ValueError(
            f"train_size {train_size} leaves no test set in a {len(seq_dim)}-sample window: "
            "the window is shorter than the requested training span")
    seq = ds.nondim(seq_dim)
    return seq, create_datasets(seq, 1, train_size, device)


def evaluate(model, seq_nd, seq_dim, t_sec, period_sec, train_size, handoff, device,
             prefix, model_name, domain="TargetDomain", plots=True):
    """Standard plot set (HANDOFF 6.1) + the metrics every row needs.

    Returns a dict of result-row fields. Teacher-forced and rollout are always produced
    together (guardrail 9.3.3) and the noise floor always travels with them (9.3.4).
    """
    train_in, train_out, test_in, test_out = create_datasets(seq_nd, 1, train_size, device)
    t_periods = t_sec / period_sec
    test_size = len(seq_nd) - train_size
    # Forecast origin = the last observed state. With lookback 1 the training transitions
    # end at x_{train_size-1} -> x_{train_size}, so state train_size is a training target
    # and scoring starts at train_size + 1 (HANDOFF REV-02/REV-04).
    origin = train_size

    tf_dim = predict_teacher_forced(model, t_periods, seq_nd.copy(), train_in, test_in,
                                    train_size, test_size,
                                    DU=ds.DU_EARTH, TU=ds.TU_EARTH, plotOn=plots)
    ro_dim = ds.redim(rollout(model, seq_nd, train_size, device))

    if plots:
        FIGDIR.mkdir(exist_ok=True)
        plt.gcf().suptitle(TRUTH_NOTE, fontsize=8)
        _save(f"{prefix}-{domain}Prediction-{model_name}.pdf")

        plot3dOrbitPredictions(seq_dim, tf_dim, networkLabel=model_name,
                               title=f"{domain}\n{TRUTH_NOTE}")
        _save(f"{prefix}-{domain}Orbit-{model_name}.pdf")

        newPlotSolutionErrors(seq_dim, tf_dim, t_periods, timeLabel="Periods",
                              percentError=True, states=STATES, units=UNITS)
        plt.gcf().set_size_inches(10, 8)
        _save(f"{prefix}-{domain}PredictionError-{model_name}.png")

        plotEnergy(seq_dim, tf_dim, t_periods, orbitalEnergyJ2,
                   xLabel="Number of Periods (T)", yLabel="J2 energy (km$^2$/s$^2$)",
                   networkLabel=f"{model_name} {domain}")
        plt.gcf().suptitle(TRUTH_NOTE, fontsize=8)
        _save(f"{prefix}-{domain}ConservedQuantity-{model_name}.pdf")

        mt.plot_rollout_error(ro_dim, seq_dim, t_sec, period_sec, handoff,
                              path=FIGDIR / f"{prefix}-{domain}Rollout-{model_name}.png",
                              title=f"{prefix} {domain} closed-loop rollout",
                              label=model_name, origin=origin)

    test = slice(origin + 1, None)                  # strictly after the last observed state
    m = mape(seq_dim[test], tf_dim[test], axis=0)   # per-state MAPE, as in the ASC tables
    ro_err = mt.position_error_at(ro_dim, seq_dim, t_sec, period_sec, periods=(1, 5),
                                  origin=origin)
    med, p95 = mt.noise_floor(handoff)
    return dict(
        mape_x=m[0], mape_y=m[1], mape_z=m[2], mape_vx=m[3], mape_vy=m[4], mape_vz=m[5],
        rmse_pos_km=float(np.sqrt(np.mean(
            np.sum((tf_dim[test, :3] - seq_dim[test, :3]) ** 2, axis=1)))),
        rollout_pos_1T_km=ro_err[1], rollout_pos_5T_km=ro_err[5],
        energy_drift_pct=mt.energy_drift(tf_dim[test]),
        handoff_med_km=med, handoff_p95_km=p95, handoff_n=mt.handoff_count(handoff),
        origin_sec=float(t_sec[origin]),
    )


def _save(name):
    plt.gcf().savefig(FIGDIR / name, bbox_inches="tight")
    plt.close("all")


def report(row, csv_path):
    run_id = mt.append_result(csv_path, row)
    print(f"[{run_id}] {row['branch']}/{row.get('style','')} {row['model']} "
          f"tl={row['tl_mode']} seed={row['seed']}: "
          f"MAPE pos {np.mean([row['mape_x'], row['mape_y'], row['mape_z']]):.4g} % | "
          f"rollout 1T {row['rollout_pos_1T_km']:.3g} km | "
          f"hand-off med {row['handoff_med_km']:.3g} / p95 {row['handoff_p95_km']:.3g} km "
          f"(n={row['handoff_n']})")
    return run_id
