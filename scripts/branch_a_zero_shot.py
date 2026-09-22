#!/usr/bin/env python
"""Branch (a): train on matched synthetic, evaluate on real with NO target training.

This is what the ASC 2025 script actually did (its target model was never trained), and
the baseline for branch (c), which starts from the source weights saved here.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import datasets_for, evaluate, load_window, provenance, report, set_seed
from qutils.ml.utils import getDevice
from qutils.orbital import ECI2OE
from tlv import datasets as ds
from tlv.models import return_model
from tlv.train import fit

MODELS = pathlib.Path("results/models")


def source_model_path(norad, model, seed):
    return MODELS / f"{norad}-a-{model}-s{seed}.pt"


def train_source(cfg, norad, model_name, seed, device, plots=True, prefix=None):
    """Train on one matched synthetic trajectory, 1 period, 5 epochs. Returns (model, s)."""
    src = cfg["source"]
    states, t_syn, _ = ds.load_synthetic(orbit=src.get("orbit", "leo"),
                                        minutes=src["minutes"], n_sys=src["n_sys"])
    seq_dim = states[src["index"]]
    period_syn = float(ECI2OE(seq_dim[0, :3], seq_dim[0, 3:])[6])

    train_size = ds.split_periods(seq_dim, period_syn, src["train_periods"])
    seq_nd, datasets = datasets_for(seq_dim, train_size, device)
    kw = {k: v for k, v in cfg["models"][model_name].items() if k in ("d_conv", "n_layers")}
    model = return_model(model_name, device, **kw)
    train_s = fit(model, datasets, cfg["epochs_src"], lr=cfg["lr"])

    src_row = evaluate(model, seq_nd, seq_dim, t_syn, period_syn, train_size,
                       np.zeros_like(t_syn), device, prefix or f"{norad}-a-{model_name}-s{seed}",
                       model_name.capitalize(), domain="SourceDomain", plots=plots)
    print(f"source domain ({src['minutes']}min-{src['n_sys']} run {src['index']}): "
          f"position MAPE {np.mean([src_row['mape_x'], src_row['mape_y'], src_row['mape_z']]):.4g} % "
          "(target: <= 0.1 %)")
    return model, train_s, src_row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--model", choices=["mamba", "lstm"], default="mamba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", default="asc")
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--config", default="configs/branch_a.yaml")
    ap.add_argument("--csv", default="results/branch_a.csv")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = getDevice()
    set_seed(a.seed)

    model, train_s, _ = train_source(cfg, a.norad, a.model, a.seed, device,
                                     plots=not a.no_plots)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), source_model_path(a.norad, a.model, a.seed))

    # zero shot: the real window is only ever evaluated, never trained on
    w = load_window(a.norad, a.minutes, a.windows, a.window_index)
    train_size = ds.split_periods(w["seq"], w["period_sec"], 1)
    seq_nd, _ = datasets_for(w["seq"], train_size, device)
    row = evaluate(model, seq_nd, w["seq"], w["t_sec"], w["period_sec"], train_size,
                   w["handoff"], device, f"{a.norad}-a-{a.model}-s{a.seed}",
                   a.model.capitalize(),
                   domain="TargetDomain", plots=not a.no_plots)
    row.update(branch="a", style=a.style, object=a.norad, model=a.model, tl_mode="none",
               train_periods=0, epochs_src=cfg["epochs_src"], epochs_tgt=0,
               train_s=train_s, seed=a.seed, **provenance(w))
    report(row, a.csv)


if __name__ == "__main__":
    main()
