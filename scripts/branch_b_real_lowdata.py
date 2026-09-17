#!/usr/bin/env python
"""Branch (b): train from scratch on very little real data (P2-04).

--style asc    train on the first period, 5 epochs, test on the remaining 4 (ASC)
--style asc2   same window, 2 epochs -- the equal-budget row branch (c) is compared against
--style dddas  train on the first 2 points, 50 epochs, d_conv 16, 7-day window (DDDAS)
"""
import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import datasets_for, evaluate, load_window, report, set_seed
from qutils.ml.utils import getDevice
from tlv import datasets as ds
from tlv.models import return_model
from tlv.train import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--style", choices=["asc", "asc2", "dddas"], default="asc")
    ap.add_argument("--model", choices=["mamba", "lstm"], default="mamba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--config", default="configs/branch_b.yaml")
    ap.add_argument("--csv", default="results/branch_b.csv")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    style = cfg["styles"][a.style]
    device = getDevice()
    set_seed(a.seed)

    w = load_window(a.norad, a.minutes, a.windows, a.window_index)
    train_size = (style["train_size"] if style["train_size"] is not None
                  else ds.split_periods(w["seq"], w["period_sec"], style["train_periods"]))
    seq_nd, datasets = datasets_for(w["seq"], train_size, device)

    model = return_model(a.model, device, d_conv=style["d_conv"])
    train_s = fit(model, datasets, style["epochs"], lr=cfg["lr"])

    prefix = f"{a.norad}-b{'' if a.style == 'asc' else '-' + a.style}"
    row = evaluate(model, seq_nd, w["seq"], w["t_sec"], w["period_sec"], train_size,
                   w["handoff"], device, prefix, a.model.capitalize(),
                   domain="TargetDomain", plots=not a.no_plots)
    row.update(branch="b", style=a.style, object=a.norad, model=a.model, tl_mode="none",
               train_periods=train_size * 60.0 / w["period_sec"], epochs_src=0,
               epochs_tgt=style["epochs"], train_s=train_s, seed=a.seed)
    report(row, a.csv)


if __name__ == "__main__":
    main()
