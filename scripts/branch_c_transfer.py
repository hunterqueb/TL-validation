#!/usr/bin/env python
"""Branch (c): synthetic pretrain -> 2-epoch real fine-tune.

The target model is the one optimized and evaluated; the ASC 2025 script trained neither.
Compare against branch (b) --style asc2 at the same seed: only the initial weights differ.
"""
import argparse
import pathlib
import sys

import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import datasets_for, evaluate, load_window, provenance, report, set_seed
from branch_a_zero_shot import source_model_path, train_source
from qutils.ml.utils import getDevice
from tlv import datasets as ds
from tlv.models import return_model, transfer
from tlv.train import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--model", choices=["mamba", "lstm"], default="mamba")
    ap.add_argument("--tl", choices=["full", "feature", "full-multi"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", default="asc")
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--config", default="configs/branch_c.yaml")
    ap.add_argument("--source-config", default="configs/branch_a.yaml")
    ap.add_argument("--csv", default="results/branch_c.csv")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    device = getDevice()
    set_seed(a.seed)

    kw = {k: v for k, v in cfg["models"][a.model].items() if k in ("d_conv", "n_layers")}
    source = return_model(a.model, device, **kw)
    saved = source_model_path(a.norad, a.model, a.seed)
    if saved.exists():          # reuse branch (a)'s weights so (a) and (c) share a source
        source.load_state_dict(torch.load(saved))
        print(f"loaded source weights {saved}")
    else:
        print(f"{saved} missing -- training the source model here (run branch (a) first "
              "to keep the two branches on identical weights)")
        source, _, _ = train_source(yaml.safe_load(open(a.source_config)), a.norad,
                                    a.model, a.seed, device, plots=False)

    w = load_window(a.norad, a.minutes, a.windows, a.window_index)
    train_size = ds.split_periods(w["seq"], w["period_sec"], cfg["target"]["train_periods"])
    seq_nd, datasets = datasets_for(w["seq"], train_size, device)

    target = transfer(source, return_model(a.model, device, **kw), a.tl)
    train_s = fit(target, datasets, cfg["epochs_tgt"], lr=cfg["lr"])

    row = evaluate(target, seq_nd, w["seq"], w["t_sec"], w["period_sec"], train_size,
                   w["handoff"], device, f"{a.norad}-c-{a.tl}-{a.model}-s{a.seed}",
                   a.model.capitalize(),
                   domain="TargetDomain", plots=not a.no_plots)
    row.update(branch="c", style=a.style, object=a.norad, model=a.model, tl_mode=a.tl,
               train_periods=cfg["target"]["train_periods"], epochs_src=cfg["epochs_src"],
               epochs_tgt=cfg["epochs_tgt"], train_s=train_s, seed=a.seed,
               **provenance(w))
    report(row, a.csv)


if __name__ == "__main__":
    main()
