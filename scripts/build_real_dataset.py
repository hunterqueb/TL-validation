#!/usr/bin/env python
"""TLE cache -> statesArrayReal.npz + meta.json + hand-off histogram (P1-06, P1-07)."""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tlv import tle_ephem as te
from tlv.spacetrack import fetch_tle_history, cache_path, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--minutes", type=int, required=True)
    ap.add_argument("--windows", type=int, required=True)
    ap.add_argument("--start")
    ap.add_argument("--stop")
    ap.add_argument("--targets", default="configs/targets.yaml")
    ap.add_argument("--out-root", default="data/real")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.targets))
    window = next((t["window"] for t in cfg["targets"].values()
                   if isinstance(t, dict) and t.get("norad") == a.norad
                   and "window" in t), {})
    start = a.start or window.get("start")
    stop = a.stop or window.get("stop")
    if not (start and stop):
        raise SystemExit("no window: pass --start/--stop or add the object to targets.yaml")

    text = fetch_tle_history(a.norad, start, stop)
    sats = te.parse_tles(text)
    print(f"{len(sats)} unique TLEs, {start} .. {stop}")

    r, v, t_jd, idx, jump = te.segment_ephemeris(sats)
    print(f"{len(r)} samples at {te.DT:.0f} s, {len(np.unique(idx))} TLEs used")
    r, v = te.teme_to_j2000(r, v, t_jd)

    states = np.hstack([r, v])
    windows = te.to_windows(states, t_jd, idx, jump, a.minutes, a.windows)

    period = te.period_sec(sats[0][1])
    out = pathlib.Path(a.out_root) / str(a.norad) / f"{a.minutes}min-{a.windows}"
    meta = te.write_real_npz(out, windows, dict(
        norad=a.norad, tle_count=len(sats), first_epoch_jd=float(t_jd[0]),
        last_epoch_jd=float(t_jd[-1]), window_start=start, window_stop=stop,
        period_sec=period,
        tle_cache=str(cache_path(a.norad, start, stop)), tle_sha256=sha256(text)))
    print(f"wrote {out}/statesArrayReal.npz {meta['shape']}")
    print(f"period {period:.1f} s; hand-off median {meta['handoff_median_km']:.3f} km, "
          f"p95 {meta['handoff_p95_km']:.3f} km over {meta['handoff_count']} hand-offs")

    nz = windows["handoff_jump_km"][windows["handoff_jump_km"] > 0]
    if nz.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(nz, bins=40)
        ax.axvline(meta["handoff_median_km"], color="k", ls="--",
                   label=f"median {meta['handoff_median_km']:.2f} km")
        ax.axvline(meta["handoff_p95_km"], color="r", ls=":",
                   label=f"p95 {meta['handoff_p95_km']:.2f} km")
        ax.set_xlabel("TLE hand-off position jump (km)")
        ax.set_ylabel("count")
        ax.set_title(f"{a.norad} truth noise floor ({a.minutes} min windows)\n"
                     "truth = SGP4 from space-track TLEs", fontsize=10)
        ax.legend()
        fig.tight_layout()
        path = pathlib.Path("figures") / f"{a.norad}-HandoffJump.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")

    if meta["handoff_median_km"] >= 2.0 or meta["handoff_p95_km"] >= 10.0:
        print("WARNING: M0 gate failed -- see PLAN.md section 6, risk 1 (change the window "
              "before changing the object; log both attempts)")


if __name__ == "__main__":
    main()
