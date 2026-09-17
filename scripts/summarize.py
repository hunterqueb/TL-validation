#!/usr/bin/env python
"""results/branch_*.csv -> results/summary.md + comparison figures (P7-01, P7-02)."""
import argparse
import glob
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MAPE = ["mape_x", "mape_y", "mape_z", "mape_vx", "mape_vy", "mape_vz"]
EXTRA = ["rollout_pos_1T_km", "rollout_pos_5T_km", "rmse_pos_km", "energy_drift_pct"]
KEYS = ["branch", "style", "model", "tl_mode"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.md")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.results}/branch_*.csv"))
    if not files:
        raise SystemExit(f"no {a.results}/branch_*.csv yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    med = df.groupby(KEYS)[MAPE + EXTRA].median()
    iqr = df.groupby(KEYS)[MAPE + EXTRA].quantile(.75) - df.groupby(KEYS)[MAPE + EXTRA].quantile(.25)
    seeds = df.groupby(KEYS)["seed"].nunique()

    lines = ["# Summary", "",
             "Truth = SGP4 from space-track TLEs. Every number is the median over seeds; "
             "+- is the inter-quartile range. Rows with fewer than 3 seeds are flagged (!).",
             "", "| branch | style | model | tl | seeds | " +
             " | ".join(MAPE) + " | rollout 1T (km) | rollout 5T (km) | energy drift (%) |",
             "|" + "---|" * (5 + len(MAPE) + 3)]
    for key in med.index:
        n = int(seeds.loc[key])
        flag = "" if n >= 3 else " (!)"
        cells = [f"{med.loc[key, c]:.4g} ± {iqr.loc[key, c]:.2g}" for c in MAPE]
        cells += [f"{med.loc[key, c]:.4g}" for c in
                  ["rollout_pos_1T_km", "rollout_pos_5T_km", "energy_drift_pct"]]
        lines.append("| " + " | ".join(list(map(str, key)) + [f"{n}{flag}"] + cells) + " |")

    nf = df[["noise_floor_med_km", "noise_floor_p95_km"]].median()
    lines += ["", f"**Noise floor** (TLE hand-off jump): median "
                  f"{nf['noise_floor_med_km']:.3g} km, p95 {nf['noise_floor_p95_km']:.3g} km. "
                  "No error above is meaningful below this."]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(med)} rows from {len(files)} csv files)")
    print("figures: read the dataviz skill before writing Summary-MAPE / Summary-Rollout (P7-02)")


if __name__ == "__main__":
    main()
