#!/usr/bin/env python
"""results/branch_*.csv -> results/summary.md + comparison figures."""
import argparse
import glob
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MAPE = ["mape_x", "mape_y", "mape_z", "mape_vx", "mape_vy", "mape_vz"]
EXTRA = ["rmse_pos_km", "energy_drift_pct"]
# windows get their own row; comparisons are paired by window and seed
KEYS = ["object", "branch", "style", "model", "tl_mode", "window"]
RUN = KEYS + ["seed"]              # one result per configuration, not per rerun


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.md")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.results}/branch_*.csv"))
    if not files:
        raise SystemExit(f"no {a.results}/branch_*.csv yet")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # CSVs are append-only; keep the last rerun so repeats don't reweight a median
    n_all = len(df)
    df = df.drop_duplicates(subset=RUN, keep="last")
    if len(df) < n_all:
        print(f"{n_all - len(df)} superseded rerun(s) dropped; last row per {RUN} kept")

    med = df.groupby(KEYS)[MAPE + EXTRA].median()
    iqr = df.groupby(KEYS)[MAPE + EXTRA].quantile(.75) - df.groupby(KEYS)[MAPE + EXTRA].quantile(.25)
    seeds = df.groupby(KEYS)["seed"].nunique()

    lines = ["# Summary", "",
             "Truth = SGP4 from space-track TLEs. Every number is the median over seeds; "
             "+- is the inter-quartile range. Rows with fewer than 3 seeds are flagged (!).",
             "", "| object | branch | style | model | tl | window | seeds | " +
             " | ".join(MAPE) + " | energy drift (%) |",
             "|" + "---|" * (len(KEYS) + 1 + len(MAPE) + 1)]
    for key in med.index:
        n = int(seeds.loc[key])
        flag = "" if n >= 3 else " (!)"
        cells = [f"{med.loc[key, c]:.4g} ± {iqr.loc[key, c]:.2g}" for c in MAPE]
        cells += [f"{med.loc[key, 'energy_drift_pct']:.4g}"]
        lines.append("| " + " | ".join(list(map(str, key)) + [f"{n}{flag}"] + cells) + " |")

    nf = df[["handoff_med_km", "handoff_p95_km", "handoff_n"]].median()
    lines += ["", f"**TLE hand-off consistency**: median {nf['handoff_med_km']:.3g} km, "
                  f"p95 {nf['handoff_p95_km']:.3g} km over {nf['handoff_n']:.0f} hand-offs "
                  "(median across runs). Successive TLEs disagree by this much, so errors "
                  "at or below it say nothing about a model. It is a consistency statistic "
                  "between orbit products, not a calibrated accuracy floor."]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(med)} rows from {len(files)} csv files)")
    print("figures: read the dataviz skill before writing Summary-MAPE")


if __name__ == "__main__":
    main()
