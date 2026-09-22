#!/usr/bin/env python
"""Write a GMAT-Thrust-Data config matched to one real object and window.

SMA is pinned through lowerAlt = upperAlt = r_p - 6378 km (the generator has no SMA key).
Epoch, SRP area, Cr and dry mass need a fixedElements/epoch patch to the generator and are
written as a commented block; until then SRP, the dominant GEO perturbation, is random per run.
"""
import argparse
import pathlib
import sys

import numpy as np
import yaml
from astropy.time import Time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from qutils.orbital import ECI2OE, MU_EARTH
from tlv import datasets as ds
from tlv.spacetrack import fetch_tle_history
from tlv.tle_ephem import parse_tles

RHO0 = 0.1570          # kg m^-2 ER^-1, the B* reference density
CDAM_LIMITS = (0.005, 0.05)    # m^2/kg -- B* is a fit parameter and is often unphysical
DRY_MASS = 80.0
CD = 2.2


def cd_a_over_m(bstar):
    """Cd*A/m [m^2/kg] from B* [1/Earth radii]. Returns (clamped, raw)."""
    raw = 2.0 * bstar / RHO0
    return float(np.clip(raw, *CDAM_LIMITS)), float(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--minutes", type=int, default=8620)
    ap.add_argument("--windows", type=int, default=9)
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--num-runs", type=int, default=10)
    ap.add_argument("--orbit", default="geo", help="GMAT orbit_type / directory name")
    ap.add_argument("--spread", type=float, default=0.1,
                    help="degrees of spread on INC/RAAN/AOP/TA for the neighbour runs")
    ap.add_argument("--out")
    a = ap.parse_args()

    states, _, extras, meta = ds.load_real(a.norad, a.minutes, a.windows)
    s0 = states[a.window_index, 0]
    oe = ECI2OE(s0[:3], s0[3:], mu=MU_EARTH)       # [a, e, i, RAAN, AOP, M0, P]
    epoch = Time(float(extras["epoch_jd"][a.window_index]), format="jd", scale="utc")

    sats = parse_tles(pathlib.Path(meta["tle_cache"]).read_text())
    bstar = sats[0][1].bstar
    cdam, raw = cd_a_over_m(bstar)

    Re_gmat = 6378.0       # generateSpacecraftThrustOpt.py uses R = 6378, not 6378.1
    r_p = float(oe[0]) * (1.0 - float(oe[1]))

    cfg = {
        "orbit_type": a.orbit,
        "numMinsToProp": a.minutes,
        "num_runs": a.num_runs,
        "lowerAlt": round(r_p - Re_gmat, 3),
        "upperAlt": round(r_p - Re_gmat, 3),
        "eccentricity": round(float(oe[1]), 6),
        "inclination": round(float(np.rad2deg(oe[2])), 4),
        "inclination_spread": a.spread,
        "RAAN": round(float(np.rad2deg(oe[3])) % 360, 4),
        "RAAN_spread": a.spread,
        "argPeriapsis": round(float(np.rad2deg(oe[4])) % 360, 4),
        "argPeriapsis_spread": a.spread,
        "trueAnomaly": round(float(np.rad2deg(oe[5])) % 360, 4),   # mean anomaly; see note
        "trueAnomaly_spread": a.spread,
        # unused by --propType none, but read at import time, so they must be present
        "deltaV": 0.002, "deltaVStd": 0.0,
        "chemThrust": 10, "chemThrustStd": 0,
        "elecThrust": 0.1, "elecThrustStd": 0,
        "chemThrustMin": int(a.minutes * 0.2), "elecThrustMin": int(a.minutes * 0.8),
        "chemPropTimeStd": 0, "elecPropTimeStd": 0,
    }

    out = pathlib.Path(a.out or f"configs/gmat/gmat_{a.norad}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Matched synthetic source domain for NORAD {a.norad}, window {a.window_index} of\n"
        f"# {a.minutes}min-{a.windows}. Copy to GMAT-Thrust-Data/config.yaml and run\n"
        f"# ./generateGMATNoThrustDataset.sh -> data/regression/{a.orbit}/"
        f"{a.minutes}min-{a.num_runs}/statesArrayNoThrust.npz\n#\n"
        f"# Osculating J2000 elements at the window start:\n"
        f"#   SMA {oe[0]:.3f} km  ECC {oe[1]:.6f}  INC {np.rad2deg(oe[2]):.4f}"
        f"  RAAN {np.rad2deg(oe[3])%360:.4f}  AOP {np.rad2deg(oe[4])%360:.4f}"
        f"  MA {np.rad2deg(oe[5])%360:.4f} deg\n"
        f"# SMA is pinned through lowerAlt = upperAlt = r_p - 6378 km; across the sampled\n"
        f"# eccentricity it spans ~{oe[0]*oe[1]:.0f} km. trueAnomaly is filled from the mean\n"
        f"# anomaly (difference ~{np.rad2deg(2*oe[1]):.3f} deg at this eccentricity).\n"
        f"# B* {bstar:.6e} 1/ER -> Cd*A/m raw {raw:.5f}, clamped {cdam:.5f} m^2/kg"
        f"{'  [B* is 0: no drag fit, expected at GEO]' if bstar == 0 else ''}\n")
    footer = (
        "\n# --- NOT READ YET: needs a fixedElements/epoch patch to the generator -------\n"
        "# Without it the epoch stays hardcoded at \"20 Jul 2020 12:00:00.000\" and SRPArea,\n"
        "# Cr, DragArea, Cd and DryMass are sampled at random per run.\n"
        f"# epoch: \"{epoch.strftime('%d %b %Y %H:%M:%S.000')}\"\n"
        "# fixedElements: true\n"
        f"# dryMass: {DRY_MASS}\n"
        "# srpArea: 2.5\n"
        "# Cr: 1.75\n"
        f"# dragArea: {cdam * DRY_MASS / CD:.4f}\n"
        f"# Cd: {CD}\n")
    out.write_text(header + "\n" + yaml.safe_dump(cfg, sort_keys=False) + footer)
    print(header)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
