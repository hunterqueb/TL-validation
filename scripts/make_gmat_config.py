#!/usr/bin/env python
"""Write a GMAT-Thrust-Data config matched to one real object and window (R2-03).

Fixed elements from the first real state of the target window, epoch from that state, and
a ballistic coefficient from the TLE B*. Run 0 is exact; runs >= 1 get the small spreads
(GMAT-Thrust-Data applies `_spread` only when i > 0 after the section 8.1 patch).
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
    ap.add_argument("--minutes", type=int, default=505)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--num-runs", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()

    states, _, extras, meta = ds.load_real(a.norad, a.minutes, a.windows)
    s0 = states[a.window_index, 0]
    oe = ECI2OE(s0[:3], s0[3:], mu=MU_EARTH)       # [a, e, i, RAAN, AOP, M0, P]
    epoch = Time(float(extras["epoch_jd"][a.window_index]), format="jd", scale="utc")

    sats = parse_tles(pathlib.Path(meta["tle_cache"]).read_text())
    bstar = sats[0][1].bstar
    cdam, raw = cd_a_over_m(bstar)

    cfg = {
        "fixedElements": True,
        "epoch": epoch.strftime("%d %b %Y %H:%M:%S.000"),
        "SMA": float(oe[0]), "ECC": float(oe[1]),
        "INC": float(np.rad2deg(oe[2])), "RAAN": float(np.rad2deg(oe[3])),
        "AOP": float(np.rad2deg(oe[4])), "TA": float(np.rad2deg(oe[5])),  # M0 ~ TA only for e~0
        "SMA_spread": 5.0, "ECC_spread": 0.0005,
        "INC_spread": 0.1, "RAAN_spread": 0.1, "AOP_spread": 0.1, "TA_spread": 0.1,
        "dryMass": DRY_MASS, "Cd": CD, "dragArea": cdam * DRY_MASS / CD,
        "srpArea": 1.0, "Cr": 1.8,
        "num_runs": a.num_runs, "numMinsToProp": a.minutes, "dt": 60,
        "inclination": 180,      # unused when fixedElements is true (F7)
    }

    out = pathlib.Path(a.out or f"configs/gmat_{a.norad}.yaml")
    header = (f"# matched source domain for NORAD {a.norad}, window {a.window_index} of "
              f"{a.minutes}min-{a.windows}\n"
              f"# B* {bstar:.6e} 1/ER -> Cd*A/m raw {raw:.5f}, clamped {cdam:.5f} m^2/kg "
              f"(limits {CDAM_LIMITS})\n"
              f"# run 0 is exact; runs >= 1 use the _spread values\n")
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    print(header + f"wrote {out}")
    print("NOTE: TA is filled from the mean anomaly; for e ~ 0 the difference is small, "
          "but check it if the object is eccentric.")


if __name__ == "__main__":
    main()
