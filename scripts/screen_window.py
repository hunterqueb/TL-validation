#!/usr/bin/env python
"""Pick the TLE window before any model is run (P1-02, H-2; REV-05 wants this recorded).

Criteria, in order:
  1. the span must hold `--days` of continuous TLE coverage;
  2. the largest gap between consecutive TLE epochs is minimised (SGP4 is only propagated
     inside a TLE's own validity window, so a long gap is a long extrapolation);
  3. no |d(sma)| step between consecutive TLEs exceeds `--max-step` km. At GEO a station-
     keeping or relocation burn shows up as a step of several km; fit noise is ~0.1 km.

Ap is not a criterion at GEO (HANDOFF 9.5 item 2).
"""
import argparse
import pathlib
import sys

import numpy as np
from astropy.time import Time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tlv.spacetrack import fetch_tle_history
from tlv.tle_ephem import parse_tles

MU = 398600.4418


def elements(sats):
    """(epoch_jd, sma_km) per TLE, from the mean motion."""
    jd = np.array([jd for jd, _ in sats])
    n = np.array([s.no_kozai for _, s in sats]) / 60.0          # rad/s
    return jd, (MU / n ** 2) ** (1 / 3)


def score(jd, sma, days):
    """Every start index that holds `days` of coverage, as (max_gap_h, max_step_km, n, i)."""
    d = jd - jd[0]
    out = []
    for i in range(len(d)):
        j = int(np.searchsorted(d, d[i] + days))
        if j >= len(d):
            break
        out.append((float(np.diff(d[i:j + 1]).max() * 24),
                    float(np.abs(np.diff(sma[i:j + 1])).max()), j - i + 1, i))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--start", required=True, help="ISO date of the search span")
    ap.add_argument("--stop", required=True)
    ap.add_argument("--minutes", type=int, default=8620, help="window length")
    ap.add_argument("--windows", type=int, default=8, help="how many windows must fit")
    ap.add_argument("--max-step", type=float, default=0.1, help="km, maneuver screen")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    sats = parse_tles(fetch_tle_history(a.norad, a.start, a.stop))
    jd, sma = elements(sats)
    days = a.windows * a.minutes / 1440.0
    print(f"{len(sats)} TLEs over {jd[-1] - jd[0]:.1f} d; "
          f"need {days:.3f} d for {a.windows} x {a.minutes} min")
    print(f"sma range {sma.ptp():.3f} km over the whole span "
          f"({'no maneuver signature' if sma.ptp() < 5 else 'CHECK: possible maneuver'})")

    cand = score(jd, sma, days)
    if not cand:
        raise SystemExit("no start date holds that much coverage -- widen --start/--stop")
    clean = [c for c in cand if c[1] <= a.max_step] or cand
    if not any(c[1] <= a.max_step for c in cand):
        print(f"WARNING: no span is free of a {a.max_step} km step; showing the best anyway")
    for gap, step, n, i in sorted(clean)[:a.top]:
        start = Time(jd[i], format="jd").utc.iso[:10]
        stop = Time(jd[i] + days + 1, format="jd").utc.iso[:10]   # +1 d of margin
        print(f"  --start {start} --stop {stop}  {n:3d} TLEs  "
              f"max gap {gap:5.2f} h  max |d(sma)| {step:.3f} km")


if __name__ == "__main__":
    main()
