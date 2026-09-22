#!/usr/bin/env python
"""Download and cache one object's TLE history."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tlv.spacetrack import fetch_tle_history, cache_path, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--start", required=True, help="ISO date, e.g. 2026-06-01")
    ap.add_argument("--stop", required=True)
    ap.add_argument("--cache-dir", default="data/tle")
    a = ap.parse_args()

    cached = cache_path(a.norad, a.start, a.stop, a.cache_dir).exists()
    text = fetch_tle_history(a.norad, a.start, a.stop, a.cache_dir)
    n = sum(1 for ln in text.splitlines() if ln.startswith("1 "))
    print(f"{'cache' if cached else 'space-track'}: {n} TLEs -> "
          f"{cache_path(a.norad, a.start, a.stop, a.cache_dir)}")
    print(f"sha256 {sha256(text)}")


if __name__ == "__main__":
    main()
