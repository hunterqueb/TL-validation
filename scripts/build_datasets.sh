#!/bin/sh
# Build every real dataset the branch scripts read. Windows come from configs/targets.yaml;
# TLEs come from data/tle/ if cached, else space-track (credentials in .env).
#
#   ./scripts/build_datasets.sh
set -eu
cd "$(dirname "$0")/.."

# Yamal 202, GEO: asc/asc2 (6 periods x 8) and dddas (7 days x 4)
python scripts/build_real_dataset.py --norad 28089 --minutes 8620  --windows 8
python scripts/build_real_dataset.py --norad 28089 --minutes 10080 --windows 4

# BLOCK DM R/B, HEO: asc/asc2 (6 periods x 8) and dddas (7 days x 3 -- 4 do not fit the span)
python scripts/build_real_dataset.py --norad 26769 --minutes 4630  --windows 8
python scripts/build_real_dataset.py --norad 26769 --minutes 10080 --windows 3
