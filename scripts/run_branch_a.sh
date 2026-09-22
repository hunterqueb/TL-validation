#!/bin/sh
# Branch (a): train on the matched GMAT source, evaluate on real with NO target training.
# Saves the source weights to results/models/ for branch (c).
#
#   ./scripts/run_branch_a.sh              # both models, 3 seeds (6 runs), ASC window
#   ./scripts/run_branch_a.sh long         # the 7-day window instead
#   SEEDS=0 MODELS=mamba ./scripts/run_branch_a.sh
#
# Needs the matched GEO source domain (a hand-run GMAT job); stops with the command to build it.
. "$(dirname "$0")/_common.sh"

case ${1:-asc} in
    asc)  style=asc;  minutes=$ASC_MIN;  windows=$ASC_WIN ;;
    long) style=long; minutes=$LONG_MIN; windows=$LONG_WIN ;;
    *) echo "unknown style '$1' (asc, long)" >&2; exit 2 ;;
esac

need_real "$minutes" "$windows"
need_synthetic "$ASC_MIN" 10          # branch_a.yaml trains the source on this set

for model in $MODELS; do
    for seed in $SEEDS; do
        run "a/$style $model seed $seed" scripts/branch_a_zero_shot.py \
            --norad "$NORAD" --minutes "$minutes" --windows "$windows" \
            --style "$style" --model "$model" --seed "$seed" --window-index "$WINDOW"
    done
done

finish
