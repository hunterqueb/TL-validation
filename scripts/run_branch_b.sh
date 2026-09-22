#!/bin/sh
# Branch (b): train from scratch on very little real data.
#
#   ./scripts/run_branch_b.sh              # all three styles, both models, 3 seeds (18 runs)
#   ./scripts/run_branch_b.sh asc asc2     # only these styles
#   SEEDS=0 MODELS=mamba ./scripts/run_branch_b.sh dddas
#
# asc    train on the first period, 5 epochs
# asc2   same window, 2 epochs -- the equal-budget row branch (c) is compared against
# dddas  train on the first 2 points, 50 epochs, d_conv 16, 7-day window
. "$(dirname "$0")/_common.sh"

STYLES=${*:-"asc asc2 dddas"}

for style in $STYLES; do
    case $style in
        asc|asc2) minutes=$ASC_MIN;  windows=$ASC_WIN ;;
        dddas)    minutes=$LONG_MIN; windows=$LONG_WIN ;;
        *) echo "unknown style '$style' (asc, asc2, dddas)" >&2; exit 2 ;;
    esac
    need_real "$minutes" "$windows"
    for model in $MODELS; do
        for seed in $SEEDS; do
            run "b/$style $model seed $seed" scripts/branch_b_real_lowdata.py \
                --norad "$NORAD" --minutes "$minutes" --windows "$windows" \
                --style "$style" --model "$model" --seed "$seed" --window-index "$WINDOW"
        done
    done
done

finish
