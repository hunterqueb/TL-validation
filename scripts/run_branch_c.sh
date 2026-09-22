#!/bin/sh
# Branch (c): synthetic pretrain -> 2-epoch real fine-tune, the corrected ASC loop.
# Run it only after branches (a) and (b) have produced results.
#
#   ./scripts/run_branch_c.sh              # {full, feature} x both models x 3 seeds (12 runs)
#   ./scripts/run_branch_c.sh full         # one transfer mode
#   ./scripts/run_branch_c.sh full long    # the 7-day window
#
# Compared against branch (b) asc2 at the same window and seed. Run
# ./scripts/run_branch_a.sh first so (a) and (c) share source weights.
. "$(dirname "$0")/_common.sh"

MODES=${1:-"full feature"}
case ${2:-asc} in
    asc)  style=asc;  minutes=$ASC_MIN;  windows=$ASC_WIN ;;
    long) style=long; minutes=$LONG_MIN; windows=$LONG_WIN ;;
    *) echo "unknown style '$2' (asc, long)" >&2; exit 2 ;;
esac

need_real "$minutes" "$windows"
need_synthetic "$ASC_MIN" 10

for model in $MODELS; do
    for seed in $SEEDS; do
        [ -f "results/models/$NORAD-a-$model-s$seed.pt" ] || echo \
            "WARNING: results/models/$NORAD-a-$model-s$seed.pt missing -- this run will train" \
            "its own source model, so (a) and (c) will not share initial weights. Run" \
            "./scripts/run_branch_a.sh first." >&2
    done
done

for mode in $MODES; do
    for model in $MODELS; do
        for seed in $SEEDS; do
            run "c/$style tl=$mode $model seed $seed" scripts/branch_c_transfer.py \
                --norad "$NORAD" --minutes "$minutes" --windows "$windows" \
                --style "$style" --model "$model" --tl "$mode" --seed "$seed" \
                --window-index "$WINDOW"
        done
    done
done

finish
