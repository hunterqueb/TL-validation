# Shared settings and helpers for the branch run scripts. Source it, do not run it.
#
# Everything is overridable from the environment, so a narrowed re-run needs no edit:
#   SEEDS=0 MODELS=mamba ./scripts/run_branch_b.sh asc
set -u

cd "$(dirname "$0")/.."          # repo root: the configs use paths relative to it

NORAD=${NORAD:-28089}
case $NORAD in                   # per-object windows, as in configs/targets.yaml
    26769) asc_min=4630; long_win=3 ;;                  # Block DM R/B, HEO, 771.3 min period
    *)     asc_min=8620; long_win=4 ;;                  # Yamal 202, GEO, 1436.1 min period
esac
ASC_MIN=${ASC_MIN:-$asc_min}; ASC_WIN=${ASC_WIN:-8}     # 6 periods (1 train + 5 forecast)
LONG_MIN=${LONG_MIN:-10080}; LONG_WIN=${LONG_WIN:-$long_win}  # 7 days
WINDOW=${WINDOW:-0}                                     # which window of the set
SEEDS=${SEEDS:-"0 1 2"}                                 # >= 3 for any headline number
MODELS=${MODELS:-"mamba lstm"}
SUMMARIZE=${SUMMARIZE:-1}

RAN=0
FAILED=0
FAILED_LABELS=""

# run <label> <script> <args...> -- one matrix cell, plots on.
run () {
    label=$1; shift
    RAN=$((RAN + 1))
    printf '\n=== %s ===\n' "$label"
    if python "$@"; then
        return 0
    fi
    FAILED=$((FAILED + 1))
    FAILED_LABELS="$FAILED_LABELS\n  $label"
    echo "FAILED: $label"
}

# need_real <minutes> <windows> -- the branch scripts read a built dataset, they do not build one.
need_real () {
    path="data/real/$NORAD/$1min-$2/statesArrayReal.npz"
    [ -f "$path" ] && return 0
    echo "missing $path" >&2
    echo "build it first:" >&2
    echo "  python scripts/build_real_dataset.py --norad $NORAD --minutes $1 --windows $2" >&2
    exit 1
}

# need_synthetic <minutes> <n_sys> -- the matched GMAT source domain.
need_synthetic () {
    python - "$1" "$2" <<'PY' || exit 1
import sys
from tlv import datasets as ds
try:
    ds.load_synthetic(orbit="geo", minutes=int(sys.argv[1]), n_sys=int(sys.argv[2]))
except FileNotFoundError as e:
    sys.exit(f"{e}\n\nThe matched GEO source domain is a GMAT run, done by hand:\n"
             f"  cp configs/gmat/gmat_28089_asc.yaml ~/git/GMAT-Thrust-Data/config.yaml\n"
             f"  ( cd ~/git/GMAT-Thrust-Data && ./generateGMATNoThrustDataset.sh )\n"
             f"Branches (a) and (c) cannot run without it.")
PY
}

finish () {
    printf '\n%s\n%d run(s), %d failed\n' "----------------------------------------" "$RAN" "$FAILED"
    if [ "$FAILED" -ne 0 ]; then
        printf 'failed runs:%b\n' "$FAILED_LABELS"
        exit 1
    fi
    echo "figures: figures/$NORAD-*   rows: results/branch_*.csv"
    [ "$SUMMARIZE" = "1" ] && python scripts/summarize.py
    exit 0
}
