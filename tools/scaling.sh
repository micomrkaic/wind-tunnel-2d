#!/usr/bin/env bash
# tools/scaling.sh -- measure OpenMP scaling of the three LBM solvers.
#
# Measures the walltime of a fixed, bounded run at different
# OMP_NUM_THREADS values and prints speedup vs 1-thread. Honest reading:
# these solvers are memory-bandwidth-bound at modest grid sizes, so
# near-linear scaling is realistic only up to the point where the inner
# f-array fits in fewer caches. On a 2-core sandbox this prints 1x and
# 2x only; on an M4 run with THREADS="1 2 4 8" to find where bandwidth
# saturates.
#
# Usage:
#   ./tools/scaling.sh                        # defaults
#   THREADS="1 2 4 8" STEPS=400 ./tools/scaling.sh
set -eu
cd "$(dirname "$0")/.."

THREADS="${THREADS:-1 2}"
STEPS="${STEPS:-400}"
RES="${RES:-20}"
MACH="${MACH:-0.10}"

timeit() {
    { time -p "$@" >/dev/null 2>&1; } 2>&1 | awk '/^real/{print $2}'
}

printf "scaling: STEPS=%s RES=%s MACH=%s THREADS=%s\n\n" \
       "$STEPS" "$RES" "$MACH" "$THREADS"

for solver in lbm_pro lbm_bench; do
    printf "  --- %s ---\n" "$solver"
    base=""
    for n in $THREADS; do
        if [ "$solver" = "lbm_pro" ]; then
            wall=$(OMP_NUM_THREADS=$n timeit env LB_SELFTEST=1 \
                     LB_STEPS=$STEPS LB_MACH=$MACH ./lbm_pro)
        else
            rm -f /tmp/lbm_ckpt_${RES}.bin
            wall=$(OMP_NUM_THREADS=$n timeit ./lbm_bench $RES steps=$STEPS)
        fi
        if [ -z "$base" ]; then base="$wall"; fi
        speedup=$(awk -v a="$base" -v b="$wall" \
                      'BEGIN{if(b>0)printf"%.2f",a/b;else print"?"}')
        printf "    %2d thr  real=%6.2fs  speedup=%sx\n" \
               "$n" "$wall" "$speedup"
    done
done
