#!/usr/bin/env bash
# ===========================================================================
# regress.sh -- regression harness for the three LBM solvers.
#
# Runs a small battery of configurations and compares the measured outputs
# against committed baselines in regress_expected.txt. Baselines were
# captured once from a known-good build; a regression is detected when a
# value drifts outside tolerance.
#
# Catches: a wrong "fix" to collision/stream/BCs (numerical drift), a
# refactor that broke mass conservation, an off-by-one in geometry, a
# shadow bug in TRT, etc. The point is NOT to validate against a benchmark
# (that's lbm_bench's job) but to detect "something changed that shouldn't
# have" between commits.
#
# Usage:
#   ./regress.sh             # run and compare
#   ./regress.sh --baseline  # overwrite baselines (do only after a vetted change)
#
# Exit non-zero on any regression so CI / git-pre-commit can hook it.
# ===========================================================================
set -eu
cd "$(dirname "$0")"

MODE="${1:-check}"
BASE="regress_expected.txt"
OUT="/tmp/regress_now.txt"

# --- the test battery --------------------------------------------------
# Each line: label | binary | env | steps | parsers (regex -> awk expression)
# Keep this battery small enough to run in under a minute total.

run_case() {
    local label="$1" binary="$2" eenv="$3" steps="$4"
    local out
    out=$(eval "env LB_SELFTEST=1 LB_STEPS=$steps $eenv ./$binary" 2>&1)
    local meanux cdm status
    meanux=$(echo "$out" | awk '/mean ux/ {print $4; exit}')
    cdm=$(   echo "$out" | awk '/Cd \(mean/ {print $6; exit}')
    status=$(echo "$out" | awk '/status/ {print $3; exit}')
    [[ -z "$cdm" ]] && cdm="n/a"
    printf "%s mean_ux=%s cd=%s status=%s\n" "$label" "$meanux" "$cdm" "$status"
}

{
    # Short runs tuned for regression detection, not convergence:
    # the point is "same binary, same inputs -> same outputs" to the
    # stated tolerances, not reproducing benchmark values. Add more
    # cases if a particular code path regresses between commits.
    run_case "pro_m010_seed  " lbm_pro    "LB_MACH=0.10 LB_PERTURB=1" 500
    run_case "pro_m005_steady" lbm_pro    "LB_MACH=0.05 LB_PERTURB=0" 500
    run_case "tunnel_cyl_m010" lbm_tunnel "LB_MACH=0.10 LB_OBS=1 LB_GEOM=0" 500
} > "$OUT"

if [[ "$MODE" == "--baseline" ]]; then
    cp "$OUT" "$BASE"
    echo "wrote baseline: $BASE"
    cat "$BASE"
    exit 0
fi

if [[ ! -f "$BASE" ]]; then
    echo "no baseline ($BASE); run with --baseline first"
    cat "$OUT"
    exit 2
fi

# compare values with tolerances
python3 - <<'PY'
import re, sys, pathlib
base = {}
for line in pathlib.Path("regress_expected.txt").read_text().splitlines():
    m = re.match(r'(\S+)\s+mean_ux=(\S+)\s+cd=(\S+)\s+status=(\S+)', line)
    if m: base[m.group(1)] = (m.group(2), m.group(3), m.group(4))
now = {}
for line in pathlib.Path("/tmp/regress_now.txt").read_text().splitlines():
    m = re.match(r'(\S+)\s+mean_ux=(\S+)\s+cd=(\S+)\s+status=(\S+)', line)
    if m: now[m.group(1)] = (m.group(2), m.group(3), m.group(4))
TOL_UX, TOL_CD = 0.02, 0.10          # 2% on mean ux, 10% on Cd (noisy)
fails = []
for k, (ux0, cd0, st0) in base.items():
    if k not in now:
        fails.append(f"{k}: MISSING from current run"); continue
    ux1, cd1, st1 = now[k]
    if st0 != st1:
        fails.append(f"{k}: status {st0!r} -> {st1!r}")
    try:
        if abs(float(ux1) - float(ux0)) / max(abs(float(ux0)), 1e-6) > TOL_UX:
            fails.append(f"{k}: mean_ux {ux0} -> {ux1} ({TOL_UX*100:.0f}% tol)")
    except ValueError: pass
    try:
        if cd0 != "n/a" and cd1 != "n/a" and abs(float(cd1)-float(cd0))/max(abs(float(cd0)),1e-6) > TOL_CD:
            fails.append(f"{k}: Cd {cd0} -> {cd1} ({TOL_CD*100:.0f}% tol)")
    except ValueError: pass
if fails:
    print("REGRESSION:")
    for f in fails: print(" ", f)
    print()
    print("--- current ---"); print(pathlib.Path("/tmp/regress_now.txt").read_text())
    print("--- baseline ---"); print(pathlib.Path("regress_expected.txt").read_text())
    sys.exit(1)
else:
    print("OK: all", len(base), "regression cases match baseline within tolerance")
PY
