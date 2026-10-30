#!/usr/bin/env bash
# ===========================================================================
# make-and-push.sh -- build everything natively, verify, push to GitHub,
# enable Pages (which triggers the WASM build in CI).
#
# Idempotent: safe to re-run. Each step prints what it did and skips if
# already satisfied.
#
# Prerequisites on your machine:
#   - gcc/clang + make
#   - SDL2 development headers (libsdl2-dev on Debian/Ubuntu;
#     `brew install sdl2` on macOS, which also needs `brew install libomp`)
#   - gh CLI, authenticated:  gh auth login
#   - git
#
# NOT required locally (CI handles it):
#   - emscripten (the WASM build runs in GitHub Actions on push)
#
# Usage:
#   ./make-and-push.sh                     # defaults
#   REPO=wind-tunnel-2d VISIBILITY=public ./make-and-push.sh
#   SKIP_PUSH=1 ./make-and-push.sh         # build + test only, no git
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

say() { printf "\n== %s ==\n" "$*"; }

# ---- 1. preflight ---------------------------------------------------------
say "preflight"
command -v make >/dev/null || { echo "ERROR: make not found"; exit 1; }
command -v cc   >/dev/null || { echo "ERROR: cc not found";   exit 1; }
if [ "${SKIP_PUSH:-0}" != "1" ]; then
    command -v gh  >/dev/null || { echo "ERROR: gh CLI not found; install from https://cli.github.com"; exit 1; }
    command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
    gh auth status >/dev/null 2>&1 || { echo "ERROR: run 'gh auth login' first"; exit 1; }
fi
echo "OK: toolchain present"

# ---- 2. build all four binaries (zero warnings expected) -----------------
say "build (zero warnings expected)"
make clean >/dev/null
if ! make -j 2>&1 | tee /tmp/build.log | grep -E '^(cc|clang|gcc)'; then
    cat /tmp/build.log; exit 1
fi
if grep -iE 'error|warning' /tmp/build.log; then
    echo "ERROR: compile output includes error/warning"; exit 1
fi
ls -la wind_tunnel lbm_tunnel lbm_pro lbm_bench

# ---- 3. headless smoke tests on each binary ------------------------------
say "smoke tests (headless)"
echo "-- lbm_pro Mach=0.10, 500 steps:"
LB_SELFTEST=1 LB_STEPS=500 LB_MACH=0.10 LB_PERTURB=1 ./lbm_pro \
    | grep -E 'mean ux|Cd \(mean|status'

echo "-- lbm_tunnel Mach=0.10, 500 steps:"
LB_SELFTEST=1 LB_STEPS=500 LB_MACH=0.10 LB_GEOM=0 LB_OBS=1 ./lbm_tunnel \
    | grep -E 'mean ux|status'

echo "-- wind_tunnel 400 steps:"
WT_SELFTEST=1 WT_STEPS=400 ./wind_tunnel | grep -E 'mean v_x|status' || true

# lbm_bench is too long for a smoke test; `make lbm-regress` covers it
# for CI-grade checks. Run it only if you ask:
if [ "${RUN_BENCH:-0}" = "1" ]; then
    echo "-- lbm_bench 20 (full validator run):"
    rm -f /tmp/lbm_ckpt_20.bin
    ./lbm_bench 20 | tail -8
fi

# ---- 4. regression harness (optional but recommended) --------------------
if [ "${SKIP_REGRESS:-0}" != "1" ]; then
    say "regression harness"
    if [ ! -f regress_expected.txt ]; then
        echo "no baseline yet; capturing it now"
        ./regress.sh --baseline | tail -5
    else
        ./regress.sh | tail -3
    fi
fi

# ---- 5. push to GitHub + enable Pages ------------------------------------
if [ "${SKIP_PUSH:-0}" = "1" ]; then
    say "SKIP_PUSH=1 -- skipping git/push"
    echo "build complete; run again without SKIP_PUSH to publish."
    exit 0
fi

say "push to GitHub"
./tools/push.sh

# ---- 6. show the result ---------------------------------------------------
say "done"
OWNER=$(gh api user --jq .login)
REPO="${REPO:-wind-tunnel-2d}"
cat <<EOF

  repo    : https://github.com/$OWNER/$REPO
  Pages   : https://$OWNER.github.io/$REPO/
  CI      : gh run watch -R $OWNER/$REPO  (first WASM build takes 2-5 min)
  install : make install    (into ~/.local)
  man     : man lbm_pro

Local demos: ./lbm_tunnel  ./lbm_pro  ./wind_tunnel
Validator  : ./lbm_bench 20 40        (~5 min on 6 cores)
EOF
