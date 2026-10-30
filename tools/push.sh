#!/usr/bin/env bash
# ===========================================================================
# tools/push.sh -- initialise the repo, push to GitHub, enable Pages.
#
# Idempotent: safe to run on an already-initialised / already-pushed tree.
# Uses the `gh` CLI (https://cli.github.com); must be authenticated:
#     gh auth login
#
# Usage:
#     ./tools/push.sh                            # default repo name/visibility
#     REPO=wind-tunnel-2d VISIBILITY=public ./tools/push.sh
#     REMOTE=origin BRANCH=main ./tools/push.sh
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-wind-tunnel-2d}"
VISIBILITY="${VISIBILITY:-public}"    # public | private
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"

# ---- preflight ------------------------------------------------------------
command -v gh  >/dev/null || { echo "gh CLI not found; install from https://cli.github.com"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "run: gh auth login"; exit 1; }

OWNER=$(gh api user --jq .login)
FULL="$OWNER/$REPO"
echo "target: $FULL  (visibility: $VISIBILITY)"

# ---- git init if needed ---------------------------------------------------
if [ ! -d .git ]; then
    git init -b "$BRANCH"
    echo "initialised repo on branch $BRANCH"
fi

# make sure we're on the target branch
cur=$(git branch --show-current || echo "")
if [ -z "$cur" ]; then
    git checkout -b "$BRANCH"
elif [ "$cur" != "$BRANCH" ]; then
    git checkout -B "$BRANCH"
fi

# ---- make sure the GitHub repo exists ------------------------------------
if gh repo view "$FULL" >/dev/null 2>&1; then
    echo "GitHub repo $FULL already exists"
else
    gh repo create "$FULL" --"$VISIBILITY" \
        --description "2D wind tunnel: MD + LBM solvers, benchmark-validated" \
        --source=. --remote="$REMOTE" --push=false
    echo "created $FULL"
fi

# ---- remote wiring --------------------------------------------------------
URL=$(gh repo view "$FULL" --json sshUrl --jq .sshUrl)
if git remote get-url "$REMOTE" >/dev/null 2>&1; then
    git remote set-url "$REMOTE" "$URL"
else
    git remote add "$REMOTE" "$URL"
fi

# ---- stage + commit if there are changes ---------------------------------
git add -A
if git diff --cached --quiet; then
    echo "no changes to commit"
else
    git commit -m "$(cat <<'EOF'
wind-tunnel-2d: initial public release

Four solvers for the same 2D tunnel:
- wind_tunnel  : particle MD (WCA pseudo-hard-sphere + Verlet lists + cell list)
- lbm_tunnel   : D2Q9-BGK pedagogical ("where BGK breaks")
- lbm_bench    : headless Schaefer-Turek 2D-2 validator (Cd=3.22, St=0.30 at res=40)
- lbm_pro      : validated TRT + Bouzidi + Guo BCs with interactive window

Includes convergence harness (Richardson), regression harness, OpenMP
scaling tool, man pages, install/uninstall, and WASM/Pages deploy.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015UYsL9gnanLXjGqqDArNjr
EOF
)"
fi

# ---- push -----------------------------------------------------------------
git push -u "$REMOTE" "$BRANCH"
echo "pushed to $FULL ($BRANCH)"

# ---- enable Pages (GitHub Actions source) --------------------------------
# This flips the Pages source to the Actions workflow so pages.yml drives
# deployment. Idempotent: no-op if already set.
echo "enabling GitHub Pages (source: GitHub Actions)..."
gh api -X POST "/repos/$FULL/pages" \
    -H "Accept: application/vnd.github+json" \
    -f "build_type=workflow" \
    >/dev/null 2>&1 \
  || gh api -X PUT "/repos/$FULL/pages" \
       -H "Accept: application/vnd.github+json" \
       -f "build_type=workflow" \
       >/dev/null 2>&1 \
  || echo "  (Pages may already be enabled; check https://github.com/$FULL/settings/pages)"

echo ""
echo "done."
echo "  repo  : https://github.com/$FULL"
echo "  Pages : https://$OWNER.github.io/$REPO/"
echo "          (first build triggers on this push; takes 2-5 min)"
echo "  check : gh run watch -R $FULL"
