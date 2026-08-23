#!/usr/bin/env bash
# One command that answers "is this actually good?" -- the gate every agent and
# every push goes through.
#
# It exists because green tests have hidden every serious bug in this project:
# the 4x-too-dark colour space, the export rendering the wrong grade, the 0-byte
# GIF, the dead undo, and most recently three CI-red commits nobody noticed. So
# this checks the things a test run does NOT: that the identity LUT still hashes
# the same, that the base install does not secretly need an optional extra, that
# the page and the server agree, and that CI is green for the commit you are on.
#
#   scripts/check.sh          fast gate: tests + invariants
#   scripts/check.sh --ci     also ask GitHub about the pushed commit
#   scripts/check.sh --live   also boot a server on a scratch root and grade
#
# Exit code is the number of failed checks, so a caller can branch on it.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 99

FAILED=0
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAILED=$((FAILED + 1)); }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

step "tools"
for t in uv ffmpeg ffprobe; do
    if command -v "$t" >/dev/null 2>&1; then pass "$t"; else fail "$t not on PATH"; fi
done
command -v node >/dev/null 2>&1 && pass "node (page harness will run)" \
    || printf '  \033[33mskip\033[0m the page harness needs node\n'

step "tests"
if uv run pytest -q 2>&1 | tail -3 | tee /tmp/ragvid-check-tests.txt | grep -qE "^[0-9]+ passed"; then
    pass "$(grep -oE '[0-9]+ passed.*' /tmp/ragvid-check-tests.txt | head -1)"
else
    fail "pytest -- $(grep -oE '[0-9]+ failed.*' /tmp/ragvid-check-tests.txt | head -1)"
fi

step "invariants"

# The identity LUT hash. spec.py pins it; if this drifts, every saved grade
# shifted underneath the people who saved them.
uv run python - <<'PY' && pass "identity LUT hashes unchanged" || fail "identity LUT hash DRIFTED"
import hashlib, sys
import numpy as np
from ragvid.spec import GradeSpec
from ragvid.lut import _grid
want = "517467be3ba6b7a8afe71a05c847061dc597f0ea92e41b422164b579fbc74291"
got = hashlib.sha256(np.ascontiguousarray(GradeSpec.identity().apply(_grid(33)))).hexdigest()
sys.exit(0 if got == want else 1)
PY

# A flat grade must stay byte-identical to a bare GradeSpec, or GradeStack
# quietly changed the output of every grade that never asked for a region.
uv run python - <<'PY' && pass "a flat stack equals a bare spec" || fail "flat stack DIVERGED from bare spec"
import sys
import numpy as np
from ragvid.spec import GradeSpec
from ragvid.region import GradeStack
s = GradeSpec(saturation=1.2, contrast=0.3, temperature=800.0)
# An IMAGE, not a LUT grid: GradeStack.apply is stricter than GradeSpec.apply
# by design, because a mask has to have somewhere to be.
img = np.linspace(0, 1, 64 * 48 * 3).reshape(48, 64, 3)
sys.exit(0 if np.array_equal(s.apply(img), GradeStack(base=s).apply(img)) else 1)
PY

# The base install must not need an optional extra. segment.py imports
# onnxruntime lazily; this is the check that keeps it lazy.
uv run python -c "
import sys, ragvid.segment, ragvid.project, ragvid.server
sys.exit(1 if 'onnxruntime' in sys.modules else 0)
" && pass "importing ragvid does not load onnxruntime" \
  || fail "onnxruntime is imported at module scope -- the base install is broken"

# The page and the server must agree, or a stale page silently drops fields.
uv run python - <<'PY' && pass "API_VERSION matches EXPECTED_API" || fail "API_VERSION / EXPECTED_API DISAGREE"
import re, sys, pathlib
from ragvid.server import API_VERSION
page = pathlib.Path("ragvid/web/index.html").read_text()
m = re.search(r"EXPECTED_API\s*=\s*(\d+)", page)
sys.exit(0 if m and int(m.group(1)) == API_VERSION else 1)
PY

# No key may ever reach a tracked file.
if git grep -nIE "(sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,})" -- . >/dev/null 2>&1; then
    fail "something that looks like an API key is tracked in git"
else
    pass "no API key in tracked files"
fi

if [ "${1:-}" = "--ci" ] || [ "${2:-}" = "--ci" ]; then
    step "ci"
    if command -v gh >/dev/null 2>&1; then
        sha=$(git rev-parse HEAD)
        concl=$(gh run list --limit 20 --json headSha,conclusion,status \
                -q "[.[] | select(.headSha==\"$sha\")] | .[0].conclusion // \"pending\"" 2>/dev/null)
        case "$concl" in
            success) pass "CI green for $(git rev-parse --short HEAD)" ;;
            pending|"") printf '  \033[33mskip\033[0m no finished CI run for this commit yet\n' ;;
            *) fail "CI is $concl for $(git rev-parse --short HEAD)" ;;
        esac
    else
        printf '  \033[33mskip\033[0m gh not installed\n'
    fi
fi

if [ "${1:-}" = "--live" ] || [ "${2:-}" = "--live" ]; then
    step "live"
    # Isolated: never the real session or settings. Forgetting this once cost a
    # real one.
    root=$(mktemp -d)
    XDG_DATA_HOME="$root/data" uv run ragvid serve --no-browser --port 8799 --root "$root" &
    pid=$!
    for _ in $(seq 1 40); do
        curl -sf localhost:8799/api/state >/dev/null 2>&1 && break
        sleep 0.5
    done
    if curl -sf localhost:8799/api/state >/dev/null 2>&1; then
        pass "server answers on 127.0.0.1:8799"
        clip=$(ls test_files/*.mp4 2>/dev/null | head -1)
        if [ -n "$clip" ] && curl -sf -X POST localhost:8799/api/project \
             -H 'Content-Type: application/json' \
             -d "{\"path\":\"$PWD/$clip\"}" -m 120 >/dev/null; then
            pass "opened $clip"
        else
            fail "could not open a clip"
        fi
    else
        fail "server never answered"
    fi
    kill "$pid" 2>/dev/null
    rm -rf "$root"
fi

step "result"
if [ "$FAILED" -eq 0 ]; then
    printf '  \033[32mall checks passed\033[0m\n\n'
else
    printf '  \033[31m%d check(s) failed\033[0m\n\n' "$FAILED"
fi
exit "$FAILED"
