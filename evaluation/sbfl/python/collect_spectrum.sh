#!/bin/bash
#
# Decoupled SBFL collector (BugsInPy). Runs INSIDE a prewarmed bugsinpy:{p}_{b} container
# and emits only DATA — no FauxPy, no ranking logic. The Ochiai ranking is computed
# afterwards on the host by compute_ochiai.py from these portable artifacts:
#
#   /out/coverage.json  : `coverage json --show-contexts` (per-test line coverage)
#   /out/outcomes.tsv   : per-test pass/fail (via _sbfl_outcome_plugin.py)
#   /out/meta.json      : provenance (src_root, python, coverage version, provision_mode)
#
# Only dependency added to the bug env is `coverage` (pure-Python, supports 3.6-3.12) —
# far more compatible than FauxPy's pytest-plugin coupling, so it reproduces many bugs
# FauxPy cannot. Environment provisioning mirrors run_fauxpy.sh (declared requirements +
# pre-release pin relaxation + editable install) so heavy-dep bugs build their real env.
#
# Usage (inside container):  collect_spectrum.sh <PROJECT> <BUGID>
set +e

PROJECT="$1"; BUGID="$2"
[ -n "$PROJECT" ] && [ -n "$BUGID" ] || { echo "Usage: collect_spectrum.sh PROJECT BUGID" >&2; exit 2; }

REPO="/BugsInPy/framework/bin/temp/${PROJECT}_${BUGID}/${PROJECT}"
[ -d "$REPO" ] || { echo "repo not found at $REPO" >&2; exit 1; }
cd "$REPO"
[ -f env/bin/activate ] || { echo "virtualenv not found at $REPO/env" >&2; exit 1; }
source env/bin/activate

# --- Provision the bug's declared environment (same policy as run_fauxpy.sh) ----------
pip install --quiet --upgrade pip setuptools wheel >/out/collect_install.log 2>&1 || true
PROVISION_MODE="none"
if [ -f bugsinpy_requirements.txt ]; then
  if pip install -r bugsinpy_requirements.txt >>/out/collect_install.log 2>&1; then
    PROVISION_MODE="exact"
  else
    sed -E 's/(==[0-9]+(\.[0-9]+)*)((rc|a|b)[0-9]+|\.dev[0-9]+)/\1/g' \
      bugsinpy_requirements.txt > /tmp/req_relaxed.txt
    if pip install -r /tmp/req_relaxed.txt >>/out/collect_install.log 2>&1; then
      PROVISION_MODE="relaxed_prerelease"
    else
      PROVISION_MODE="partial"
    fi
  fi
fi
{ [ -f setup.py ] || [ -f pyproject.toml ]; } && pip install -e . >>/out/collect_install.log 2>&1 || true

# --- Add the lightweight collector deps (coverage is the only hard requirement) -------
# coverage<6 supports Python 3.6; pytest-timeout is best-effort (outer `timeout` is the
# real hang guard). Neither perturbs the bug's own pytest/test dependencies.
# `coverage` is the ONLY hard dependency (coverage<6 supports Python 3.6). We start it via
# `coverage run` so it is active before any source import (pytest-cov's --cov starts late
# and silently records nothing for packages imported during bootstrap). pytest-timeout is
# best-effort (outer `timeout` is the real hang guard).
pip install --quiet 'coverage<6' >>/out/collect_install.log 2>&1 || \
  pip install --quiet coverage >>/out/collect_install.log 2>&1
pip install --quiet pytest-timeout >>/out/collect_install.log 2>&1 || true
COV_VER=$(pip show coverage 2>/dev/null | awk '/^Version:/{print $2}')

# --- Resolve the failing test's module + the source package ---------------------------
TEST_FILE=$(python /sbfl_py/resolve_test_target.py bugsinpy_run_test.sh "$REPO")
[ -n "$TEST_FILE" ] || { echo "could not resolve test target" >&2; exit 1; }
SRC=""
for d in */; do d="${d%/}"
  case "$d" in env|venv|tests|test|testing|docs|doc|examples|example|build|dist|.git) continue;; esac
  [ -f "$d/__init__.py" ] && { SRC="$d"; break; }
done
if [ -z "$SRC" ]; then
  for parent in src lib; do
    [ -d "$parent" ] || continue
    for d in "$parent"/*/; do d="${d%/}"; [ -f "$d/__init__.py" ] && { SRC="$d"; break; }; done
    [ -n "$SRC" ] && break
  done
fi
[ -n "$SRC" ] || SRC="."
echo "[collect] test_file=$TEST_FILE src=$SRC coverage=$COV_VER provision=$PROVISION_MODE"

# --- Collect per-test coverage (dynamic test contexts) + outcomes ---------------------
: > /out/outcomes.tsv
export SBFL_OUTCOMES=/out/outcomes.tsv
export PYTHONPATH="/sbfl_py:${PYTHONPATH}"
TIMEOUT_ARGS=""
python -c "import pytest_timeout" 2>/dev/null && TIMEOUT_ARGS="--timeout=120 --timeout-method=thread"
rm -f .coverage

# `coverage run` starts measurement before any import; _sbfl_context_plugin switches the
# coverage context to each test's nodeid (matching /out/outcomes.tsv) and records outcomes.
[ "$SRC" = "." ] && SRC_ABS="$REPO" || SRC_ABS="$REPO/$SRC"
timeout --kill-after=15 600 coverage run --source="$SRC_ABS" -m pytest "$TEST_FILE" \
  -o addopts="" -p no:cacheprovider -p _sbfl_context_plugin $TIMEOUT_ARGS \
  >/out/pytest.log 2>&1
echo "[collect] pytest exit=$? outcomes=$(wc -l < /out/outcomes.tsv 2>/dev/null || echo 0)"

# --- Export the portable coverage spectrum (cwd=$REPO holds the .coverage file) --------
coverage json --show-contexts -o /out/coverage.json >/dev/null 2>&1
[ -s /out/coverage.json ] || { echo "[collect] no coverage.json produced (see /out/pytest.log)" >&2; exit 1; }
NOUT=$(wc -l < /out/outcomes.tsv 2>/dev/null || echo 0)
[ "$NOUT" -gt 0 ] || { echo "[collect] no test outcomes recorded" >&2; exit 1; }

PYVER=$(grep -hoE 'python_version="[^"]+"' bugsinpy_bug.info 2>/dev/null | head -1 | sed -E 's/.*"(.*)"/\1/')
FAILS=$(awk -F'\t' '$2=="failed"||$2=="error"{print $1}' /out/outcomes.tsv | sort -u | wc -l)
printf '{"project":"%s","bug_id":"%s","src_root":"%s","src_pkg":"%s","engine":"coverage","granularity":"line","python":"%s","coverage":"%s","provision_mode":"%s","test_file":"%s","n_failing":%s}\n' \
  "$PROJECT" "$BUGID" "$REPO" "$SRC" "$PYVER" "$COV_VER" "$PROVISION_MODE" "$TEST_FILE" "$FAILS" > /out/meta.json

echo "OK ${PROJECT}_${BUGID} failing_tests=${FAILS} coverage_files=$(grep -c '"executed_lines"' /out/coverage.json 2>/dev/null || echo '?')"
