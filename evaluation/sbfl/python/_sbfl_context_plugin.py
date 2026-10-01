"""pytest plugin for the decoupled collector: per-test coverage context + outcomes.

Used with `coverage run -m pytest -p _sbfl_context_plugin` (coverage is started by
`coverage run` BEFORE any source import, avoiding pytest-cov's "No data was collected"
import-timing failure). For each test we:

  * switch coverage's dynamic context to the test's nodeid (so coverage attributes the
    test's executed lines to that test) — keyed identically to the outcomes below, so
    compute_ochiai.py joins them exactly;
  * record pass/fail to $SBFL_OUTCOMES (a test is failing if any phase failed/errored).

Hooks are stable across pytest 3.x–7.x. Only dependency is `coverage` itself.
"""

import os

_OUT = os.environ.get("SBFL_OUTCOMES", "/out/outcomes.tsv")


def _coverage():
    try:
        import coverage
        return coverage.Coverage.current()
    except Exception:
        return None


def pytest_runtest_logstart(nodeid, location):
    # Fires before setup; attribute this test's whole protocol to its nodeid context.
    cov = _coverage()
    if cov is not None:
        try:
            cov.switch_context(nodeid)
        except Exception:
            pass


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        try:
            with open(_OUT, "a") as f:
                f.write(f"{report.nodeid}\t{report.outcome}\n")
        except OSError:
            pass
