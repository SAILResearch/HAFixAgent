"""Resolve a BugsInPy `bugsinpy_run_test.sh` to a pytest-runnable module file.

BugsInPy bugs specify their failing test in one of two forms:

  pytest:    pytest tests/foo/test_bar.py::TestClass::test_method
  unittest:  python -m unittest -q a.b.test_mod.TestClass.test_method

FauxPy is a pytest plugin, so the SBFL FL stage runs the failing test's *module*
under pytest. For the pytest form the module file is explicit; for the unittest form
we must turn the dotted path (`a.b.test_mod.TestClass.test_method`) into a file path
(`a/b/test_mod.py`). The module/class boundary is ambiguous from the string alone, so
we resolve it against the checked-out repo: the longest dotted prefix that maps to an
existing `.py` file is the module; the remainder is the class/method selector.

Run from the repo root (cwd = the checked-out project). Prints the repo-relative
module path on success, nothing (exit 1) if no target can be resolved.

Usage (inside the container, cwd=repo):
    python /sbfl_py/resolve_test_target.py bugsinpy_run_test.sh
"""

import os
import re
import sys

# A path token ending in .py, not crossing a ':' (so `foo.py::test` -> `foo.py`).
_PYTEST_PATH = re.compile(r"([^\s:]+\.py)")
# A dotted identifier path with at least one dot (e.g. a.b.C.d).
_DOTTED = re.compile(r"[A-Za-z_][\w.]*\.[\w.]+")


def resolve(run_test_text: str, repo_root: str = ".") -> str:
    """Return a repo-relative pytest module path, or "" if none resolves.

    Lines are tried in order (mirroring the historical `head -1`): the first line
    that yields a runnable module wins.
    """
    for line in run_test_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Form 1/2 — explicit pytest .py path (with or without ::selector).
        m = _PYTEST_PATH.search(line)
        if m:
            return m.group(1)

        # Form 3 — unittest dotted path: probe the filesystem for the module file.
        if "unittest" in line:
            for dotted in _DOTTED.findall(line):
                toks = dotted.split(".")
                # Longest prefix that is an actual .py file is the module.
                for i in range(len(toks), 0, -1):
                    rel = os.path.join(*toks[:i]) + ".py"
                    if os.path.exists(os.path.join(repo_root, rel)):
                        return rel
    return ""


def main(argv):
    if len(argv) < 2:
        print("usage: resolve_test_target.py <run_test.sh> [repo_root]", file=sys.stderr)
        return 2
    repo_root = argv[2] if len(argv) > 2 else "."
    try:
        text = open(argv[1]).read()
    except OSError as e:
        print(f"cannot read {argv[1]}: {e}", file=sys.stderr)
        return 2
    target = resolve(text, repo_root)
    if not target:
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
