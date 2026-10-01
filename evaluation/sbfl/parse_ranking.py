"""Unify SBFL ranking output into a top-N suspicious-line list (dataset-agnostic).

Supports GZoltar (Java/Defects4J) and FauxPy (Python/BugsInPy). We run the tools
ourselves and parse their output; we depend on neither vendor/ITER nor the FauxPy
replication package at runtime (their cached outputs are used only as format references
/ test fixtures).

GZoltar line-granularity entity format (one entry per covered line):

    <package>$<TopClass>[$Nested...]#<method>(<argTypes>):<line>;<suspiciousness>

e.g. ``com.google.javascript.jscomp$Compiler#<clinit>():70;1.0``
maps to source file ``com/google/javascript/jscomp/Compiler.java`` at line 70.
The first ``$`` separates the package from the top-level class; any further ``$``
denote nested/anonymous classes, which still live in the top-level class's file.

FauxPy statement-granularity output (``Scores_<technique>.csv``, e.g. Scores_Ochiai.csv):

    Entity,Score
    <abs_path>/pkg/mod.py::<line>,<suspiciousness>

The entity path is absolute (where FauxPy ran under ``--src``); stripping the ``--src``
prefix yields the project-relative source path. Rows are score-descending.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SuspiciousLine:
    """One ranked suspicious source line."""
    file: str              # source path (repo-relative if src_root given, else package-relative)
    line: int
    suspiciousness: float
    rank: int              # 1-based rank in the (deduped) ranking
    qualified_name: str    # original tool entity name, kept for debugging/traceability


_GZOLTAR_LINE_RE = re.compile(r":(\d+)\s*$")


def parse_gzoltar_name(name: str) -> Optional[Tuple[str, int]]:
    """Parse a GZoltar entity name into (package_relative_java_path, line).

    Returns None if the name does not match the expected ``...#...:line`` shape.
    """
    if "#" not in name:
        return None
    class_part, member_part = name.split("#", 1)

    m = _GZOLTAR_LINE_RE.search(member_part)
    if not m:
        return None
    line = int(m.group(1))

    # Package is everything before the first '$'; the class binary name follows.
    if "$" in class_part:
        pkg, cls = class_part.split("$", 1)
    else:
        pkg, cls = "", class_part
    top_class = cls.split("$", 1)[0]  # strip nested/anonymous class suffixes
    if not top_class:
        return None

    pkg_path = pkg.replace(".", "/")
    rel = f"{pkg_path}/{top_class}.java" if pkg_path else f"{top_class}.java"
    return rel, line


def parse_gzoltar_ranking(
    csv_path: str,
    src_root: str = "",
    top_n: Optional[int] = None,
) -> List[SuspiciousLine]:
    """Parse a GZoltar ``ochiai.ranking.csv`` into ranked SuspiciousLine list.

    Preserves GZoltar's own ranking order (the file is already sorted by
    suspiciousness with GZoltar's tie-breaking); we only deduplicate repeated
    (file, line) pairs, keeping the first (highest-ranked) occurrence.

    Args:
        csv_path: path to ``ochiai.ranking.csv``.
        src_root: source root to prepend (e.g. ``defects4j export -p dir.src.classes``
                  output like ``src``); if empty, paths are package-relative.
        top_n: keep only the first N unique lines (None = all).
    """
    seen = set()
    result: List[SuspiciousLine] = []
    rank = 0

    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) < 2:
                continue
            name, susp = row[0], row[1]
            if name == "name":  # header
                continue
            parsed = parse_gzoltar_name(name)
            if parsed is None:
                continue
            try:
                susp_v = float(susp)
            except ValueError:
                continue
            rel, line = parsed
            path = f"{src_root.rstrip('/')}/{rel}" if src_root else rel
            key = (path, line)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            result.append(SuspiciousLine(path, line, susp_v, rank, name))
            if top_n is not None and len(result) >= top_n:
                break

    return result


def _strip_src_root(path: str, src_root: str) -> str:
    """Make an absolute FauxPy entity path project-relative by stripping src_root."""
    if not src_root:
        return path
    sr = src_root.rstrip("/")
    if path.startswith(sr + "/"):
        return path[len(sr) + 1:]
    if path.startswith(sr):
        return path[len(sr):].lstrip("/")
    return path


def parse_fauxpy_ranking(
    csv_path: str,
    src_root: str = "",
    top_n: Optional[int] = None,
) -> List[SuspiciousLine]:
    """Parse a FauxPy ``Scores_<technique>.csv`` into ranked SuspiciousLine list.

    Format: header ``Entity,Score``; rows ``<abs_path>/mod.py::<line>,<score>``.
    Sorts by suspiciousness descending (FauxPy already emits this order; we sort
    defensively, stable so ties keep file order), deduplicates repeated (file, line)
    keeping the highest-scored occurrence, and strips ``src_root`` to yield
    project-relative paths.

    Args:
        csv_path: path to ``Scores_Ochiai.csv`` (or another technique).
        src_root: the ``--src`` value FauxPy was run with; stripped from entity paths.
        top_n: keep only the first N unique lines (None = all).
    """
    rows: List[tuple] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)  # comma-delimited
        for row in reader:
            if len(row) < 2:
                continue
            entity, score = row[0], row[1]
            if entity == "Entity":  # header
                continue
            if "::" not in entity:
                continue
            path_part, line_part = entity.rsplit("::", 1)
            try:
                line = int(line_part)
                susp_v = float(score)
            except ValueError:
                continue
            rel = _strip_src_root(path_part, src_root)
            rows.append((susp_v, rel, line, entity))

    # Stable sort by descending suspiciousness (preserves emit order within ties).
    rows.sort(key=lambda r: -r[0])

    seen = set()
    result: List[SuspiciousLine] = []
    rank = 0
    for susp_v, rel, line, entity in rows:
        key = (rel, line)
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        result.append(SuspiciousLine(rel, line, susp_v, rank, entity))
        if top_n is not None and len(result) >= top_n:
            break
    return result
