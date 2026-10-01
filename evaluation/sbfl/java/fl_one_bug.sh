#!/bin/bash
#
# Checkout + compile a single Defects4J bug, run GZoltar SBFL, and save the ranking
# plus metadata to /out. Intended to run inside a defects4j container, e.g.:
#
#   docker run --rm -v <repo>/evaluation/sbfl/java:/sbfl_java:ro \
#       -v <host_out_dir>:/out defects4j:latest \
#       bash /sbfl_java/fl_one_bug.sh <PROJECT> <BUGID> [relevant|modified]
#
# Writes to /out:
#   ochiai.ranking.csv   GZoltar Ochiai line-level ranking
#   meta.json            {project, bug_id, src_root, scope}
#   gzoltar.log          full GZoltar stdout/stderr
#
set -e

PROJECT="$1"
BUGID="$2"
SCOPE="${3:-relevant}"
[ -n "$PROJECT" ] && [ -n "$BUGID" ] || { echo "Usage: fl_one_bug.sh PROJECT BUGID [scope]" >&2; exit 2; }

# Match the RQ1 agent container env (hafix_agent/environments/defects4j_docker.py) so
# the FL stage compiles bugs identically to the repair stage. JAVA_HOME/PATH are also in
# the image ENV; set explicitly here for an interactive-shell-independent, documented match.
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export DEFECTS4J_HOME=/defects4j
export PATH=/defects4j/framework/bin:$JAVA_HOME/bin:$PATH
export _JAVA_OPTIONS="-Xmx4g -XX:MaxPermSize=512m"
# Locale fix: the image sets LANG=en_US.UTF-8 but never generates it, so glibc falls back
# to US-ASCII and `javac` fails on non-ASCII test files (Gson/JacksonXml/JxPath). The agent
# avoids this via targeted relevant-test compilation; our full `defects4j compile` needs a
# real UTF-8 locale. C.UTF-8 is glibc's built-in UTF-8 (always present); coverage/rankings
# are locale-independent, so this only enables compilation — it does not change results.
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="/tmp/d4j_${PROJECT}_${BUGID}"

rm -rf "$WORK"
defects4j checkout -p "$PROJECT" -v "${BUGID}b" -w "$WORK" >/dev/null 2>&1
cd "$WORK"
defects4j compile >/dev/null 2>&1

bash "$HERE/run_gzoltar.sh" \
  --work-dir "$WORK" --jars "$HERE/lib" --output /out --scope "$SCOPE" \
  >/out/gzoltar.log 2>&1

SRC_ROOT=$(defects4j export -p dir.src.classes 2>/dev/null)
printf '{"project":"%s","bug_id":"%s","src_root":"%s","scope":"%s"}\n' \
  "$PROJECT" "$BUGID" "$SRC_ROOT" "$SCOPE" > /out/meta.json

# Save the developer patch (ground truth) for offline FL-accuracy analysis only.
# This is NEVER fed to the agent — used solely to check whether SBFL's top-N
# contains a developer-modified line.
PATCH="/defects4j/framework/projects/${PROJECT}/patches/${BUGID}.src.patch"
[ -f "$PATCH" ] && cp "$PATCH" /out/dev.src.patch || true

echo "OK ${PROJECT}_${BUGID} src_root=${SRC_ROOT} lines=$(tail -n +2 /out/ochiai.ranking.csv | wc -l)"
