#!/bin/bash
#
# Run GZoltar SBFL (Ochiai, line granularity) on a checked-out + compiled
# Defects4J bug, producing ochiai.ranking.csv. Self-contained: uses the bundled
# GZoltar v1.7.3 jars and `defects4j export` (no per-project build hacks, no ITER).
#
# Must run inside a defects4j container where the bug is already checked out and
# compiled (`defects4j checkout` + `defects4j compile`).
#
# Usage:
#   run_gzoltar.sh --work-dir DIR --jars DIR --output DIR [--scope relevant|modified]
#
#   --scope relevant : instrument all test-covered classes (classes.relevant).
#                      Realistic FL — used for RQ3. (default)
#   --scope modified : instrument only the modified class (classes.modified).
#                      Leaky; used ONLY to reproduce/validate against ITER fixtures.
#
set -euo pipefail

WORK_DIR=""
JARS=""
OUTPUT=""
SCOPE="relevant"

while [ "${1:-}" != "" ]; do
  case "$1" in
    --work-dir) WORK_DIR="$2"; shift 2;;
    --jars)     JARS="$2"; shift 2;;
    --output)   OUTPUT="$2"; shift 2;;
    --scope)    SCOPE="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

[ -n "$WORK_DIR" ] && [ -n "$JARS" ] && [ -n "$OUTPUT" ] || {
  echo "Usage: run_gzoltar.sh --work-dir DIR --jars DIR --output DIR [--scope relevant|modified]" >&2
  exit 2
}

CLI_JAR="$JARS/gzoltarcli.jar"
AGENT_JAR="$JARS/gzoltaragent.jar"
[ -s "$CLI_JAR" ]   || { echo "Missing $CLI_JAR" >&2; exit 2; }
[ -s "$AGENT_JAR" ] || { echo "Missing $AGENT_JAR" >&2; exit 2; }

cd "$WORK_DIR"

# --- Defects4J-provided directories, classpath, and scoping -------------------
BIN_CLASSES=$(defects4j export -p dir.bin.classes 2>/dev/null)
BIN_TESTS=$(defects4j export -p dir.bin.tests 2>/dev/null)
CP_TEST=$(defects4j export -p cp.test 2>/dev/null)
REL_TESTS=$(defects4j export -p tests.relevant 2>/dev/null)

if [ "$SCOPE" = "modified" ]; then
  APP_CLASSES=$(defects4j export -p classes.modified 2>/dev/null)
else
  APP_CLASSES=$(defects4j export -p classes.relevant 2>/dev/null)
fi

ABS_BIN_CLASSES="$WORK_DIR/$BIN_CLASSES"
ABS_BIN_TESTS="$WORK_DIR/$BIN_TESTS"
# Pin GZoltar's expected JUnit 4.12 + hamcrest FIRST so they shadow any incompatible
# JUnit a project ships in cp.test (e.g. Jsoup/Closure/Mockito trigger
# NoSuchMethodError: Description.getMethodName() otherwise). Matches ITER's classpath order.
JUNIT_JAR="$JARS/junit-4.12.jar"
HAMCREST_JAR="$JARS/hamcrest-core-1.3.jar"
CP="$JUNIT_JAR:$HAMCREST_JAR:$ABS_BIN_CLASSES:$ABS_BIN_TESTS:$CP_TEST:$CLI_JAR"

# GZoltar include lists are ':'-separated globs (matching ITER's working v1.7.3 usage).
# Application classes: exact FQNs. Test classes: FQN#* to select all their methods.
INCLUDES=$(echo $APP_CLASSES | sed 's/ /:/g')
TEST_INCLUDES=$(echo $REL_TESTS | sed 's/[^ ][^ ]*/&#*/g; s/ /:/g')

echo "[gzoltar] scope=$SCOPE  app-classes=$(echo $APP_CLASSES | wc -w)  rel-tests=$(echo $REL_TESTS | wc -w)"

# --- 1. Collect the test methods to run ---------------------------------------
java -cp "$CP" com.gzoltar.cli.Main listTestMethods "$ABS_BIN_TESTS" \
  --outputFile "$WORK_DIR/tests.txt" \
  --includes "$TEST_INCLUDES"
[ -s "$WORK_DIR/tests.txt" ] || { echo "[gzoltar] no test methods collected" >&2; exit 1; }

# --- 2. Run each test with online instrumentation -> coverage spectrum --------
java -javaagent:"$AGENT_JAR"=destfile="$WORK_DIR/gzoltar.ser",buildlocation="$ABS_BIN_CLASSES",includes="$INCLUDES",excludes="",inclnolocationclasses=false,output="file" \
  -cp "$CP" com.gzoltar.cli.Main runTestMethods \
  --testMethods "$WORK_DIR/tests.txt" \
  --collectCoverage
[ -s "$WORK_DIR/gzoltar.ser" ] || { echo "[gzoltar] coverage collection failed" >&2; exit 1; }

# --- 3. Ochiai fault-localization report (line granularity) -------------------
java -cp "$CP" com.gzoltar.cli.Main faultLocalizationReport \
  --buildLocation "$ABS_BIN_CLASSES" \
  --granularity "line" \
  --inclPublicMethods --inclStaticConstructors --inclDeprecatedMethods \
  --dataFile "$WORK_DIR/gzoltar.ser" \
  --outputDirectory "$WORK_DIR" \
  --family "sfl" --formula "ochiai" --metric "entropy" --formatter "txt"

RANKING="$WORK_DIR/sfl/txt/ochiai.ranking.csv"
[ -s "$RANKING" ] || { echo "[gzoltar] ranking not produced" >&2; exit 1; }

mkdir -p "$OUTPUT"
cp "$RANKING" "$OUTPUT/ochiai.ranking.csv"
echo "[gzoltar] wrote $OUTPUT/ochiai.ranking.csv ($(wc -l < "$RANKING") lines)"
