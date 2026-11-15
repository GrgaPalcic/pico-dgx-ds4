#!/usr/bin/env bash
# An nsys capture of a --mode bench run and its per-step kernel breakdown: rank 0 under nsys (the fork's
# recipe, docs/engine.md "Launch and stop"), then each kernel class's exclusive and busy time per step
# (engine/server/tools/kprof.py). The capture range is bench mode's own (cudaProfilerApi): the first
# context's first k, from its 4th step. Runs on spark-0 from the switchless tree; run4.sh holds the ring
# for the run. No root.
#   engine/server/tools/prof.sh OUT ARGS...
#   engine/server/tools/prof.sh /tmp/b16 --golden G --max-ctx 266240 --slots 8 --ctx 32768 --steps 32 \
#       --bench-stride 4096 --ks 16 --bench-groups 2 --shared-kv
# The report and its export stay in OUT.nsys-rep / OUT.sqlite.
set -euo pipefail
[ $# -ge 1 ] || {
  sed -n '2,10p' "$0" >&2
  exit 2
}
NSYS=${NSYS:-/usr/local/cuda-13.0/bin/nsys}
out=$1
shift
WRAP="$NSYS profile -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -s none -f true -o $out" \
  engine/server/run4.sh --mode bench "$@"
"$NSYS" export -t sqlite -f true -o "$out.sqlite" "$out.nsys-rep"
python3 engine/server/tools/kprof.py "$out.sqlite"
