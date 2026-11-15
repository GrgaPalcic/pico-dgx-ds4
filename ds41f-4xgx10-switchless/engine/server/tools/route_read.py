#!/usr/bin/env python3
"""Read a route dump: rank 0's per-step route records (engine/model/model.cu, DSGB10_ROUTE_DUMP=FILE).

  python3 route_read.py FILE            one JSON line per record
  python3 route_read.py FILE --summary  the records' shape, timing and per-layer expert counts

A record is little-endian and variable-length (R rows, at most 64):

    char magic[4] = "RTD2"    u32 step    i32 R    u64 t_ns   (CLOCK_REALTIME at the append, ns)
    i32 slot[R]               i32 pos[R]                       the step's session slots and positions
    i16 e[40][R][6]           the step's routed experts
    i16 top_id[40][R][12]     the router's selection keys' experts, descending (ties: the lower id)
    f32 top_sc[40][R][12]     those keys: sqrtf(softplus(logit)) + bias over all 384 experts

read(path) yields one dict per record (step, R, t_ns, slot, pos, e, top_id, top_sc as nested lists); a
'.gz' path opens transparently. A truncated last record (a dump cut short) is reported on stderr and
ends the iteration. --summary prints the record count, the R histogram, the t_ns range, whether t_ns
never decreases, and the median over steps of the distinct experts a layer's rows touched (layers 0,
10, 20, 30, 39).
"""
import gzip
import json
import statistics
import struct
import sys

LAYERS = 40  # model::kLayers
TOPK = 6     # model::kTopK: routed experts a row
TOP = 12     # model::kRouteTop: selection keys a row
MAX_ROWS = 64  # model::kMaxRows
MAGIC = b"RTD2"
HDR = struct.Struct("<4sIiQ")
SUMMARY_LAYERS = (0, 10, 20, 30, 39)


def body_bytes(R):
    return 8 * R + LAYERS * R * (TOPK * 2 + TOP * 2 + TOP * 4)


def _grid(flat, R, w):
    """[LAYERS][R][w] nested lists from the flat field of a record of R rows."""
    return [[list(flat[(l * R + r) * w:(l * R + r + 1) * w]) for r in range(R)] for l in range(LAYERS)]


def read(path):
    """Yield one record dict per record of the dump at path ('.gz' opens transparently)."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as f:
        while True:
            hdr = f.read(HDR.size)
            if not hdr:
                return
            if len(hdr) < HDR.size:
                print("%s: truncated header at the end of the file" % path, file=sys.stderr)
                return
            magic, step, R, t_ns = HDR.unpack(hdr)
            if magic != MAGIC:
                raise ValueError("%s: record magic %r, not %r (not a route dump)" % (path, magic, MAGIC))
            if not 1 <= R <= MAX_ROWS:
                raise ValueError("%s: step %d: R = %d, not 1..%d" % (path, step, R, MAX_ROWS))
            body = f.read(body_bytes(R))
            if len(body) < body_bytes(R):
                print("%s: truncated record %d (R = %d)" % (path, step, R), file=sys.stderr)
                return
            off = 0
            slot = list(struct.unpack_from("<%di" % R, body, off))
            pos = list(struct.unpack_from("<%di" % R, body, off + 4 * R))
            off += 8 * R
            e = _grid(struct.unpack_from("<%dh" % (LAYERS * R * TOPK), body, off), R, TOPK)
            off += LAYERS * R * TOPK * 2
            top_id = _grid(struct.unpack_from("<%dh" % (LAYERS * R * TOP), body, off), R, TOP)
            off += LAYERS * R * TOP * 2
            top_sc = _grid(struct.unpack_from("<%df" % (LAYERS * R * TOP), body, off), R, TOP)
            yield dict(step=step, R=R, t_ns=t_ns, slot=slot, pos=pos, e=e, top_id=top_id, top_sc=top_sc)


def summarize(path):
    n = 0
    hist = {}
    t0 = t1 = None
    mono = True
    per_layer = {L: [] for L in SUMMARY_LAYERS}
    for r in read(path):
        n += 1
        hist[r["R"]] = hist.get(r["R"], 0) + 1
        t0 = r["t_ns"] if t0 is None else min(t0, r["t_ns"])
        if t1 is not None and r["t_ns"] < t1:
            mono = False
        t1 = r["t_ns"] if t1 is None else max(t1, r["t_ns"])
        for L in per_layer:
            per_layer[L].append(len({x for row in r["e"][L] for x in row if x >= 0}))
    print("%s: %d records" % (path, n))
    if not n:
        return
    print("R: " + ", ".join("%d x%d" % (R, c) for R, c in sorted(hist.items())))
    print("t_ns: %d .. %d (%.3f s), never decreasing: %s" % (t0, t1, (t1 - t0) / 1e9, "yes" if mono else "no"))
    print("distinct experts per layer, median over steps: " +
          ", ".join("L%d %.1f" % (L, statistics.median(per_layer[L])) for L in SUMMARY_LAYERS))


def main(argv):
    a = argv[1:]
    if not a or a[0].startswith("-"):
        sys.exit(__doc__)
    if "--summary" in a[1:]:
        summarize(a[0])
    else:
        for r in read(a[0]):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
