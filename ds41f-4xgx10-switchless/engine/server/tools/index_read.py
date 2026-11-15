#!/usr/bin/env python3
"""Read an index dump: rank 0's per-step index-layer selections (engine/model/model.cu, DSGB10_INDEX_DUMP=FILE).

  python3 index_read.py FILE            one JSON line per record
  python3 index_read.py FILE --summary  the record count and, per layer, the mean overlap of consecutive
                                        steps' selections of the same slot

A record is little-endian and variable-length (R rows, at most 64; n_layers index layers):

    char magic[4] = "IDX1"    u32 step    i32 R    u64 t_ns   (CLOCK_REALTIME at the append, ns)
    i32 n_layers              i32 layer[n_layers]              the index layers, in layer order
    i32 slot[R]               i32 pos[R]                       the step's session slots and positions
    i32 sel[n_layers][R][512]  the positions the indexer selected, as it wrote them (-1: unused)

read(path) yields one dict per record (step, R, t_ns, layer, slot, pos, sel as nested lists); a '.gz' path
opens transparently. A truncated last record (a dump cut short) is reported on stderr and ends the
iteration. --summary prints the record count and, per layer, the mean overlap |A n B| / |A| of consecutive
steps' selections of the same slot, valid entries only (A the earlier selection of that slot, B the later).
"""
import gzip
import json
import struct
import sys

LAYERS = 40  # model::kLayers
TOPK = 512  # model::kIdxTopk: positions a layer selects a row
MAX_ROWS = 64  # model::kMaxRows
MAGIC = b"IDX1"
HDR = struct.Struct("<4sIiQ")


def record_bytes(n_layers, R):
    return HDR.size + 4 + 4 * n_layers + 8 * R + 4 * n_layers * R * TOPK


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
                raise ValueError("%s: record magic %r, not %r (not an index dump)" % (path, magic, MAGIC))
            if not 1 <= R <= MAX_ROWS:
                raise ValueError("%s: step %d: R = %d, not 1..%d" % (path, step, R, MAX_ROWS))
            nl = f.read(4)
            if len(nl) < 4:
                print("%s: truncated record %d (R = %d)" % (path, step, R), file=sys.stderr)
                return
            n_layers = struct.unpack("<i", nl)[0]
            if not 0 <= n_layers <= LAYERS:
                raise ValueError("%s: step %d: n_layers = %d, not 0..%d" % (path, step, n_layers, LAYERS))
            body = f.read(record_bytes(n_layers, R) - HDR.size - 4)
            if len(body) < record_bytes(n_layers, R) - HDR.size - 4:
                print("%s: truncated record %d (R = %d, n_layers = %d)" % (path, step, R, n_layers), file=sys.stderr)
                return
            off = 0
            layer = list(struct.unpack_from("<%di" % n_layers, body, off))
            off += 4 * n_layers
            slot = list(struct.unpack_from("<%di" % R, body, off))
            off += 4 * R
            pos = list(struct.unpack_from("<%di" % R, body, off))
            off += 4 * R
            flat = struct.unpack_from("<%di" % (n_layers * R * TOPK), body, off)
            sel = [[list(flat[(l * R + r) * TOPK:(l * R + r + 1) * TOPK]) for r in range(R)] for l in range(n_layers)]
            yield dict(step=step, R=R, t_ns=t_ns, layer=layer, slot=slot, pos=pos, sel=sel)


def summarize(path):
    n = 0
    per_layer = {}
    prev = {}  # (layer, slot) -> the last selection of that slot (a set of valid positions)
    for r in read(path):
        n += 1
        for i, L in enumerate(r["layer"]):
            for row in range(r["R"]):
                key = (L, r["slot"][row])
                a = {p for p in r["sel"][i][row] if p >= 0}
                b = prev.get(key)
                if b and a:  # |A n B| / |A|, A the earlier selection, valid entries only
                    per_layer.setdefault(L, []).append(len(a & b) / len(a))
                prev[key] = a
    print("%s: %d records" % (path, n))
    if not n:
        return
    print("overlap of consecutive steps of the same slot, per layer: " +
          ", ".join("L%d %.3f (%d)" % (L, sum(v) / len(v), len(v)) if v else "L%d n/a" % L
                    for L, v in sorted(per_layer.items())))


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
