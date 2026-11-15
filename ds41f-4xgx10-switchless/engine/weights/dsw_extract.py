#!/usr/bin/env python3
"""Extract the stored bytes of a few layers from a per-rank .dsw weight file, for offline study.

  python3 engine/weights/dsw_extract.py FILE OUTDIR --layers 0,20,39 [--mtp]

Standard library only, no GPU: it runs on a Spark as a normal user (the file is that box's own rank
slice). It reads with os.pread, so nothing is mapped and the page cache is dropped again after each
tensor (POSIX_FADV_DONTNEED); a layer is ~1.7 GiB of the 72 GiB file, so three layers read ~5 GiB.

Out, in OUTDIR:
- `<name>.<dtype>.bin` per tensor of the listed layers (--mtp: also the drafter's mtp.0..2), the bytes
  exactly as stored, tiled layouts included: OUTDIR/README.md says where the un-tiling is documented;
- SHA256SUMS over those files;
- toc.json: the file's header and every TOC entry, extracted or not.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toc  # the .dsw formats (HDR, ENT, the dtype/kind/layout names)

CHUNK = 256 << 20  # the most read at once (a tensor's bytes)
# TocEntry fields (dsw.h, in toc.ENT's order); SHAPE and SRC_SHAPE are 4 values each
NAME, SOURCE, SLICE, DTYPE, NDIM, SHAPE, OFFSET, BYTES, KIND, NSTACK, NSEG, CRC, SRC_NDIM, LAYOUT, SRC_SHAPE = \
    0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17

# Where a stored layout's un-tiling is documented (docs/layout.md, engine/weights/layout.h)
UNTILE = {
    "L1": "docs/layout.md, \"Tile layouts (version 2)\", L1 (FP8 dense); engine/weights/layout.h l1_offset / "
          "l1_untile (element (n, k) at ((n/8)*(K/64) + k/64)*512 + (n%8)*64 + k%64)",
    "L2": "docs/layout.md, \"Tile layouts (version 2)\", L2 (bf16 [N][K] as bytes [N][2K], then L1 with K' = 2K); "
          "engine/weights/layout.h l1_offset / l1_untile with K' = 2K",
    "L3": "docs/layout.md, \"Tile layouts (version 2)\", L3 (MXFP4: each expert's matrix tiled on its own, values "
          "and scales alike); engine/weights/layout.h l3_matrix / l3_value_offset / l3_scale_offset",
    "natural": "docs/layout.md, \"Partition\" and \"File format (version 2)\": the checkpoint's own row-major "
               "layout, nothing to un-tile",
}


def header_json(path, h):
    return dict(file=os.path.abspath(path), version=h[1], header_bytes=h[2], rank=h[3], world=h[4],
                n_tensors=h[5], toc_entry_bytes=h[6], toc_offset=h[7], data_offset=h[8], data_bytes=h[9],
                file_bytes=h[10], alignment=h[11], toc_crc32c="%08x" % h[12], model=toc.cstr(h[13]),
                source=toc.cstr(h[14]), created=toc.cstr(h[15]))


def entry_json(e):
    return dict(name=toc.cstr(e[NAME]), source_tensor=toc.cstr(e[SOURCE]), slice=toc.cstr(e[SLICE]),
                dtype=toc.DTYPE[e[DTYPE]], dtype_id=e[DTYPE], shape=list(e[SHAPE:SHAPE + e[NDIM]]),
                offset=e[OFFSET], bytes=e[BYTES], layout=toc.LAYOUT[e[LAYOUT]], layout_id=e[LAYOUT],
                kind=toc.KIND[e[KIND]], kind_id=e[KIND], nstack=e[NSTACK], nseg=e[NSEG],
                crc32c="%08x" % e[CRC], src_shape=list(e[SRC_SHAPE:SRC_SHAPE + e[SRC_NDIM]]))


def layer_of(name):
    """The `layers.N.` or `mtp.N.` prefix of a tensor name, else None."""
    p = name.split(".")
    return (p[0], p[1]) if len(p) > 2 and p[1].isdigit() else None


def extract(fd, e, outdir, sums):
    """One tensor's stored bytes to <name>.<dtype>.bin; returns its file name and byte count."""
    name, dt, off, n = toc.cstr(e[NAME]), toc.DTYPE[e[DTYPE]], e[OFFSET], e[BYTES]
    fn = "%s.%s.bin" % (name, dt)
    h = hashlib.sha256()
    got = 0
    with open(os.path.join(outdir, fn), "wb") as out:
        while got < n:
            b = os.pread(fd, min(CHUNK, n - got), off + got)
            if not b:
                sys.exit("dsw_extract: %s: read %d of %d bytes at %d" % (name, got, n, off))
            out.write(b)
            h.update(b)
            got += len(b)
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    sums.append((fn, h.hexdigest()))
    return fn, n


def readme(outdir, hj, rows):
    """OUTDIR/README.md: the rank slice, the files, and where each stored layout's un-tiling is documented."""
    layouts = []
    for r in rows:
        if r["layout"] not in layouts:
            layouts.append(r["layout"])
    gib = sum(r["bytes"] for r in rows) / 2**30
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("# %s: layers %s of rank %d\n\n" % (os.path.basename(hj["file"]),
                                                    ", ".join(str(l) for l in hj["layers"]), hj["rank"]))
        f.write("Rank %d of %d (docs/layout.md's table: each box holds its own rank's slice of the "
                "checkpoint), cut from %s, created %s. The %d tensors below are %.2f GiB of the file's "
                "%.2f GiB; their bytes are exactly as stored (the TOC's layout), not un-tiled.\n\n"
                % (hj["rank"], hj["world"], hj["source"], hj["created"], len(rows), gib,
                   hj["data_bytes"] / 2**30))
        f.write("| tensor | dtype | layout | kind | shape | bytes | file |\n|---|---|---|---|---|---:|---|\n")
        for r in rows:
            f.write("| `%s` | %s | %s | %s | %s | %d | `%s` |\n"
                    % (r["name"], r["dtype"], r["layout"], r["kind"], "x".join(str(d) for d in r["shape"]),
                       r["bytes"], r["file"]))
        f.write("\n## Un-tiling\n\n")
        for l in layouts:
            f.write("- **%s**: %s\n" % (l, UNTILE[l]))
        f.write("\n`toc.json` holds the file's header and every TOC entry (name, dtype, shape, offset, bytes,\n"
                "layout, kind, nstack, nseg, crc32c, slice); `SHA256SUMS` the hashes of the files above.\n")


def main(argv):
    a = argv[1:]
    if len(a) < 2 or a[0].startswith("-"):
        sys.exit(__doc__)
    path, outdir, layers, mtp = a[0], a[1], set(), False
    rest = a[2:]
    while rest:
        k = rest.pop(0)
        if k == "--layers" and rest:
            layers = {int(x) for x in rest.pop(0).split(",")}
        elif k == "--mtp":
            mtp = True
        else:
            sys.exit("dsw_extract: bad argument %s\n%s" % (k, __doc__))
    if not layers and not mtp:
        sys.exit("dsw_extract: --layers N,N,.. (or --mtp) is required\n%s" % __doc__)
    os.makedirs(outdir, exist_ok=True)
    with open(path, "rb") as f:
        h = toc.HDR.unpack(f.read(toc.HDR.size))
        if toc.cstr(h[0]) != "DSGB10W1":
            sys.exit("dsw_extract: %s: magic %r, not a .dsw file" % (path, h[0]))
        f.seek(h[7])
        ents = [toc.ENT.unpack(f.read(512)[:toc.ENT.size]) for _ in range(h[5])]
        rows, sums, seen = [], [], set()
        for e in ents:
            name, lay = toc.cstr(e[NAME]), layer_of(toc.cstr(e[NAME]))
            if not lay:
                continue
            if not ((lay[0] == "layers" and int(lay[1]) in layers) or (mtp and lay[0] == "mtp")):
                continue
            if lay[0] == "layers":
                seen.add(int(lay[1]))
            fn, n = extract(f.fileno(), e, outdir, sums)
            r = entry_json(e)
            r["file"] = fn
            rows.append(r)
            print("%s -> %s (%.1f MiB)" % (name, fn, n / 2**20))
    with open(os.path.join(outdir, "SHA256SUMS"), "w") as f:
        for fn, dig in sorted(sums):
            f.write("%s  %s\n" % (dig, fn))
    hj = header_json(path, h)
    hj["layers"] = sorted(layers)
    hj["mtp"] = mtp
    with open(os.path.join(outdir, "toc.json"), "w") as f:
        json.dump(dict(header=hj, tensors=[entry_json(e) for e in ents]), f, indent=1, sort_keys=True)
        f.write("\n")
    readme(outdir, hj, rows)
    for l in sorted(layers - seen):
        print("dsw_extract: no tensor of layer %d in %s" % (l, path), file=sys.stderr)
    print("%d tensors, %.2f GiB -> %s" % (len(rows), sum(r["bytes"] for r in rows) / 2**30, outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
