#!/usr/bin/env python3
"""route_read.py against synthetic dumps: two records (R = 3 and R = 64) written by hand and read back
field for field, plain and gzip, and a dump cut short (a record's body, a header).

  cd ds41f-4xgx10-switchless && python3 -m unittest engine/server/tools/test_route_read.py
"""
import contextlib
import gzip
import io
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_read as rr  # noqa: E402


def make_record(step, R, t_ns, seed):
    """A record's fields, the way k_route_copy's kernel and wait() fill them."""
    slot = [seed + r for r in range(R)]
    pos = [1000 + seed + 2 * r for r in range(R)]
    e = [[[(seed + l * 7 + r * 6 + k) % 384 for k in range(rr.TOPK)] for r in range(R)] for l in range(rr.LAYERS)]
    top_id = [[[(seed + l + r + k) % 384 for k in range(rr.TOP)] for r in range(R)] for l in range(rr.LAYERS)]
    # dyadic, so the f32 the record holds is exactly these values
    top_sc = [[[float(l) + 0.5 * r + 0.125 * k for k in range(rr.TOP)] for r in range(R)] for l in range(rr.LAYERS)]
    return dict(step=step, R=R, t_ns=t_ns, slot=slot, pos=pos, e=e, top_id=top_id, top_sc=top_sc)


def pack_record(r):
    R = r["R"]
    b = rr.HDR.pack(rr.MAGIC, r["step"], R, r["t_ns"])
    b += struct.pack("<%di" % R, *r["slot"]) + struct.pack("<%di" % R, *r["pos"])
    for field, w, fmt in (("e", rr.TOPK, "h"), ("top_id", rr.TOP, "h"), ("top_sc", rr.TOP, "f")):
        flat = [r[field][l][row][k] for l in range(rr.LAYERS) for row in range(R) for k in range(w)]
        b += struct.pack("<%d%s" % (len(flat), fmt), *flat)
    return b


class RouteReadTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.recs = [make_record(7, 3, 1763000000123456789, 1), make_record(8, rr.MAX_ROWS, 1763000000987654321, 5)]
        self.plain = os.path.join(self.dir.name, "routes.rtd")
        with open(self.plain, "wb") as f:
            for r in self.recs:
                f.write(pack_record(r))

    def check(self, path):
        got = list(rr.read(path))
        self.assertEqual([g["step"] for g in got], [r["step"] for r in self.recs])
        self.assertEqual([g["R"] for g in got], [r["R"] for r in self.recs])
        for want, g in zip(self.recs, got):
            for field in ("t_ns", "slot", "pos", "e", "top_id", "top_sc"):
                self.assertEqual(g[field], want[field], "step %d: %s" % (want["step"], field))
            self.assertEqual(len(g["e"]), rr.LAYERS)
            self.assertEqual(len(g["e"][39]), want["R"])
            self.assertEqual(len(g["e"][0][0]), rr.TOPK)
            self.assertEqual(len(g["top_sc"][0][0]), rr.TOP)

    def test_plain(self):
        self.check(self.plain)

    def test_gzip(self):
        path = os.path.join(self.dir.name, "routes.rtd.gz")
        with open(self.plain, "rb") as src, gzip.open(path, "wb") as dst:
            dst.write(src.read())
        self.check(path)

    def test_truncated_record(self):
        path = os.path.join(self.dir.name, "cut.rtd")
        with open(path, "wb") as f:  # the first record, then half of the second's body
            f.write(pack_record(self.recs[0]))
            f.write(pack_record(self.recs[1])[: rr.HDR.size + rr.body_bytes(rr.MAX_ROWS) // 2])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = list(rr.read(path))
        self.assertEqual([g["step"] for g in got], [7])
        self.assertIn("truncated record 8", err.getvalue())

    def test_truncated_header(self):
        path = os.path.join(self.dir.name, "cut2.rtd")
        with open(path, "wb") as f:
            f.write(pack_record(self.recs[0]) + b"RTD2\x01\x02")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = list(rr.read(path))
        self.assertEqual(len(got), 1)
        self.assertIn("truncated header", err.getvalue())

    def test_not_a_dump(self):
        path = os.path.join(self.dir.name, "other.bin")
        with open(path, "wb") as f:
            f.write(b"XXXX" + pack_record(self.recs[0])[4:])
        with self.assertRaises(ValueError):
            list(rr.read(path))

    def test_summary(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rr.summarize(self.plain)
        text = out.getvalue()
        self.assertIn("2 records", text)
        self.assertIn("3 x1", text)
        self.assertIn("64 x1", text)
        self.assertIn("never decreasing: yes", text)


if __name__ == "__main__":
    unittest.main()
