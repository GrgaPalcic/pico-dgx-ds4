"""Tests for tools/vision/rows_cache.py.

Run from ``ds41f-4xgx10-switchless``: ``python3 -m unittest tools/vision/test_rows_cache.py`` (or
``python3 -m pytest tools/vision/test_rows_cache.py -q``).

The cache holds the encoder's span rows by image hash: a get returns them, a miss returns None, and a put
evicts the least recently used entries until the stored rows fit the cap.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rows_cache import RowsCache  # noqa: E402  (the module under test)


def rows(n, fill=b"\x01"):
    """n bytes of rows, as the encoder's payload: bf16 [rows][5120] little-endian."""
    return fill * n


class RowsCacheTest(unittest.TestCase):
    def test_miss_on_an_empty_cache(self):
        c = RowsCache(1024)
        self.assertIsNone(c.get(b"a" * 32))
        self.assertIsNone(c.get(b"another key"))

    def test_hit_returns_the_stored_entry(self):
        c = RowsCache(1024)
        c.put(b"k", rows(10), 7, (4, 5, 6, 7))
        self.assertEqual(c.get(b"k"), (rows(10), 7, (4, 5, 6, 7)))
        self.assertEqual(c.get(b"k")[1:], (7, (4, 5, 6, 7)))  # the rows bytes survive byte for byte
        self.assertIsNone(c.get(b"other"))

    def test_eviction_at_the_cap(self):
        c = RowsCache(20)
        c.put(b"a", rows(10, b"a"), 1, (1, 1, 1, 1))
        c.put(b"b", rows(10, b"b"), 2, (2, 2, 2, 2))
        self.assertIsNotNone(c.get(b"a"))  # 20 bytes: exactly at the cap, nothing evicted
        c.put(b"c", rows(10, b"c"), 3, (3, 3, 3, 3))  # 30 > 20: the least recently used goes
        self.assertIsNone(c.get(b"b"))
        self.assertIsNotNone(c.get(b"a"))
        self.assertIsNotNone(c.get(b"c"))

    def test_a_get_moves_the_entry_to_the_most_recently_used_end(self):
        c = RowsCache(20)
        c.put(b"a", rows(10, b"a"), 1, (1, 1, 1, 1))
        c.put(b"b", rows(10, b"b"), 2, (2, 2, 2, 2))
        self.assertIsNotNone(c.get(b"a"))  # a is now the most recently used, b the least
        c.put(b"c", rows(10, b"c"), 3, (3, 3, 3, 3))
        self.assertIsNone(c.get(b"b"))
        self.assertIsNotNone(c.get(b"a"))
        self.assertIsNotNone(c.get(b"c"))

    def test_an_entry_larger_than_the_cap_is_not_stored(self):
        c = RowsCache(10)
        c.put(b"small", rows(4), 1, (1, 1, 1, 1))
        c.put(b"big", rows(11), 2, (2, 2, 2, 2))
        self.assertIsNone(c.get(b"big"))
        self.assertIsNotNone(c.get(b"small"))  # and it did not evict what fits

    def test_eviction_frees_as_much_as_the_put_needs(self):
        c = RowsCache(20)
        c.put(b"a", rows(8), 1, (1, 1, 1, 1))
        c.put(b"b", rows(8), 2, (2, 2, 2, 2))
        c.put(b"c", rows(8), 3, (3, 3, 3, 3))  # 24 > 20: a and b fit together, so only a goes
        self.assertIsNone(c.get(b"a"))
        self.assertIsNotNone(c.get(b"b"))
        self.assertIsNotNone(c.get(b"c"))

    def test_putting_the_same_key_again_replaces_it(self):
        c = RowsCache(20)
        c.put(b"a", rows(8), 1, (1, 1, 1, 1))
        c.put(b"a", rows(16), 9, (9, 9, 9, 9))  # replaces: the 8 old bytes are not counted twice
        self.assertEqual(c.get(b"a"), (rows(16), 9, (9, 9, 9, 9)))
        c.put(b"b", rows(4), 2, (2, 2, 2, 2))  # 20 bytes total: still fits, so a survives
        self.assertIsNotNone(c.get(b"a"))
        self.assertIsNotNone(c.get(b"b"))

    def test_a_zero_cap_stores_nothing(self):
        c = RowsCache(0)
        c.put(b"a", rows(1), 1, (1, 1, 1, 1))
        self.assertIsNone(c.get(b"a"))


if __name__ == "__main__":
    unittest.main()
