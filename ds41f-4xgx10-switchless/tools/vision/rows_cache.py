"""rows_cache.py: the vision encoder's LRU cache of image span rows.

Every turn of a conversation with images resends all of them, so the encoder would run the ViT and the aligner
again for an image it has already seen. Entries are keyed by the sha256 of the image's raw bytes and hold the
reply's payload: the span rows bf16 (the bytes dsfront gets), the row count and the grid. A hit skips the encode
and nothing else: the image bytes are still fetched to compute the key.
"""

import threading
from collections import OrderedDict


class RowsCache:
    """An LRU of (rows_bytes, rows, grid) keyed by the image's sha256, holding at most cap_bytes of rows.

    The least recently used entry is evicted first; a get counts as a use. An entry whose rows exceed the cap
    is not stored at all (it would evict everything and still not fit). The lock guards the dict and the byte
    total, so the encoder's worker threads can share one cache.
    """

    def __init__(self, cap_bytes):
        self.cap_bytes = cap_bytes
        self._lock = threading.Lock()
        self._entries = OrderedDict()  # key -> (rows_bytes, rows, grid), least recently used first
        self._bytes = 0

    def get(self, key):
        """The entry stored for key, or None; a hit moves the entry to the most recently used end."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key, rows_bytes, rows, grid):
        """Store one image's rows, evicting the least recently used entries until the rows fit the cap."""
        n = len(rows_bytes)
        if n > self.cap_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= len(old[0])
            self._entries[key] = (rows_bytes, rows, grid)
            self._bytes += n
            while self._bytes > self.cap_bytes:
                _, (b, _, _) = self._entries.popitem(last=False)
                self._bytes -= len(b)
