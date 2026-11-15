// Sessions' saved state (state.cu): what a session needs to continue from a position without
// recomputing what it already holds. The window rings and the compressor carries are per position mod
// 256 / 16, so they are snapshotted; the compressed caches are per position and append-only, so a
// session keeps a copy of them (by pages) and a continuation reuses entries below its divergence. The
// draft module's windows (DSpark, spec.cu) are per position mod 256 too and go with the snapshots.
//
// The store's device memory is one pool, reserved once at startup (Model::state_reserve) with the same
// size on every rank, and a best-fit allocator with coalescing hands out its blocks. Every rank runs
// the same operations in the same order, so every rank holds the same blocks at the same offsets. When
// the pool is full, the least recently used session that the current command does not use is evicted
// (the same one on every rank), and an allocation that still finds no room is skipped: a snapshot not
// taken, pages not saved, no end snapshot. Nothing in the store fails a request.
//
// With a disk tier (Model::state_disk, spill.h) an evicted session is spilled instead: its blocks go to
// a file of the same size on every rank, allocated by the same best fit, so every rank again holds the
// same blocks at the same offsets. When the file is full, its least recently used sessions are dropped.
// A spilled session comes back (state_unspill, or any operation that needs its blocks) as far as the
// store has room.
//
// Durable copies (TS2). Every finished turn's session is written through to a complete copy in the
// disk tier's file, named by one small index file per session per rank (<base>.d/<h>.<seq>.c) written
// after its blocks: a stop, a watchdog restart or a power cut loses at most the turn in flight. A
// crash leaves the index files and the copies they name; the next start (state_recover) reads them and
// its sessions come back as spilled ones. An index file is only ever removed after the blocks it names
// are no longer in the file (settle), and the spill file is never removed while a copy names it.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/impl.h"
#include "model/pages.h"
#include "model/prefill.h"

namespace model::detail {

constexpr size_t kRingBytes = (size_t)attn::kRing * (kHead + kHead / 32);
constexpr size_t kCarryBytes = (size_t)16 * kHead * 4 * 2;
constexpr size_t kEndBytes = kLayers * kRingBytes + kKvSlots * kCarryBytes;

class SpillFile;

struct Store {
  struct Sess {
    std::vector<uint8_t*> pages;    // compressed caches: page i holds positions [i, i + 1) * kPagePos
    int saved = 0;                  // positions [0, saved) are in the pages
    std::map<int, uint8_t*> snaps;  // window snapshots by position, the draft module's windows after them
    uint8_t* end = nullptr;         // every layer's ring and the carries (and the draft module's windows)
    int end_pos = -1;               // at this position
    uint64_t used = 0;              // the clock when an operation last used the session
    // Its last complete copy in the disk tier's file (seq 0: none): the file offsets of its blocks and
    // their checksums as they went out. With `disk` those blocks are the session; else the pool holds
    // its blocks too and the copy is what a recovery reads back.
    struct Copy {
      uint64_t seq = 0;
      int saved = 0, end_pos = -1;
      uint64_t used = 0;
      int64_t written = 0;
      std::vector<std::pair<size_t, uint64_t>> pages;            // offset, checksum
      std::map<int, std::pair<size_t, uint64_t>> snaps;
      std::pair<size_t, uint64_t> end{Pool::kNone, 0};
      uint64_t wseq = 0;        // its last queued write (reading it back waits for that)
      bool recovered = false;   // it came from an index file at this start
    } copy;
    int cvalid = 0;     // positions [0, cvalid) unchanged in the pool since the copy
    bool disk = false;  // the pool holds none of its blocks: the copy is the session
  };
  std::map<int, Sess> sess;
  SharedKv kv;               // the one pool: the sessions' pages, and (--shared-kv) the slot table and zero page
  size_t bytes = 0;          // the pool's bytes in sessions' blocks
  std::unique_ptr<SpillFile> disk;  // the disk tier (none: evicted sessions are freed)
  Pool dpool;                       // its file's offsets
  size_t dbytes = 0;                // the file's bytes in spilled sessions' blocks
  std::string base;                 // the file is base.spill, the copies' index files base.d/
  uint64_t run_id = 0;              // what the copies' state depends on (main.cpp run_id)
  uint64_t copy_seq = 0;            // the last copy issued
  // Blocks freed while an index file may still name them, and the index files to remove: settle() lets
  // every write land, unlinks the files, syncs the directory, and only then frees the blocks.
  std::vector<std::pair<size_t, size_t>> pending;
  std::vector<std::string> unlink_pending;
  std::string cdir;  // base + ".d", mode 0700, one file per session per rank
  int dirfd = -1;    // the directory, for fsync
  uint64_t clock = 0;
  StoreStats stats;          // what the disk tier and the shared pool did (--stats-file)
  std::vector<int> pinned;   // the sessions the current command uses: never evicted during it
  std::vector<int> evicted;  // sessions evicted to make room, since rank 0's server last asked
  Sess& use(int h) {         // session h (made if new), used now
    Sess& s = sess[h];
    s.used = ++clock;
    return s;
  }
  ~Store();  // state.cu: waits for the disk tier's writes and removes its file
};

}  // namespace model::detail
