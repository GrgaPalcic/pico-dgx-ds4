// The shared KV pool's host side (docs/shared-kv.md): the page geometry, the best-fit allocator the
// store's pool and its disk tier share, the zero page and the slots' table rows. No CUDA here: state.cu
// (the store) and model.cu (the model) hold the device memory and queue the copies, and the pool's
// blocks are the sessions' pages, the slot table and the zero page alike.
//
// A page holds kPagePos positions of all four kv sources at the store's page format, so a page a slot
// writes is already in the store's: saving a session is bookkeeping, not a copy. A session's pages are
// a list of pool blocks (Store::Sess::pages); a slot's row is the device's view of the session live in
// it, and every entry above the session's list holds the zero page, so an out-of-list read returns what
// a fresh contiguous cache held.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "model/config.h"

namespace model::detail {

constexpr int kPagePos = 4096;  // positions per page of compressed caches
constexpr int kPageShift = 12;  // log2 of it: a ratio-1 source's entries per page
constexpr size_t kArrBytes[4] = {256, 32, 64, 4};  // per entry: KV, KV scales, index key, key scales
constexpr size_t kEntryBytes = kArrBytes[0] + kArrBytes[1] + kArrBytes[2] + kArrBytes[3];
constexpr size_t kPageBytes = (size_t)(kPagePos / 2 * 3 + kPagePos) * kEntryBytes;  // sources 2, 8, 14 (ratio 2), 20
constexpr size_t kPoolGrain = 64 << 10;  // the pool's blocks: multiples of 64 KiB, at 64 KiB offsets

// Best fit over [base, base + size); free ranges are coalesced, ties go to the lowest offset. The
// offsets alone (take_off, give_off) serve the disk tier's file and its staging buffer.
struct Pool {
  static constexpr size_t kNone = SIZE_MAX;
  uint8_t* base = nullptr;
  size_t size = 0;
  std::map<size_t, size_t> free;  // offset -> bytes
  static constexpr size_t round(size_t bytes) { return (bytes + kPoolGrain - 1) / kPoolGrain * kPoolGrain; }
  uint8_t* take(size_t bytes);  // nullptr when no free range holds round(bytes)
  void give(uint8_t* p, size_t bytes);
  size_t take_off(size_t bytes);  // kNone when no free range holds round(bytes)
  bool take_at(size_t off, size_t bytes);  // [off, off + round(bytes)) when it is free (kept sessions)
  void give_off(size_t off, size_t bytes);
  size_t largest() const;
};

constexpr size_t kPageBlock = Pool::round(kPageBytes);  // a page's bytes as the pool hands them out

// A kv source's four arrays' offsets from a page's base: ckv, csc, ik, iksc. A ratio-2 source's page
// holds kPagePos / 2 entries, a ratio-1 source's kPagePos.
inline void layout_of(int ratio, uint32_t off[4]) {
  const size_t E = (size_t)kPagePos / ratio;
  size_t o = 0;
  for (int a = 0; a < 4; ++a) off[a] = (uint32_t)o, o += E * kArrBytes[a];
}
// Source k's array a (k: a kv source, 0-2 ratio 2 and 3 ratio 1) from a page's base.
size_t page_off(int k, int a);
// The pages positions [0, upto) need.
inline int pages_for(int upto) { return (upto + kPagePos - 1) / kPagePos; }
// A slot's table row: one entry per page the whole context could need.
inline int max_pages_of(int max_ctx) { return max_ctx / kPagePos + 1; }
// The bytes the flag's pool holds besides the sessions' pages: the zero page. (The slot table is the
// model's own allocation, made with it: serve captures its step graphs before the pool exists.)
inline size_t shared_kv_overhead(int /*nslots*/, int /*max_ctx*/) { return kPageBlock; }
// --shared-kv's sizing (docs/shared-kv.md §2.9). A flag-off run reserves a live slot's working state
// per unit (model.cu: cap = unit_ctx / ratio + 64 entries of 356 B a kv source, one window ring a layer
// and the four sources' carries) and --session-gib (12 GiB) for the store; the flag keeps the same total
// KV memory, so its pool's cap is the store's default plus what --slots would have reserved, and an
// explicit --session-gib caps the sessions' pages as it is given. Both are bounded by the smallest
// rank's offer (the free memory after the fixed state, measured with the slots' caches unallocated).
constexpr double kSessionGib = 12;  // --session-gib's default: the store's size, and the flag's cap by default
constexpr int kRingSlots = 256;     // a layer's window ring (attn::kRing: the window + verify rows)
constexpr int kCarrySlots = 16;     // a compressor carry's ring (position & 15)

// What a flag-off run reserves for --slots slots' live state at this --max-ctx: `big_slots` slots of
// kUnitMul units each, the rest one unit of --max-ctx / kUnitMul (model.cu's nunits and unit_ctx). The
// flag refuses --big-slots, so a flag-on command line's every slot covers --max-ctx.
uint64_t slot_caches_bytes(int slots, int big_slots, int max_ctx);
// The flag's cap: `session_gib` (--session-gib; the store's default is passed when it was not given)
// plus, when it was not given, the slots' reservation, and the flag's own blocks on top (the pool holds
// the slot table and the zero page besides the sessions' pages).
uint64_t shared_kv_cap(double session_gib, bool given, int slots, int big_slots, int max_ctx);
// The pool every rank reserves: the cap, bounded by the smallest offer.
inline uint64_t shared_kv_pool(uint64_t cap, uint64_t offer) { return cap < offer ? cap : offer; }
// The sessions' pages a pool of `bytes` holds (its own blocks taken out) and the positions they cover.
inline int pool_pages(uint64_t bytes, int nslots, int max_ctx) {
  const size_t over = shared_kv_overhead(nslots, max_ctx);
  return bytes > over ? (int)((bytes - over) / kPageBlock) : 0;
}

// Which session is live in which slot. 0 means "no session" (a free slot); a session's handle is
// anything else, and the server's are >= 1 (serve.cpp's next_h_). state_slot_row sets an entry, the
// flag's room calls read it, and state_free lets a slot go.
struct SlotSessions {
  int h[kMaxSlots] = {};
  int of(int slot) const { return slot >= 0 && slot < kMaxSlots ? h[slot] : 0; }
  int slot_of(int session) const {
    if (!session) return -1;
    for (int s = 0; s < kMaxSlots; ++s)
      if (h[s] == session) return s;
    return -1;
  }
  void set(int slot, int session) { h[slot] = session; }
};

// The shared KV pool: the blocks, the zero page in them and the slots' table rows. `zero` is one block
// taken out of the pool's free map for good, so it is never handed out; the rows' images are what the
// device rows are written from (model.cu, state.cu).
struct SharedKv {
  using Evict = std::function<bool()>;  // frees room: the store's least recently used session, false when none
  Pool pool;
  uint8_t* zero = nullptr;    // the zero page: zeroed with the pool, never handed out
  // One page held back for a copy-on-write (a fork's or a restart's tail page): those are allocated inside a
  // state operation, which has no error path. A copy that finds the pool full takes it; a session's growth
  // (Model::Impl::session_room) takes one first whenever it is gone, or the growth fails. Not taken at the
  // pool's reserve, so the kept sessions' blocks get their offsets back first (state_disk).
  uint8_t* spare = nullptr;
  std::vector<uint64_t> rows;  // every slot's row image, max_pages entries each
  std::vector<int> held;       // the entries each row holds (the rest are the zero page)
  int nslots = 0, max_pages = 0;
  // --shared-kv (§2.7): the pages two or more sessions hold, and how many hold each. A page nobody
  // shares has no entry: its one session's hold is not counted. A page goes back to the pool when the
  // last session holding it gives it up (release), in whatever order the sessions are freed.
  std::map<const uint8_t*, int> refs;

  // The rows' images: `nslots` rows of `max_pages` entries, all the zero page.
  void reset(int nslots_, int max_pages_, uint8_t* zero_page);
  // Row `slot` from a session's pages: the `n` pages, then the zero page.
  void list(int slot, int n, const uint8_t* const* pages);
  // One page at row `slot`'s end; returns the entry it went to.
  int add(int slot, uint8_t* page);
  // Everything above position `from` leaves the session: the row's entries from `from`'s page on go back
  // to the zero page and the pages above it back to the pool (a page another session still holds stays).
  // `slot` < 0: no row to fix.
  void drop(int slot, std::vector<uint8_t*>& pages, size_t& bytes, int from);
  // The pages a session's list needs for `upto` positions, from the pool or from `evict`. False: no room
  // and nothing to evict, and the list, `bytes` and the pool are as they were.
  bool room(std::vector<uint8_t*>& pages, size_t& bytes, int upto, const Evict& evict);
  // --shared-kv: `n` of another session's pages enter this session's list, shared read-only.
  void hold(const uint8_t* const* pages, int n);
  // --shared-kv: one session gives `page` up (its list drops it, or the session goes). True: no session
  // holds it any more, so the caller returns it to the pool. False: another session's list still has it.
  bool release(const uint8_t* page);
  int holders(const uint8_t* page) const;  // the sessions holding it (1: nobody shares it)

  const uint64_t* row(int slot) const { return rows.data() + (size_t)slot * max_pages; }
  size_t row_bytes() const { return (size_t)max_pages * sizeof(uint64_t); }
};

}  // namespace model::detail
