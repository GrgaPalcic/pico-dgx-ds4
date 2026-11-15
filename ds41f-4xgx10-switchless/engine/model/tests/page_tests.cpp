// The shared KV pool's host side (pages.h), pure C++: the best-fit allocator's behaviour as state.cu's
// pool has it today, the page geometry against state.cu's page_off, the slots' table rows, the zero page
// and the full pool's outcome. Built and run without CUDA:
//   g++ -std=c++20 -O1 -Wall -I engine engine/model/pages.cpp engine/model/tests/page_tests.cpp -o /tmp/page_tests
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "model/pages.h"

using namespace model::detail;

namespace {

int checks = 0, failed = 0;

void check(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failed;
    std::printf("page_tests: FAIL %s\n", what);
  }
}
void check_eq(long long got, long long want, const char* what) {
  ++checks;
  if (got != want) {
    ++failed;
    std::printf("page_tests: FAIL %s: %lld, expected %lld\n", what, got, want);
  }
}

// The pool's free bytes: what a page released comes back to.
size_t free_bytes(const Pool& p) {
  size_t n = 0;
  for (const auto& [off, bytes] : p.free) n += bytes;
  return n;
}

// A pool over a host buffer: the tests' pool, the same layout a rank's device pool has.
struct TestPool {
  std::vector<uint8_t> buf;
  Pool pool;
  explicit TestPool(int grains) : buf((size_t)grains * kPoolGrain) {
    pool.base = buf.data();
    pool.size = buf.size();
    pool.free.emplace(0, buf.size());
  }
};

void test_pool() {
  {  // best fit: the smallest free range that holds the request, ties to the lowest offset
    TestPool t(16);
    check_eq(t.pool.take_off(4 * kPoolGrain) / kPoolGrain, 0, "best fit: the first take");
    check_eq(t.pool.take_off(6 * kPoolGrain) / kPoolGrain, 4, "best fit: the second take");
    t.pool.give_off(0, 4 * kPoolGrain);  // free: 4 grains at 0, 6 at 10
    check_eq(t.pool.take_off(3 * kPoolGrain) / kPoolGrain, 0, "best fit: the 4-grain range over the 6");
    check_eq(t.pool.take_off(1 * kPoolGrain) / kPoolGrain, 3, "best fit: the 1-grain range left");
    check_eq(t.pool.take_off(6 * kPoolGrain) / kPoolGrain, 10, "best fit: the 6-grain range");
    check(t.pool.take(kPoolGrain) == nullptr, "best fit: a full pool hands out nothing");
  }
  {  // coalescing: a freed range joins its neighbours, whole
    TestPool t(16);
    t.pool.take_off(2 * kPoolGrain);
    t.pool.take_off(3 * kPoolGrain);
    t.pool.take_off(4 * kPoolGrain);  // free: 7 grains at 9
    check_eq((long long)t.pool.free.size(), 1, "coalescing: one range to start");
    t.pool.give_off(2 * kPoolGrain, 3 * kPoolGrain);  // 5 grains at 2, a gap to 9
    check_eq((long long)t.pool.free.size(), 2, "coalescing: two ranges, the gap between");
    check_eq((long long)t.pool.free.begin()->first, 2 * (long long)kPoolGrain, "coalescing: at the lower offset");
    check_eq((long long)t.pool.free.begin()->second, 3 * kPoolGrain, "coalescing: the returned range");
    t.pool.give_off(5 * kPoolGrain, 4 * kPoolGrain);  // joins 9 above and 2 below
    check_eq((long long)t.pool.free.size(), 1, "coalescing: one range after the joins");
    check_eq((long long)t.pool.free.begin()->first, 2 * (long long)kPoolGrain, "coalescing: at the lowest offset");
    check_eq((long long)t.pool.free.begin()->second, 14 * kPoolGrain, "coalescing: whole");
    t.pool.give_off(0, 2 * kPoolGrain);
    check_eq((long long)t.pool.free.size(), 1, "coalescing: the pool is one range again");
    check_eq((long long)t.pool.free.begin()->first, 0, "coalescing: from the pool's start");
    check_eq((long long)t.pool.free.begin()->second, 16 * kPoolGrain, "coalescing: the pool's whole size");
  }
  {  // take_at: a kept file's offsets, on the grain and inside a free range
    TestPool t(16);
    check(t.pool.take_at(2 * kPoolGrain, 3 * kPoolGrain), "take_at: inside the free range");
    check_eq((long long)t.pool.free.size(), 2, "take_at: the range is split");
    check(!t.pool.take_at(3 * kPoolGrain, kPoolGrain), "take_at: across a range's end");
    check_eq(t.pool.largest(), 11 * kPoolGrain, "largest: the rest");
  }
  {  // round: the pool's grain
    check_eq((long long)Pool::round(0), 0, "round: zero");
    check_eq((long long)Pool::round(1), (long long)kPoolGrain, "round: one byte");
    check_eq((long long)Pool::round(kPoolGrain), (long long)kPoolGrain, "round: a grain");
    check_eq((long long)Pool::round(kPoolGrain + 1), 2 * (long long)kPoolGrain, "round: past a grain");
  }
}

// The page format state.cu writes and the store's disk tier checksums: every source, every array.
void test_geometry() {
  check_eq((long long)kPageBytes, 3645440, "geometry: a page");
  check_eq((long long)kPageBlock, 3670016, "geometry: a page as the pool hands it out");
  const long long want[4][4] = {{0, 524288, 589824, 720896},
                                {729088, 1253376, 1318912, 1449984},
                                {1458176, 1982464, 2048000, 2179072},
                                {2187264, 3235840, 3366912, 3629056}};
  for (int k = 0; k < 4; ++k)
    for (int a = 0; a < 4; ++a) {
      char what[64];
      std::snprintf(what, sizeof what, "geometry: source %d array %d", k, a);
      check_eq((long long)page_off(k, a), want[k][a], what);
    }
  uint32_t off[4];
  layout_of(2, off);
  check_eq(off[0], 0, "geometry: ratio 2 ckv");
  check_eq(off[1], 524288, "geometry: ratio 2 csc");
  check_eq(off[2], 589824, "geometry: ratio 2 ik");
  check_eq(off[3], 720896, "geometry: ratio 2 iksc");
  check_eq((long long)off[3] + (kPagePos / 2) * 4, 729088, "geometry: ratio 2's block");
  layout_of(1, off);
  check_eq(off[0], 0, "geometry: ratio 1 ckv");
  check_eq(off[1], 1048576, "geometry: ratio 1 csc");
  check_eq(off[2], 1179648, "geometry: ratio 1 ik");
  check_eq(off[3], 1441792, "geometry: ratio 1 iksc");
  check_eq((long long)off[3] + kPagePos * 4, 1458176, "geometry: ratio 1's block");
  for (int k = 0; k < 4; ++k) {
    layout_of(k < 3 ? 2 : 1, off);
    for (int a = 0; a < 4; ++a)
      check_eq((long long)page_off(k, a), (long long)(page_off(k, 0) + off[a]),
               "geometry: page_off is the source's block plus layout_of");
  }
  check_eq((long long)page_off(3, 3) + kPagePos * 4, (long long)kPageBytes, "geometry: the page's end");
  check_eq(pages_for(0), 0, "pages_for: nothing");
  check_eq(pages_for(1), 1, "pages_for: one position");
  check_eq(pages_for(kPagePos), 1, "pages_for: a page");
  check_eq(pages_for(kPagePos + 1), 2, "pages_for: past a page");
  check_eq(pages_for(1048576), 256, "pages_for: 1M positions");
  check_eq(max_pages_of(1048576), 257, "max_pages_of: 1M positions");
  check_eq((long long)shared_kv_overhead(16, 1048576), 3670016, "overhead: the zero page");
}

// A pool with the zero page taken out of it, as state_reserve leaves it.
struct Fixture {
  std::vector<uint8_t> buf;
  SharedKv kv;
  explicit Fixture(int blocks, int nslots, int max_pages) : buf((size_t)blocks * kPageBlock) {
    kv.pool.base = buf.data();
    kv.pool.size = buf.size();
    kv.pool.free.emplace(0, buf.size());
    uint8_t* zero = kv.pool.take(kPageBlock);
    kv.reset(nslots, max_pages, zero);
  }
  uint8_t* take() { return kv.pool.take(kPageBlock); }
};

void test_rows() {
  Fixture f(8, 2, 8);
  uint8_t* const zero = f.kv.zero;
  check(zero != nullptr, "row: the zero page is taken");
  check_eq((long long)f.kv.row_bytes(), 64, "row: eight entries");
  {  // a row's image: the session's pages, then the zero page
    uint8_t* pages[3] = {f.take(), f.take(), f.take()};
    f.kv.list(0, 3, pages);
    for (int i = 0; i < 3; ++i) check(f.kv.row(0)[i] == (uint64_t)(uintptr_t)pages[i], "row: a page");
    for (int i = 3; i < 8; ++i) check(f.kv.row(0)[i] == (uint64_t)(uintptr_t)zero, "row: the zero page above");
    for (int i = 0; i < 8; ++i) check(f.kv.row(1)[i] == (uint64_t)(uintptr_t)zero, "row: another slot's row");
    // one page more: one entry, the rest untouched
    uint8_t* p = f.take();
    check_eq(f.kv.add(0, p), 3, "row: the page's entry");
    check(f.kv.row(0)[3] == (uint64_t)(uintptr_t)p, "row: the page is in");
    check(f.kv.row(0)[4] == (uint64_t)(uintptr_t)zero, "row: the entry above is the zero page");
    check_eq((long long)(f.kv.rows.size()), 16, "row: two slots' rows");
  }
  {  // a full row: list then add to the last entry
    Fixture g(12, 1, 4);
    uint8_t* pages[3] = {g.take(), g.take(), g.take()};
    g.kv.list(0, 3, pages);
    check_eq(g.kv.add(0, g.take()), 3, "row: the last entry");
    check_eq((long long)g.kv.held[0], 4, "row: four entries held");
  }
}

void test_zero_page() {
  Fixture f(4, 1, 8);
  check(f.kv.zero != nullptr, "zero: taken");
  uint8_t* p[3];
  for (int i = 0; i < 3; ++i) {
    p[i] = f.take();
    check(p[i] != nullptr && p[i] != f.kv.zero, "zero: never handed out");
  }
  check(f.take() == nullptr, "zero: the pool is empty");
  for (int i = 0; i < 3; ++i) f.kv.pool.give(p[i], kPageBlock);  // everything back
  check_eq((long long)f.kv.pool.free.size(), 1, "zero: one free range again");
  check_eq((long long)f.kv.pool.free.begin()->first, (long long)kPageBlock, "zero: its block is not in the free map");
  check_eq((long long)f.kv.pool.free.begin()->second, 3 * (long long)kPageBlock, "zero: the pool's three blocks");
  for (int i = 0; i < 3; ++i) {
    uint8_t* q = f.take();
    check(q != nullptr && q != f.kv.zero, "zero: never handed out again");
  }
}

void test_drop() {
  Fixture f(8, 1, 8);
  uint8_t* zero = f.kv.zero;
  std::vector<uint8_t*> pages;
  size_t bytes = 0;
  const int n = 5;
  for (int i = 0; i < n; ++i) pages.push_back(f.take()), bytes += kPageBlock;
  f.kv.list(0, n, pages.data());
  f.kv.drop(0, pages, bytes, kPagePos);  // keep the page holding position 0's group: one
  check_eq((long long)pages.size(), 1, "drop: one page kept");
  check_eq((long long)bytes, (long long)kPageBlock, "drop: the pool's bytes in pages");
  check(f.kv.row(0)[0] == (uint64_t)(uintptr_t)pages[0], "drop: the kept page stays");
  for (int i = 1; i < 8; ++i) check(f.kv.row(0)[i] == (uint64_t)(uintptr_t)zero, "drop: the entries above");
  check_eq((long long)f.kv.held[0], 1, "drop: the row holds one page");
  check_eq((long long)f.kv.pool.free.size(), 1, "drop: one free range again");
  check_eq((long long)f.kv.pool.largest(), 6 * (long long)kPageBlock, "drop: the pages are back in the pool");
  for (int i = 0; i < 4; ++i) check(f.take() != nullptr, "drop: the freed pages are handed out again");
  f.kv.drop(0, pages, bytes, 0);  // nothing kept
  check(pages.empty() && bytes == 0, "drop: nothing left");
  for (int i = 0; i < 8; ++i) check(f.kv.row(0)[i] == (uint64_t)(uintptr_t)zero, "drop: the whole row is zero");
  f.kv.drop(0, pages, bytes, 3 * kPagePos + 1);  // an empty list, a position above it
  check(pages.empty() && bytes == 0, "drop: an empty list is unchanged");
}

void test_room() {
  Fixture f(6, 1, 8);
  std::vector<uint8_t*> pages;
  size_t bytes = 0;
  const auto never = [] { return false; };
  {  // room grows a list a page at a time
    check(f.kv.room(pages, bytes, kPagePos + 1, never), "room: two pages");
    check_eq((long long)pages.size(), 2, "room: two pages in the list");
    check_eq((long long)bytes, 2 * (long long)kPageBlock, "room: the pool's bytes in pages");
    check(f.kv.room(pages, bytes, kPagePos, never), "room: already there");
    check_eq((long long)pages.size(), 2, "room: nothing added");
  }
  {  // a full pool: false, and the list, the bytes and the pool as they were
    std::vector<uint8_t*> more;
    size_t bmore = 0;
    while (f.take()) {}
    const auto free_before = f.kv.pool.free;
    bool called = false;
    const auto record = [&called] {
      called = true;
      return false;
    };
    check(!f.kv.room(more, bmore, 3 * kPagePos, record), "room: the full pool");
    check(called, "room: it asked to evict");
    check(more.empty() && bmore == 0, "room: nothing added");
    check(f.kv.pool.free == free_before, "room: the pool is as it was");
    check(!f.kv.room(pages, bytes, 4 * kPagePos, never), "room: past the list");
    check_eq((long long)pages.size(), 2, "room: the list is as it was");
    check_eq((long long)bytes, 2 * (long long)kPageBlock, "room: the bytes are as they were");
    check(f.kv.pool.free == free_before, "room: the pool is as it was, again");
    // an eviction that frees a page: the room is there
    bool freed = false;
    const auto evict = [&] {
      if (freed) return false;
      freed = true;
      f.kv.pool.give(pages.back(), kPageBytes);
      bytes -= kPageBlock;
      pages.pop_back();
      return true;
    };
    check(f.kv.room(more, bmore, kPagePos, evict), "room: after an eviction");
    check_eq((long long)more.size(), 1, "room: the page came from the eviction");
  }
}

// --shared-kv (§2.7): the pages two sessions share. A fork's list takes its base's pages (SharedKv::hold)
// and a session's list gives a page up (SharedKv::release): the page goes back to the pool when the last
// session holding it does, in whatever order the sessions are freed. A spill writes a shared page to disk
// like any other; a session read back from disk gets new, unshared pages (state_unspill's store_alloc).
void test_shared() {
  const auto never = [] { return false; };
  // One session's list gives its pages up, as state_free, state_spill and SharedKv::drop do: a page the
  // session was the last holder of goes back to the pool; one another session still holds stays, and the
  // pool's bytes in pages (the store's counter, one for every session) stay with it.
  const auto give_up = [](SharedKv& kv, const std::vector<uint8_t*>& pages, size_t& bytes) {
    for (uint8_t* p : pages)
      if (kv.release(p)) kv.pool.give(p, kPageBytes), bytes -= kPageBlock;
  };
  {  // a fork of a base, the base freed first
    Fixture f(12, 1, 8);
    const size_t all = free_bytes(f.kv.pool);
    std::vector<uint8_t*> base, fork;
    size_t bytes = 0;
    check(f.kv.room(base, bytes, 3 * kPagePos, never), "shared: the base's three pages");
    check_eq((long long)free_bytes(f.kv.pool), (long long)(all - 3 * kPageBlock), "shared: three pages out");
    fork.assign(base.begin(), base.begin() + 2);  // the fork shares the first two
    f.kv.hold(fork.data(), 2);
    check_eq(f.kv.holders(base[0]), 2, "shared: two sessions hold the first");
    check_eq(f.kv.holders(base[1]), 2, "shared: two sessions hold the second");
    check_eq(f.kv.holders(base[2]), 1, "shared: the base alone holds the third");
    give_up(f.kv, base, bytes);  // the base goes: its own page comes back, the shared two stay
    check_eq(f.kv.holders(base[0]), 1, "shared: the fork still holds it");
    check_eq((long long)bytes, 2 * (long long)kPageBlock, "shared: the two shared pages are still out");
    check_eq((long long)free_bytes(f.kv.pool), (long long)(all - 2 * kPageBlock), "shared: only its own came back");
    give_up(f.kv, fork, bytes);  // the fork goes: both come back
    check_eq((long long)bytes, 0, "shared: the pool's bytes in pages");
    check_eq((long long)free_bytes(f.kv.pool), (long long)all, "shared: all three are back");
    check_eq(f.kv.holders(base[0]), 1, "shared: nobody shares it now");
    for (int i = 0; i < 3; ++i) check(f.take() != nullptr, "shared: handed out again");
  }
  {  // a fork of a fork: three sessions, the middle one freed first
    Fixture g(12, 1, 8);
    const size_t all = free_bytes(g.kv.pool);
    std::vector<uint8_t*> a, b, c;
    size_t bytes = 0;
    check(g.kv.room(a, bytes, 2 * kPagePos, never), "shared: the base's two pages");
    b.assign(a.begin(), a.end()), c.assign(a.begin(), a.end());
    g.kv.hold(b.data(), 2), g.kv.hold(c.data(), 2);
    check_eq(g.kv.holders(a[0]), 3, "shared: three sessions hold it");
    give_up(g.kv, b, bytes);  // the middle one goes: the base and the last one keep it
    check_eq(g.kv.holders(a[0]), 2, "shared: two left");
    check_eq((long long)free_bytes(g.kv.pool), (long long)(all - 2 * kPageBlock), "shared: nothing came back");
    give_up(g.kv, a, bytes);  // the base goes: the last one keeps it
    check_eq(g.kv.holders(c[0]), 1, "shared: one holder left");
    check_eq((long long)free_bytes(g.kv.pool), (long long)(all - 2 * kPageBlock), "shared: still nothing back");
    give_up(g.kv, c, bytes);  // the last one goes: both come back
    check_eq((long long)bytes, 0, "shared: the pool's bytes in pages");
    check_eq((long long)free_bytes(g.kv.pool), (long long)all, "shared: both are back");
  }
  {  // drop: a shared page above the position kept stays for the other session
    Fixture g(12, 1, 8);
    const size_t all = free_bytes(g.kv.pool);
    std::vector<uint8_t*> pages, fork;
    size_t bytes = 0;
    check(g.kv.room(pages, bytes, 4 * kPagePos, never), "shared: four pages");
    g.kv.list(0, 4, pages.data());
    fork = pages;  // a fork of all four
    g.kv.hold(fork.data(), 4);
    g.kv.drop(0, pages, bytes, 2 * kPagePos);  // the base keeps two: the two above are given up, but shared
    check_eq((long long)pages.size(), 2, "shared: two pages kept");
    check_eq((long long)bytes, 4 * (long long)kPageBlock, "shared: the pages are still out");
    check_eq(g.kv.holders(fork[3]), 1, "shared: the fork alone holds it now");
    check_eq((long long)free_bytes(g.kv.pool), (long long)(all - 4 * kPageBlock), "shared: nothing came back");
    check(g.kv.row(0)[2] == (uint64_t)(uintptr_t)g.kv.zero, "shared: the row's entry above is the zero page");
    check_eq((long long)g.kv.held[0], 2, "shared: the row holds two pages");
    give_up(g.kv, fork, bytes);  // the fork goes: the two it alone held come back
    check_eq((long long)bytes, 2 * (long long)kPageBlock, "shared: the base's two are still out");
    check_eq((long long)free_bytes(g.kv.pool), (long long)(all - 2 * kPageBlock), "shared: two are back");
    give_up(g.kv, pages, bytes);  // the base goes: the rest come back
    check_eq((long long)bytes, 0, "shared: the pool's bytes in pages");
    check_eq((long long)free_bytes(g.kv.pool), (long long)all, "shared: all four are back");
  }
}

// --shared-kv's sizing rule (docs/shared-kv.md §2.9, pages.h): a flag-off run reserves each live slot's
// caches at this --max-ctx and 12 GiB for the store; the flag's cap is 12 GiB plus what --slots would
// have reserved (an explicit --session-gib is the cap as given) and the flag's own blocks on top, and
// the pool is that bounded by the smallest rank's offer.
void test_sizing() {
  const uint64_t gib = 1073741824ull;
  // One slot at 1M covers --max-ctx in one unit: 3 x (1048576 / 2 + 64) + (1048576 + 64) = 2,621,696
  // entries of 356 B, 40 rings of 256 x (512 + 16) B and the four carries of 2 x 16 x 512 floats:
  // 938,992,640 B (895.4 MiB). 16 slots at 1M: 13.99 GiB.
  check_eq((long long)slot_caches_bytes(1, 0, 1048576), 938992640, "sizing: one slot at 1M");
  check_eq((long long)slot_caches_bytes(16, 0, 1048576), 15023882240, "sizing: 16 slots at 1M");
  // A test engine at 262144 and 4 slots: 3 x (131072 + 64) + (262144 + 64) entries: 239,068,160 B a
  // slot (228.0 MiB), 956,272,640 B the four (912.0 MiB).
  check_eq((long long)slot_caches_bytes(4, 0, 262144), 956272640, "sizing: 4 slots at 262144");
  // --big-slots: 4 slots of kUnitMul units of max_ctx / 8 and 12 of one (the model's nunits: 44 units
  // of 122,414,080 B). The flag refuses --big-slots; the arithmetic is the reservation it replaces.
  check_eq((long long)slot_caches_bytes(16, 4, 1048576), 5386219520, "sizing: 4 big and 12 small at 1M");
  check_eq((long long)slot_caches_bytes(1, 0, 131072), 122414080, "sizing: one unit at 1M / 8");
  check_eq((long long)slot_caches_bytes(0, 0, 1048576), 0, "sizing: no slots");
  // The cap: 12 GiB + the slots' 13.99 GiB, plus the flag's own block in the pool (the zero page,
  // 3,670,016 B; the slot table is the model's own allocation): 25.996 GiB.
  const uint64_t cap = shared_kv_cap(12, false, 16, 0, 1048576);
  check_eq((long long)cap, 27912454144, "sizing: the cap at the service's flags");
  check_eq((long long)cap, (long long)(12 * gib + slot_caches_bytes(16, 0, 1048576) + kPageBlock),
           "sizing: the cap's parts");
  // An explicit --session-gib is the cap as it is given (plus the zero page).
  check_eq((long long)shared_kv_cap(4, true, 16, 0, 1048576), (long long)(4 * gib + kPageBlock),
           "sizing: an explicit --session-gib");
  check_eq((long long)shared_kv_cap(0.02, true, 2, 0, 65536),
           (long long)(0.02 * 1073741824.0) + kPageBlock, "sizing: a test's small --session-gib");
  // A test engine at 262144 and 4 slots: 12 GiB + 912.0 MiB + the zero page = 12.89 GiB.
  check_eq((long long)shared_kv_cap(12, false, 4, 0, 262144), 13844844544, "sizing: the cap at 262144");
  // The pool: the cap, bounded by every rank's offer (the smallest is taken).
  check_eq((long long)shared_kv_pool(cap, cap + gib), (long long)cap, "sizing: an offer above the cap");
  check_eq((long long)shared_kv_pool(cap, 6 * gib), (long long)(6 * gib), "sizing: an offer below it");
  check_eq((long long)shared_kv_pool(cap, cap), (long long)cap, "sizing: an offer equal to it");
  // The pages a pool holds besides the flag's own blocks, and the positions they cover.
  check_eq(pool_pages(cap, 16, 1048576), 7604, "sizing: the service's pages");
  check_eq((long long)pool_pages(cap, 16, 1048576) * kPagePos, 31145984, "sizing: the service's positions");
  check_eq(pool_pages(kPageBlock + 3735552, 16, 1048576), 1, "sizing: a one-page pool");
  check_eq(pool_pages(3735552, 16, 1048576), 0, "sizing: a pool of its own blocks alone");
}

// Which session is live in which slot: what state_slot_row sets and the flag's room calls read.
void test_slots() {
  SlotSessions ss;
  for (int s = 0; s < model::kMaxSlots; ++s) check_eq(ss.of(s), 0, "slots: a fresh slot has no session");
  check_eq(ss.slot_of(0), -1, "slots: 0 is no session");
  check_eq(ss.slot_of(7), -1, "slots: a session nobody holds");
  ss.set(3, 7);
  check_eq(ss.of(3), 7, "slots: the session");
  check_eq(ss.slot_of(7), 3, "slots: its slot");
  check_eq(ss.of(2), 0, "slots: a neighbour");
  check_eq(ss.slot_of(9), -1, "slots: another session");
  ss.set(3, -1);  // the golden and bench modes' handles are negative
  check_eq(ss.of(3), -1, "slots: a negative handle");
  check_eq(ss.slot_of(-1), 3, "slots: its slot");
  check_eq(ss.slot_of(7), -1, "slots: the session it replaced is gone");
  ss.set(3, 0);  // the slot is let go
  check_eq(ss.of(3), 0, "slots: let go");
  check_eq(ss.slot_of(-1), -1, "slots: and the session with it");
  check_eq(ss.of(-1), 0, "slots: a slot below the range");
  check_eq(ss.of(model::kMaxSlots), 0, "slots: a slot past the range");
  check_eq(ss.of(999), 0, "slots: a slot far past the range");
  ss.set(0, 1);
  ss.set(model::kMaxSlots - 1, 2);
  check_eq(ss.slot_of(1), 0, "slots: the first");
  check_eq(ss.slot_of(2), model::kMaxSlots - 1, "slots: the last");
  check_eq(ss.of(1), 0, "slots: the first slot still holds its own session");
}

}  // namespace

int main() {
  test_pool();
  test_geometry();
  test_rows();
  test_zero_page();
  test_drop();
  test_room();
  test_shared();
  test_sizing();
  test_slots();
  std::printf("page_tests: %d checks, %d failed\n", checks, failed);
  return failed ? 1 : 0;
}
