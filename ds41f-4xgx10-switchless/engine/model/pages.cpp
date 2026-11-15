// The shared KV pool's host side (pages.h): the best-fit allocator, the page layout and the slots' table
// rows. No CUDA: state.cu and model.cu drive the device memory and the copies.
#include "model/pages.h"

#include <algorithm>

namespace model::detail {

size_t Pool::take_off(size_t bytes) {
  const size_t n = round(bytes);
  auto best = free.end();
  for (auto it = free.begin(); it != free.end(); ++it)
    if (it->second >= n && (best == free.end() || it->second < best->second)) best = it;
  if (best == free.end()) return kNone;
  const size_t off = best->first, len = best->second;
  free.erase(best);
  if (len > n) free.emplace(off + n, len - n);
  return off;
}

uint8_t* Pool::take(size_t bytes) {
  const size_t off = take_off(bytes);
  return off == kNone ? nullptr : base + off;
}

void Pool::give(uint8_t* p, size_t bytes) { give_off((size_t)(p - base), bytes); }

bool Pool::take_at(size_t off, size_t bytes) {
  const size_t n = round(bytes);
  auto it = free.upper_bound(off);
  if (it == free.begin()) return false;
  --it;
  const size_t fo = it->first, fl = it->second;
  if (off + n > fo + fl) return false;
  free.erase(it);
  if (off > fo) free.emplace(fo, off - fo);
  if (off + n < fo + fl) free.emplace(off + n, fo + fl - off - n);
  return true;
}

size_t Pool::largest() const {
  size_t n = 0;
  for (const auto& f : free) n = std::max(n, f.second);
  return n;
}

void Pool::give_off(size_t off, size_t bytes) {
  size_t len = round(bytes);
  auto next = free.lower_bound(off);
  if (next != free.end() && next->first == off + len) {
    len += next->second;
    next = free.erase(next);
  }
  if (next != free.begin()) {
    const auto prev = std::prev(next);
    if (prev->first + prev->second == off) {
      off = prev->first;
      len += prev->second;
      free.erase(prev);
    }
  }
  free.emplace(off, len);
}

// A source's block in the page, then its arrays: ckv, csc, ik, iksc.
size_t page_off(int k, int a) {
  const int ratio = k < 3 ? 2 : 1;
  uint32_t off[4];
  layout_of(ratio, off);
  size_t base = 0;
  for (int j = 0; j < k; ++j) base += (size_t)(kPagePos / (j < 3 ? 2 : 1)) * kEntryBytes;
  return base + off[a];
}

namespace {
// One unit's working state at this unit_ctx, as model.cu allocates it: the four kv sources' caches
// (cap = unit_ctx / ratio + 64 entries of 356 B), a window ring a layer and the four carries.
uint64_t unit_state_bytes(int unit_ctx) {
  uint64_t caches = 0;
  for (const int L : {2, 8, 14, 20}) caches += ((uint64_t)unit_ctx / model::ratio_of(L) + 64) * kEntryBytes;
  const uint64_t rings = (uint64_t)model::kLayers * ((uint64_t)kRingSlots * model::kHead +       // e4m3
                                                     (uint64_t)kRingSlots * model::kHead / 32);  // ue8m0, one per 32
  const uint64_t carries = (uint64_t)model::kKvSlots * 2 * kCarrySlots * model::kHead * sizeof(float);
  return caches + rings + carries;
}
}  // namespace

uint64_t slot_caches_bytes(int slots, int big_slots, int max_ctx) {
  if (slots < 1 || max_ctx < 1) return 0;
  const int nbig = big_slots > 0 && big_slots < slots ? big_slots : slots;
  const int mul = nbig < slots ? model::kUnitMul : 1;
  return (uint64_t)(nbig * mul + (slots - nbig)) * unit_state_bytes(max_ctx / mul);
}

uint64_t shared_kv_cap(double session_gib, bool given, int slots, int big_slots, int max_ctx) {
  const uint64_t base = given ? (uint64_t)(session_gib * 1073741824.0)
                              : (uint64_t)(kSessionGib * 1073741824.0) + slot_caches_bytes(slots, big_slots, max_ctx);
  return base + shared_kv_overhead(slots, max_ctx);
}

void SharedKv::reset(int nslots_, int max_pages_, uint8_t* zero_page) {
  nslots = nslots_, max_pages = max_pages_, zero = zero_page;
  rows.assign((size_t)nslots * max_pages, (uint64_t)(uintptr_t)zero);
  held.assign(nslots, 0);
  refs.clear();
}

void SharedKv::list(int slot, int n, const uint8_t* const* pages) {
  uint64_t* r = rows.data() + (size_t)slot * max_pages;
  for (int i = 0; i < n; ++i) r[i] = (uint64_t)(uintptr_t)pages[i];
  for (int i = n; i < max_pages; ++i) r[i] = (uint64_t)(uintptr_t)zero;
  held[slot] = n;
}

int SharedKv::add(int slot, uint8_t* page) {
  const int i = held[slot]++;
  rows[(size_t)slot * max_pages + i] = (uint64_t)(uintptr_t)page;
  return i;
}

void SharedKv::drop(int slot, std::vector<uint8_t*>& pages, size_t& bytes, int from) {
  const int n = pages_for(from);
  while ((int)pages.size() > n) {
    if (release(pages.back())) pool.give(pages.back(), kPageBytes), bytes -= Pool::round(kPageBytes);
    pages.pop_back();
  }
  if (slot >= 0) {  // the row mirrors the list: the entries above it back to the zero page
    const int keep = std::min(n, (int)pages.size());
    uint64_t* r = rows.data() + (size_t)slot * max_pages;
    for (int i = keep; i < max_pages; ++i) r[i] = (uint64_t)(uintptr_t)zero;
    held[slot] = keep;
  }
}

// --shared-kv (§2.7): a fork's list takes `n` pages another session holds. The first session to share a
// page makes its entry (two holders), every later one adds to it.
void SharedKv::hold(const uint8_t* const* pages, int n) {
  for (int i = 0; i < n; ++i) {
    const auto it = refs.find(pages[i]);
    if (it == refs.end()) refs.emplace(pages[i], 2);
    else ++it->second;
  }
}

bool SharedKv::release(const uint8_t* page) {
  const auto it = refs.find(page);
  if (it == refs.end()) return true;  // nobody shares it: the session giving it up was its only holder
  if (--it->second > 1) return false;
  refs.erase(it);  // one holder left: it keeps the page, and its hold is uncounted again
  return false;
}

int SharedKv::holders(const uint8_t* page) const {
  const auto it = refs.find(page);
  return it == refs.end() ? 1 : it->second;
}

bool SharedKv::room(std::vector<uint8_t*>& pages, size_t& bytes, int upto, const Evict& evict) {
  const int want = pages_for(upto);
  const size_t have = pages.size();
  while ((int)pages.size() < want) {
    if (uint8_t* p = pool.take(kPageBytes)) {
      pages.push_back(p);
      bytes += Pool::round(kPageBytes);
      continue;
    }
    if (evict()) continue;
    while (pages.size() > have) {  // no room and nothing to evict: as it was
      pool.give(pages.back(), kPageBytes);
      bytes -= Pool::round(kPageBytes);
      pages.pop_back();
    }
    return false;
  }
  return true;
}

}  // namespace model::detail
