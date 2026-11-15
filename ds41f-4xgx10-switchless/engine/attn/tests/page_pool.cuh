// The kernel tests' shared pool: one pool whose pages hold all four kv sources (docs/shared-kv.md §2.3),
// each source's four arrays at its own offsets in a page (model/pages.h), so a --pages run lays the
// caches out the way the engine's page_view does instead of one source alone in a page. `bad` builds
// the views the way the engine did before commit 5f0d039 (layout_of(ratio): a ratio-2 source other than
// the first lands in the first one's block), the --pages-badlayout control.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "attn/pages.cuh"
#include "model/pages.h"
#include "testutil.h"

namespace attn_test {

// Source k's array a from a page's base: what pages.cpp's page_off returns (the sources lie one after
// another, each in its own block, a ratio-2 source's block half a ratio-1 source's). The attn tests do
// not link the model's host side, so the offsets are built from pages.h's constants here.
inline uint32_t src_off(int k, int a, bool bad) {
  const int ratio = k < 3 ? 2 : 1;
  uint32_t off[4];
  model::detail::layout_of(ratio, off);
  if (bad) return off[a];
  size_t base = 0;
  for (int j = 0; j < k; ++j)
    base += (size_t)(model::detail::kPagePos / (j < 3 ? 2 : 1)) * model::detail::kEntryBytes;
  return (uint32_t)(base + off[a]);
}

// `slots` rows of `pages` pages each (the longest source's), a zero page above every row's list, and the
// pages in reverse address order so that no two neighbours are adjacent.
struct PagePool {
  static constexpr size_t kArr[4] = {256, 32, 64, 4};  // per entry: ckv, csc, ik, iksc
  int slots = 0, pages = 0, max_pages = 0, cap[4] = {}, npg[4] = {};
  uint32_t off[4][4] = {};
  uint8_t* zero = nullptr;
  Dev<uint8_t> mem;
  Dev<uint64_t> tab;
  PagePool(int cap2, int cap1, int slots_, bool bad) : slots(slots_) {  // cap2 <= 0: nothing is allocated
    if (cap2 <= 0) return;
    for (int k = 0; k < 4; ++k) {
      const int ratio = k < 3 ? 2 : 1, E = model::detail::kPagePos / ratio;
      cap[k] = k < 3 ? cap2 : cap1;
      npg[k] = (cap[k] + E - 1) / E;
      pages = std::max(pages, npg[k]);
      for (int a = 0; a < 4; ++a) off[k][a] = src_off(k, a, bad);
    }
    max_pages = pages + 4;
    mem = Dev<uint8_t>((size_t)slots * pages * model::detail::kPageBlock + model::detail::kPageBlock);
    zero = mem.p + (size_t)slots * pages * model::detail::kPageBlock;
    std::vector<uint64_t> h((size_t)slots * max_pages);
    for (int s = 0; s < slots; ++s)
      for (int i = 0; i < max_pages; ++i)
        h[(size_t)s * max_pages + i] = (uint64_t)(uintptr_t)(i < pages ? page(s, i) : zero);
    tab = Dev<uint64_t>(h.data(), h.size());
  }
  uint8_t* page(int s, int i) const {
    return mem.p + ((size_t)s * pages + (pages - 1 - i)) * model::detail::kPageBlock;
  }
  // Source k's view of slot s's row, as Model::Impl::page_view builds it; the second form is a step's rows,
  // their slots from the step's table (page_view's s < 0).
  attn::KvPages view(int k, int s) const {
    attn::KvPages p{};
    p.tab = tab.p + (s < 0 ? 0 : (size_t)s * max_pages);
    p.stride = (uint32_t)max_pages;
    for (int a = 0; a < 4; ++a) p.off[a] = off[k][a];
    p.shift = k < 3 ? 11 : 12;
    return p;
  }
  attn::KvPages view(int k, const int* slots) const {  // a step's rows
    attn::KvPages p = view(k, -1);
    p.slot = slots;
    return p;
  }
  attn::Compressed ckv(int k, int s, int n) const { return {nullptr, nullptr, n, nullptr, 0, 0, view(k, s)}; }
  attn::Compressed ckv(int k, const int* slots, int n) const {
    return {nullptr, nullptr, n, nullptr, 0, 0, view(k, slots)};
  }
  attn::IndexKeys ik(int k, int s, int n) const { return {nullptr, nullptr, n, nullptr, 0, 0, view(k, s)}; }
  attn::IndexKeys ik(int k, const int* slots, int n) const {
    return {nullptr, nullptr, n, nullptr, 0, 0, view(k, slots)};
  }
  // A contiguous array of source k's `cap[k]` entries <-> slot s's pages (array a).
  void put(int k, int a, int s, const uint8_t* src, cudaStream_t st) const {
    const int E = model::detail::kPagePos / (k < 3 ? 2 : 1);
    for (int i = 0; i < npg[k]; ++i) {
      const size_t n = std::min<size_t>(E, (size_t)cap[k] - (size_t)i * E) * kArr[a];
      CUDA_CHECK(cudaMemcpyAsync(page(s, i) + off[k][a], src + (size_t)i * E * kArr[a], n,
                                 cudaMemcpyDeviceToDevice, st));
    }
  }
  void get(int k, int a, int s, uint8_t* dst, cudaStream_t st) const {
    const int E = model::detail::kPagePos / (k < 3 ? 2 : 1);
    for (int i = 0; i < npg[k]; ++i) {
      const size_t n = std::min<size_t>(E, (size_t)cap[k] - (size_t)i * E) * kArr[a];
      CUDA_CHECK(cudaMemcpyAsync(dst + (size_t)i * E * kArr[a], page(s, i) + off[k][a], n,
                                 cudaMemcpyDeviceToDevice, st));
    }
  }
};

}  // namespace attn_test
