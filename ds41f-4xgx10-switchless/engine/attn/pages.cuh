// The shared KV pool's pages, as a kernel sees them (docs/shared-kv.md §2.3). A page holds 4096
// positions of all four kv sources at the store's page format, so entry e of a source's cache is at
// `tab[e >> shift] + off[a] + (e & ((1 << shift) - 1)) * bytes`; `tab` is null for the contiguous path
// (--shared-kv off, and the drafter's own windows), and the helpers are then exactly the pointer
// arithmetic of before.
#pragma once
#include <cstddef>
#include <cstdint>

namespace attn {

// A kv source's four caches (ckv 256 B, csc 32, ik 64, iksc 4 per entry) in the shared pool's pages,
// or the slot's contiguous caches. `slot` gives each step row its slot (ops::StepIn::slot), `stride` the
// slot table's entries; a null `slot` means `tab` is this one slot's row (the eager paths, use_slot).
struct KvPages {
  const uint64_t* tab = nullptr;  // [stride] page base pointers; null: contiguous
  const int* slot = nullptr;      // step rows -> slots; null: `tab` is this slot's row
  uint32_t stride = 0;            // entries per slot (max_pages)
  uint32_t off[4] = {};           // bytes from a page's base: ckv, csc, ik, iksc (page_off)
  int shift = 11;                 // log2(entries per page): 11 (ratio 2) or 12 (ratio 1)
  __host__ __device__ bool paged() const { return tab != nullptr; }  // the host picks the kernel instance
};

// The row's own table (one load per step row, not per entry): row r's slot's.
__device__ __forceinline__ KvPages pages_at(KvPages p, int r) {
  if (p.slot) {
    p.tab += (size_t)p.slot[r] * p.stride;
    p.slot = nullptr;
  }
  return p;
}

// Entry e of array a (0 ckv, 1 csc, 2 ik, 3 iksc) as a pointer. kPaged is the process's flag as a
// template parameter: the false instantiation compiles to `base + e * bytes`, as before.
template <bool kPaged>
__device__ __forceinline__ const uint8_t* page_ent(const KvPages& p, const uint8_t* base, int e, int a, size_t bytes) {
  if constexpr (!kPaged) return base + (size_t)e * bytes;
  return reinterpret_cast<const uint8_t*>(p.tab[e >> p.shift]) + p.off[a] +
         (size_t)(e & ((1 << p.shift) - 1)) * bytes;
}

template <bool kPaged>
__device__ __forceinline__ uint8_t* page_ent_m(const KvPages& p, uint8_t* base, int e, int a, size_t bytes) {
  if constexpr (!kPaged) return base + (size_t)e * bytes;
  return reinterpret_cast<uint8_t*>(p.tab[e >> p.shift]) + p.off[a] + (size_t)(e & ((1 << p.shift) - 1)) * bytes;
}

}  // namespace attn
