// What the shared pool's page indirection costs the three decode kernels that touch the caches
// (docs/shared-kv.md §2.4): the compressor's emit (compressor.cu's emit_kernel, one CTA a row, the kernel the
// step waits for), the decode indexer (index_dense) and the compressed attention (sparse_attn). Each case runs
// with its caches in contiguous arrays and again with them in a realistic pool: one 4 GiB allocation (as the
// engine reserves), the session's pages placed far apart in it, a zero page above its list and the table row
// the kernels read. Every timed launch follows an L2 flush, as bench_index does: the engine's step streams
// ~60 MB a step, so its L2 is cold the same way, and the paged case's table entry and store lines are then as
// cold as they are in a step. Each case is the median of --iters launches. Its outputs are compared with the
// contiguous run's bit for bit, and the contiguous launch runs twice first: a difference there is the run's
// nondeterminism, not the pages'. Every snapshot follows a stream sync: the kernels run on a non-blocking
// stream, so a host copy is not ordered with them, and an unsynced reference is a buffer the kernel may not
// have written yet.
//   bench_emit [--iters N] [--iqm] [--rows 1,6,16] [--kernels emit,index,attn] ctx...
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "fill.cuh"
#include "testutil.h"

using namespace attn_test;

namespace {

constexpr int kTopk = 512, kPool = 2048, kBucket = 4096, kSplits = kTopk / 64 + 2;
constexpr int kHead = attn::kHead, kIdxDim = attn::kIdxDim;
constexpr size_t kArena = 4ull << 30;                 // the pool's allocation
constexpr int kPagePos = 4096;                        // positions a page holds (model/pages.h)
constexpr size_t kArrBytes[4] = {256, 32, 64, 4};     // per entry: KV, KV scales, index key, key scales
constexpr size_t kEntryBytes = 356;

cudaStream_t g_s;
int g_iters = 30, g_bad = 0;
bool g_iqm = false;  // --iqm: report the interquartile mean (the timer ticks every ~2 us)
Dev<uint8_t> g_flush, g_arena;
Dev<uint32_t> g_sink;

// reads the flush buffer: the L2 then holds clean lines only, as in a step, where weights stream through
// (a memset would leave it dirty, and the timed kernel would pay for the write-backs)
__global__ void flush_kernel(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}

// median (or interquartile mean) time of f in microseconds, each launch after an L2 flush
template <class F>
double time_us(F&& f) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> t;
  for (int i = 0; i < g_iters + 2; ++i) {
    flush_kernel<<<192, 512, 0, g_s>>>(reinterpret_cast<const uint4*>(g_flush.p), g_flush.n / 16, g_sink.p);
    CUDA_CHECK(cudaEventRecord(e0, g_s));
    f();
    CUDA_CHECK(cudaEventRecord(e1, g_s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (i >= 2) t.push_back(1e3 * ms);
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  std::sort(t.begin(), t.end());
  if (g_iqm) {
    double sum = 0;
    for (size_t i = t.size() / 4; i < t.size() - t.size() / 4; ++i) sum += t[i];
    return sum / (double)(t.size() - 2 * (t.size() / 4));
  }
  return t[t.size() / 2];
}

// The store's page geometry (model/pages.h's layout_of/page_off/kPageBlock) for kv source k: 0-2 ratio 2,
// 3 ratio 1, the sources lying one after another in the page.
struct PageGeom {
  int ratio, shift;
  size_t E, off[4], block;
  explicit PageGeom(int k) : ratio(k < 3 ? 2 : 1), shift(k < 3 ? 11 : 12) {
    E = (size_t)kPagePos / ratio;
    size_t o = 0, base = 0;
    for (int j = 0; j < k; ++j) base += (size_t)(kPagePos / (j < 3 ? 2 : 1)) * kEntryBytes;
    for (int a = 0; a < 4; ++a) off[a] = base + o, o += E * kArrBytes[a];
    block = ((size_t)(kPagePos / 2 * 3 + kPagePos) * kEntryBytes + 65535) / 65536 * 65536;
  }
};

// One case's caches in the pool: `npg` pages of the shared 4 GiB arena placed far apart (a stride of the arena
// over the pages, the last page first, so neighbours are not adjacent), a zero page at the arena's end and the
// table row the kernels read. `put`/`gather` move an array's bytes between the contiguous caches and the pages.
struct Pool {
  PageGeom g;
  int cap = 0, npg = 0, max_pages = 0;
  size_t stride = 0;
  uint8_t* zero = nullptr;
  Dev<uint64_t> tab;
  Pool(int k, int cap_) : g(k), cap(cap_) {
    npg = (cap + (int)g.E - 1) / (int)g.E;
    max_pages = npg + 2;
    stride = (kArena - g.block) / (size_t)(npg + 1) / 65536 * 65536;  // pages on the pool's 64 KiB grain, as the engine's
    zero = g_arena.p + kArena - g.block;
    std::vector<uint64_t> h((size_t)max_pages);
    for (int i = 0; i < max_pages; ++i)
      h[(size_t)i] = (uint64_t)(uintptr_t)(i < npg ? page(i) : zero);
    tab = Dev<uint64_t>(h.data(), h.size());
  }
  uint8_t* page(int i) const { return g_arena.p + (size_t)(npg - 1 - i) * stride; }
  attn::KvPages view() const {  // one session: the row itself, no step's slot table
    attn::KvPages p{};
    p.tab = tab.p;
    p.stride = (uint32_t)max_pages;
    for (int a = 0; a < 4; ++a) p.off[a] = (uint32_t)g.off[a];
    p.shift = g.shift;
    return p;
  }
  void put(int a, const uint8_t* src, size_t bytes) {  // the contiguous array into the pages
    for (int i = 0; i < npg; ++i) {
      const size_t n = std::min(g.E, (size_t)cap - (size_t)i * g.E);
      CUDA_CHECK(cudaMemcpyAsync(page(i) + g.off[a], src + (size_t)i * g.E * bytes, n * bytes,
                                 cudaMemcpyDeviceToDevice, g_s));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  void gather(int a, uint8_t* dst, size_t bytes) const {  // the pages' array a into a contiguous buffer
    for (int i = 0; i < npg; ++i) {
      const size_t n = std::min(g.E, (size_t)cap - (size_t)i * g.E);
      CUDA_CHECK(cudaMemcpyAsync(dst + (size_t)i * g.E * bytes, page(i) + g.off[a], n * bytes,
                                 cudaMemcpyDeviceToDevice, g_s));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
};

// the bucket a step at `ctx` is captured for (the runner's graph), and the entries its caches hold
int bucket_of(int ctx) { return (ctx + kBucket - 1) / kBucket * kBucket; }
int cap_of(int ctx, int ratio) { return bucket_of(ctx) / ratio + 64; }

// the positions of `rows` consecutive decode rows at the end of `ctx` (ascending, one session)
std::vector<int> positions(int ctx, int rows) {
  std::vector<int> h(rows);
  for (int r = 0; r < rows; ++r) h[r] = ctx - rows + r;
  return h;
}

// ---- the comparison ---------------------------------------------------------------------------------

// A case's outputs as bytes, each array with its name and element size (bytes): a mismatch is reported in the
// case's own terms (`where` names the element, `val` its two values).
struct Snap {
  std::vector<std::vector<uint8_t>> arr;
  std::vector<std::string> name;
  std::vector<size_t> esz;
  void add(const char* n, size_t e, std::vector<uint8_t> bytes) {
    name.push_back(n);
    esz.push_back(e);
    arr.push_back(std::move(bytes));
  }
};

// The first element two snapshots differ in, and how many elements differ (count 0: equal). Elements are the
// snapshot's: a bf16 output or a 32-bit selection counts as one.
struct Diff {
  size_t array = 0, elem = 0, count = 0, total = 0;
  uint64_t ref = 0, got = 0;
  bool same() const { return count == 0; }
};

inline uint64_t le_bytes(const uint8_t* p, size_t n) {
  uint64_t v = 0;
  for (size_t i = 0; i < n; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

inline Diff diff_of(const Snap& ref, const Snap& got) {
  Diff d;
  for (size_t a = 0; a < ref.arr.size(); ++a) {
    const size_t esz = ref.esz[a], n = std::min(ref.arr[a].size(), got.arr[a].size()) / esz;
    d.total += n;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t* r = ref.arr[a].data() + i * esz;
      const uint8_t* g = got.arr[a].data() + i * esz;
      if (!std::memcmp(r, g, esz)) continue;
      if (!d.count) d.array = a, d.elem = i, d.ref = le_bytes(r, esz), d.got = le_bytes(g, esz);
      ++d.count;
    }
  }
  return d;
}

// a device array's bytes, for the snapshot
template <class T>
std::vector<uint8_t> bytes_of(const Dev<T>& d) {
  const std::vector<T> h = d.host();
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(h.data()),
                              reinterpret_cast<const uint8_t*>(h.data() + h.size()));
}

// ---- the compressor's emit -------------------------------------------------------------------------

// compressor_emit: `rows` decode rows' projections, one CTA a row, the entries into the caches.
struct EmitCase {
  int rows, ctx, k, cap;
  Dev<int> pos;
  Dev<float> proj, score, carry_kv, carry_sc;
  Dev<float2> tab;
  Dev<uint16_t> cnorm, wk, knorm;
  Dev<uint8_t> ckv_kv, ckv_sc, ik_k, ik_sc;
  Dev<uint8_t> got_ckv, got_sc, got_ik, got_iksc;  // the paged run's arrays, gathered for the comparison
  std::optional<Pool> pool;
  EmitCase(int rows_, int ctx_, int k_) : rows(rows_), ctx(ctx_), k(k_), cap(cap_of(ctx_, k_ < 3 ? 2 : 1)) {
    const std::vector<int> h = positions(ctx, rows);
    pos = Dev<int>(h.data(), h.size());
    tab = Dev<float2>((size_t)(bucket_of(ctx) + 1) * 32);
    attn::rope_table_build(tab.p, bucket_of(ctx) + 1, true, g_s);
    proj = Dev<float>((size_t)rows * kHead);
    score = Dev<float>((size_t)rows * kHead);
    fill::f32(proj, 1.f);
    fill::f32(score, 1.f);
    carry_kv = Dev<float>((size_t)16 * kHead);
    carry_sc = Dev<float>((size_t)16 * kHead);
    fill::f32(carry_kv, 0.5f);
    fill::f32(carry_sc, 0.5f);
    cnorm = Dev<uint16_t>(kHead);
    wk = Dev<uint16_t>((size_t)kIdxDim * kHead);  // the 128 KB the emit reads
    knorm = Dev<uint16_t>(kIdxDim);
    fill::bf16_range(cnorm, 0.5f, 1.5f);
    fill::bf16(wk, 0.05f);
    fill::bf16_range(knorm, 0.5f, 1.5f);
    ckv_kv = Dev<uint8_t>((size_t)cap * 256);
    ckv_sc = Dev<uint8_t>((size_t)cap * 32);
    ik_k = Dev<uint8_t>((size_t)cap * 64);
    ik_sc = Dev<uint8_t>((size_t)cap * 4);
    got_ckv = Dev<uint8_t>((size_t)cap * 256);
    got_sc = Dev<uint8_t>((size_t)cap * 32);
    got_ik = Dev<uint8_t>((size_t)cap * 64);
    got_iksc = Dev<uint8_t>((size_t)cap * 4);
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  Pool& pages() {
    if (!pool) {
      pool.emplace(k, cap);
      pool->put(0, ckv_kv.p, 256);
      pool->put(1, ckv_sc.p, 32);
      pool->put(2, ik_k.p, 64);
      pool->put(3, ik_sc.p, 4);
    }
    return *pool;
  }
  void run(bool paged) {
    attn::Compressed cc{ckv_kv.p, ckv_sc.p, cap};
    attn::IndexKeys ik{ik_k.p, ik_sc.p, cap};
    if (paged) {
      cc.pages = pages().view();
      ik.pages = pages().view();
    }
    const attn::CompressorWeights w{k < 3 ? 2 : 1, nullptr, nullptr, cnorm.p, wk.p, knorm.p};
    attn::compressor_emit(proj.p, k < 3 ? score.p : nullptr, pos.p, rows, w,
                          attn::CompressorCarry{carry_kv.p, carry_sc.p}, tab.p, cc, ik, nullptr, 0, g_s);
  }
  Snap snap(bool paged) {
    Snap s;
    if (!paged) {
      s.add("ckv", 1, bytes_of(ckv_kv)), s.add("ckv scales", 1, bytes_of(ckv_sc));
      s.add("index keys", 1, bytes_of(ik_k)), s.add("index key scales", 1, bytes_of(ik_sc));
    } else {
      Pool& p = pages();
      p.gather(0, got_ckv.p, 256);
      p.gather(1, got_sc.p, 32);
      p.gather(2, got_ik.p, 64);
      p.gather(3, got_iksc.p, 4);
      s.add("ckv", 1, bytes_of(got_ckv)), s.add("ckv scales", 1, bytes_of(got_sc));
      s.add("index keys", 1, bytes_of(got_ik)), s.add("index key scales", 1, bytes_of(got_iksc));
    }
    return s;
  }
  std::string where(size_t a, size_t e) const {  // a byte of an entry
    return "entry " + std::to_string(e / kArrBytes[a]) + " byte " + std::to_string(e % kArrBytes[a]);
  }
  std::string val(size_t /*a*/, uint64_t b) const {
    char t[8];
    std::snprintf(t, sizeof(t), "0x%02x", (unsigned)b);
    return t;
  }
};

// ---- the decode indexer ---------------------------------------------------------------------------

// index_dense: the dense scan of the context's keys for `rows` rows (layers 2/8/14 at ratio 2, layer 20 at
// ratio 1 with its block-maxima pool).
struct IndexCase {
  int rows, ctx, k, ratio, cap, max_scan, kstride;
  bool pool_sel;
  Dev<int> pos, top, pl, counters;
  Dev<float2> tab;
  Dev<uint8_t> ik_k, ik_sc;
  Dev<uint16_t> iq, keys, bkeys, tkey, pkey;
  Dev<float> wp;
  Dev<uint32_t> scratch;
  std::optional<Pool> pool;
  IndexCase(int rows_, int ctx_, int k_)
      : rows(rows_), ctx(ctx_), k(k_), ratio(k_ < 3 ? 2 : 1), pool_sel(k_ == 3), cap(cap_of(ctx_, k_ < 3 ? 2 : 1)) {
    max_scan = bucket_of(ctx) / ratio;
    kstride = (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    const std::vector<int> h = positions(ctx, rows);
    pos = Dev<int>(h.data(), h.size());
    tab = Dev<float2>((size_t)(bucket_of(ctx) + 1) * 32);
    attn::rope_table_build(tab.p, bucket_of(ctx) + 1, true, g_s);
    ik_k = Dev<uint8_t>((size_t)cap * 64);
    ik_sc = Dev<uint8_t>((size_t)cap * 4);
    fill::u8(ik_k, 0xFF, 0);
    fill::u8(ik_sc, 7, 120);
    iq = Dev<uint16_t>((size_t)rows * attn::kIdxHeads * kIdxDim);
    wp = Dev<float>((size_t)rows * attn::kIdxHeads);
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    keys = Dev<uint16_t>((size_t)rows * kstride);
    bkeys = Dev<uint16_t>((size_t)rows * kstride / 8);
    top = Dev<int>((size_t)rows * kTopk);
    tkey = Dev<uint16_t>((size_t)rows * kTopk);
    pl = Dev<int>((size_t)rows * kPool);
    pkey = Dev<uint16_t>((size_t)rows * kPool);
    counters = Dev<int>(std::max(rows, attn::kMaxDecodeRows));
    scratch = Dev<uint32_t>(attn::index_dense_scratch() / 4);
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  Pool& pages() {
    if (!pool) {
      pool.emplace(k, cap);
      pool->put(2, ik_k.p, 64);
      pool->put(3, ik_sc.p, 4);
    }
    return *pool;
  }
  attn::IndexKeys ik(bool paged) {
    if (paged) return attn::IndexKeys{nullptr, nullptr, 0, nullptr, 0, 0, pages().view()};
    return attn::IndexKeys{ik_k.p, ik_sc.p, cap};
  }
  void run(bool paged) {
    attn::IndexSel t{kTopk, top.p, tkey.p, kTopk}, ps{};
    if (pool_sel) ps = attn::IndexSel{kPool, pl.p, pkey.p, kPool};
    attn::index_dense(iq.p, wp.p, pos.p, tab.p, rows, ratio, ik(paged), max_scan, keys.p, kstride, bkeys.p,
                      kstride / 8, t, ps, scratch.p, false, g_s);
  }
  Snap snap(bool /*paged*/) {
    Snap s;
    s.add("keys", 2, bytes_of(keys)), s.add("block maxima", 2, bytes_of(bkeys));
    s.add("top", 4, bytes_of(top)), s.add("top keys", 2, bytes_of(tkey));
    if (pool_sel) s.add("pool", 4, bytes_of(pl)), s.add("pool keys", 2, bytes_of(pkey));
    return s;
  }
  std::string where(size_t a, size_t e) const {  // the row and the element inside it
    const size_t w[6] = {(size_t)kstride, (size_t)kstride / 8, (size_t)kTopk,
                         (size_t)kTopk,   (size_t)kPool,      (size_t)kPool};
    return "row " + std::to_string(e / w[a]) + " element " + std::to_string(e % w[a]);
  }
  std::string val(size_t a, uint64_t v) const {  // the selections are 32-bit positions
    char t[32];
    if (a == 2 || a == 4)
      std::snprintf(t, sizeof(t), "0x%08x (%d)", (unsigned)v, (int)(uint32_t)v);
    else
      std::snprintf(t, sizeof(t), "0x%04x", (unsigned)v);
    return t;
  }
};

// ---- the compressed attention ----------------------------------------------------------------------

// sparse_attn: the window and the 512 selected entries of `rows` rows, 10 CTAs a row (2 window blocks +
// 512/64), the combine included.
struct AttnCase {
  int rows, ctx, k, ratio, cap, splits;
  Dev<int> pos, topk;
  Dev<float2> tab;
  Dev<uint8_t> ckv_kv, ckv_sc, ring_kv, ring_sc;
  Dev<uint16_t> q, out;
  Dev<float> sink, po, pml;
  std::optional<Pool> pool;
  AttnCase(int rows_, int ctx_, int k_)
      : rows(rows_), ctx(ctx_), k(k_), ratio(k_ < 3 ? 2 : 1), cap(cap_of(ctx_, k_ < 3 ? 2 : 1)),
        splits(kTopk / 64 + 2) {
    const std::vector<int> h = positions(ctx, rows);
    pos = Dev<int>(h.data(), h.size());
    tab = Dev<float2>((size_t)(bucket_of(ctx) + 1) * 32);
    attn::rope_table_build(tab.p, bucket_of(ctx) + 1, true, g_s);
    ckv_kv = Dev<uint8_t>((size_t)cap * 256);
    ckv_sc = Dev<uint8_t>((size_t)cap * 32);
    fill::u8(ckv_kv, 0xFF, 0);
    fill::u8(ckv_sc, 0x1F, 0x28);  // e4m3 scales without NaN (0x7F), as fused_tests
    ring_kv = Dev<uint8_t>((size_t)attn::kRing * kHead);
    ring_sc = Dev<uint8_t>((size_t)attn::kRing * kHead / 32);
    fill::u8(ring_kv, 0x77, 0);  // e4m3 without NaN, as fused_tests
    fill::u8(ring_sc, 7, 120);
    q = Dev<uint16_t>((size_t)rows * attn::kHeads * kHead);
    fill::bf16(q, 1.f);
    out = Dev<uint16_t>((size_t)rows * attn::kHeads * kHead);
    sink = Dev<float>(attn::kHeads);
    fill::f32(sink, 1.f);
    po = Dev<float>((size_t)rows * kSplits * attn::kHeads * kHead);
    pml = Dev<float>((size_t)rows * kSplits * attn::kHeads * 2);
    std::vector<int> hk((size_t)rows * kTopk);  // ascending positions of the row's compressed entries
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < kTopk; ++i) hk[(size_t)r * kTopk + i] = (int)((int64_t)i * ((h[r] + 1) / ratio) / kTopk);
    topk = Dev<int>(hk.data(), hk.size());
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  Pool& pages() {
    if (!pool) {
      pool.emplace(k, cap);
      pool->put(0, ckv_kv.p, 256);
      pool->put(1, ckv_sc.p, 32);
    }
    return *pool;
  }
  void run(bool paged) {
    const attn::Window win{ring_kv.p, ring_sc.p};
    attn::Compressed cc{ckv_kv.p, ckv_sc.p, cap};
    if (paged) cc.pages = pages().view();
    attn::sparse_attn(q.p, pos.p, attn::WinBounds{}, rows, tab.p, win, cc, topk.p, kTopk, sink.p, out.p, splits,
                      attn::AttnScratch{po.p, pml.p, kSplits}, g_s, false);
  }
  Snap snap(bool /*paged*/) {
    Snap s;
    s.add("out", 2, bytes_of(out));  // bf16: one element a row's head's dim
    return s;
  }
  std::string where(size_t /*a*/, size_t e) const {
    const size_t row = (size_t)attn::kHeads * kHead;
    return "row " + std::to_string(e / row) + " head " + std::to_string(e % row / kHead) + " column " +
           std::to_string(e % kHead);
  }
  std::string val(size_t /*a*/, uint64_t v) const {
    char t[32];
    std::snprintf(t, sizeof(t), "0x%04x (%g)", (unsigned)v, (double)bf2f((uint16_t)v));
    return t;
  }
};

// what a mismatch is, in the case's own terms: the array, how many elements differ, the first of them and both
// values
template <class C>
std::string diff_text(C& c, const Snap& s, const Diff& d) {
  return s.name[d.array] + ": " + std::to_string(d.count) + " of " + std::to_string(d.total) +
         " elements differ, first at " + c.where(d.array, d.elem) + ": got " + c.val(d.array, d.got) + " ref " +
         c.val(d.array, d.ref);
}

// one case: the contiguous run (its outputs the reference), a second contiguous run (the same launch twice must
// give the same bytes), the paged run, the two medians and their difference. Each snapshot follows a stream
// sync: the kernels run on a non-blocking stream, so a host copy is not ordered with them and the reference
// could otherwise be read before the kernel wrote it. `snap(paged)` copies the run's outputs off the device.
template <class C, class SnapFn>
void bench_case(C& c, const char* what, SnapFn&& snap) {
  c.run(false);
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  const Snap ref = snap(false);
  c.run(false);
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  const Snap again = snap(false);
  const double tc = time_us([&] { c.run(false); });
  c.run(true);
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  const Snap got = snap(true);
  const double tp = time_us([&] { c.run(true); });
  const Diff det = diff_of(ref, again), pg = diff_of(ref, got);
  const double d = tp - tc;
  std::printf("%-30s %7d %5d %8.2f %8.2f %+8.2f %+8.2f%%  %s\n", what, c.ctx, c.rows, tc, tp, d, 100.0 * d / tc,
              !det.same() ? "NONDETERMINISTIC" : pg.same() ? "same" : "DIFFERENT");
  if (!det.same()) std::printf("    contiguous vs contiguous: %s\n", diff_text(c, ref, det).c_str());
  if (!pg.same()) std::printf("    contiguous vs paged: %s\n", diff_text(c, ref, pg).c_str());
  g_bad += !det.same() || !pg.same();
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* q = s; *q;) {
    v.push_back(std::atoi(q));
    while (*q && *q != ',') ++q;
    if (*q) ++q;
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ctxs, rows{1, 6, 16};
  bool emit = true, index = true, attn = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) g_iters = std::atoi(argv[++i]);
    else if (a == "--iqm") g_iqm = true;
    else if (a == "--rows" && i + 1 < argc) rows = parse_list(argv[++i]);
    else if (a == "--kernels" && i + 1 < argc) {
      const std::string k = argv[++i];
      emit = k.find("emit") != std::string::npos;
      index = k.find("index") != std::string::npos;
      attn = k.find("attn") != std::string::npos;
    } else ctxs.push_back(std::atoi(argv[i]));
  }
  if (ctxs.empty()) ctxs = {1024, 131072};
  CUDA_CHECK(cudaStreamCreateWithFlags(&g_s, cudaStreamNonBlocking));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  g_flush = Dev<uint8_t>((size_t)std::max(prop.l2CacheSize, 1 << 20) * 4);
  g_sink = Dev<uint32_t>(1);
  g_arena = Dev<uint8_t>(kArena);  // the pool: zeroed once, as the engine's is at reserve
  std::printf("%s: %d SMs, L2 %d MB; %s of %d launches, each after an L2 flush (us); caches in a %zu GiB pool\n",
              prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, g_iqm ? "interquartile mean" : "median",
              g_iters, kArena >> 30);
  std::printf("%-30s %7s %5s %8s %8s %8s %9s  %s\n", "case", "context", "rows", "contig", "paged", "diff", "diff",
              "paged outputs");
  for (int ctx : ctxs) {
    for (int k : {0, 3}) {  // a kv source: 0-2 the ratio-2 layers, 3 layer 20
      const char* src = k < 3 ? "2,8,14 (r2)" : "20 (r1)";
      for (int R : rows) {
        if (emit) {
          EmitCase c(R, ctx, k);
          char what[64];
          std::snprintf(what, sizeof(what), "emit %s", src);
          bench_case(c, what, [&](bool paged) { return c.snap(paged); });
        }
        if (index && (R == 1 || R == 6)) {
          IndexCase c(R, ctx, k);
          char what[64];
          std::snprintf(what, sizeof(what), "index_dense %s", src);
          bench_case(c, what, [&](bool paged) { return c.snap(paged); });
        }
        if (attn && (R == 1 || R == 6)) {
          AttnCase c(R, ctx, k);
          char what[64];
          std::snprintf(what, sizeof(what), "sparse_attn %s", src);
          bench_case(c, what, [&](bool paged) { return c.snap(paged); });
        }
      }
    }
  }
  std::printf("%s\n", g_bad ? "DIFFERENT: a run's outputs are not the reference's, or the runs are not deterministic"
                            : "all cases equal");
  return g_bad ? 1 : 0;
}
