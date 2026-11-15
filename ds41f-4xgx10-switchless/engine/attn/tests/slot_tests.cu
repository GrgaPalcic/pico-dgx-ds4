// Session slots (attn.h, docs/batching.md): steps of 2-4 sessions in shuffled slots against each session's
// rows run alone (a one-session step on a copy of its slot's state, no slot table), bit for bit:
//   - kv_write (the window ring and the dequantised rows), compressor_emit (ratio 2 and 1: entries, index keys,
//     the carry), decode_prep (the same plus q and the head weights), kv_write_sets (pos0 + r, or per-row
//     positions with rows left out);
//   - sparse_attn (window and compressed entries per row);
//   - index_dense (ratio 2; ratio 1 with the pool), index_dense_blocks, index_fused (a dense quarter with the
//     pool, a block list), index_merge2: scan keys, block maxima, top-k and pool with their keys.
// Steps: four one-row sessions (B2), sessions of 1-4 rows (B3's anchors and drafts), up to 8 rows; and B4's 9-16
// rows in 2-4 sessions of at most 8 rows (the instances that take rows in groups of up to 8); positions from 3 to
// 70k (a long session beside short ones: index_dense's CTA shares, the selections' gather paths), both parities
// at a session's start (a ratio-2 group completed from the carry right after another session's row).
// Every run is also checked to have written what it should (the ring, the carry, entries and index keys where
// a row completes one; non-zero outputs), so that equality is not vacuous. Prints one line per group; exits
// non-zero on any difference. --no-table runs the steps without their slot tables: a negative control, every
// group must differ but index_merge2's, which takes no table (it fails only where a scan without its table wrote
// nothing); decode_prep is left out past 8 rows there (without a table they are one session: attn.h).
// --pages runs every group again with the caches in the shared pool's pages (page_pool.cuh): one pool whose pages
// hold all four kv sources at the engine's offsets (page_off), filled from the contiguous states, in reverse
// address order, with the step's own slot table the one the kernels read. Every group fills two sources in the
// same pages -- its own and a second ratio-2 one (source 1) -- and runs the kernels for each: every paged output
// must equal the contiguous run's bit for bit (the step and each session alone).
// --pages-badlayout is the same with the views built from layout_of(ratio), the engine's bug before commit
// 5f0d039: a ratio-2 source other than the first lands in the first one's block. A negative control: every group
// must differ.
//   slot_tests [--no-table | --pages | --pages-badlayout]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "attn/attn.h"
#include "fill.cuh"
#include "page_pool.cuh"
#include "testutil.h"

using namespace attn_test;
using attn::kHead;

namespace {

cudaStream_t g_s;
int g_checks = 0, g_fail = 0, g_group_checks = 0, g_group_fail = 0;
std::string g_group;

void begin(const std::string& name) {
  g_group = name;
  g_group_checks = g_checks;
  g_group_fail = g_fail;
}
void end() {
  std::printf("%-66s %3d checks  %s\n", g_group.c_str(), g_checks - g_group_checks,
              g_fail == g_group_fail ? "identical" : "DIFFERENT");
}
// n elements at device a against device b
template <class T>
void same(const std::string& what, const T* a, const T* b, size_t n) {
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  std::vector<T> ha(n), hb(n);
  CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
  size_t bad = 0, first = 0;
  if (std::memcmp(ha.data(), hb.data(), n * sizeof(T)))
    for (size_t i = 0; i < n; ++i)
      if (std::memcmp(&ha[i], &hb[i], sizeof(T))) {
        if (!bad) first = i;
        ++bad;
      }
  ++g_checks;
  if (bad) {
    ++g_fail;
    std::printf("  %s: %s differs in %zu of %zu (first at %zu)\n", g_group.c_str(), what.c_str(), bad, n, first);
  }
}
// the run wrote what it should: device a differs from b (the state before it) somewhere
template <class T>
void changed(const std::string& what, const T* a, const T* b, size_t n) {
  CUDA_CHECK(cudaStreamSynchronize(g_s));
  std::vector<T> ha(n), hb(n);
  CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
  ++g_checks;
  if (!std::memcmp(ha.data(), hb.data(), n * sizeof(T))) {
    ++g_fail;
    std::printf("  %s: %s not written\n", g_group.c_str(), what.c_str());
  }
}
template <class T>
void nonzero(const std::string& what, const T* a, size_t n) {
  Dev<T> z(n);
  changed(what, a, z.p, n);
}

bool g_no_table = false;  // --no-table: the steps without their slot tables, a negative control (must differ)
bool g_pages = false;     // --pages: the same runs with the caches in the pool's pages (docs/shared-kv.md §2.3)
bool g_bad = false;       // --pages-badlayout: the same with layout_of's views, the negative control (must differ)
bool paged() { return g_pages || g_bad; }

constexpr int kSlots = 5, kTopk = 512, kPool = 2048, kParts = 4, kMaxCtx = 72064;

// Whether part `part` of a dense scan split in kParts (indexer.cu's dense_range) holds none of the entries of
// a session whose last row is at `last`: a short session's later parts. Its selections are then all
// position -1 and key 0.
bool part_empty(int last, int ratio, int part) {
  const int n = (last + 1) / ratio, chunk = ((n + kParts - 1) / kParts + 63) & ~63;
  return std::min(n, part * chunk) >= n;
}

// One session of a step: `rows` rows at positions last - rows + 1 .. last, its state in slot `slot`.
struct Sess {
  int rows, last, slot;
};
struct Step {
  std::vector<Sess> ss;
  int rows = 0;
  std::vector<int> first, hpos, hslot;
  Dev<int> pos, slot;
  explicit Step(std::vector<Sess> s) : ss(std::move(s)) {
    for (const Sess& e : ss) {
      first.push_back(rows);
      for (int r = 0; r < e.rows; ++r) hpos.push_back(e.last - e.rows + 1 + r), hslot.push_back(e.slot);
      rows += e.rows;
    }
    if (rows > attn::kMaxDecodeRows) die("slot_tests: %d rows", rows);
    pos = Dev<int>(hpos.data(), hpos.size());
    slot = Dev<int>(hslot.data(), hslot.size());
  }
  const int* table() const { return g_no_table ? nullptr : slot.p; }
  bool emits(size_t i, int ratio) const {  // a row of session i completes a compressed entry
    for (int r = first[i]; r < first[i] + ss[i].rows; ++r)
      if ((hpos[r] + 1) % ratio == 0) return true;
    return false;
  }
  std::string name() const {
    std::string n;
    for (const Sess& e : ss)
      n += (n.empty() ? "" : " ") + std::to_string(e.rows) + "@" + std::to_string(e.last) + "/s" +
           std::to_string(e.slot);
    return n;
  }
  int max_last() const {
    int m = 0;
    for (const Sess& e : ss) m = std::max(m, e.last);
    return m;
  }
};

// kSlots copies of a state buffer of n elements each, random
template <class T>
struct Slots {
  size_t n;
  Dev<T> d;
  Slots(size_t n_) : n(n_), d(n_ * kSlots) {}
  T* at(int s) const { return d.p + (size_t)s * n; }
  size_t stride() const { return n * sizeof(T); }
  // slot s's state, alone
  Dev<T> take(int s) const {
    Dev<T> o(n);
    CUDA_CHECK(cudaMemcpyAsync(o.p, at(s), n * sizeof(T), cudaMemcpyDeviceToDevice, g_s));
    return o;
  }
};
Slots<uint8_t> slots_u8(size_t n, uint32_t mask, uint32_t add) {
  Slots<uint8_t> s(n);
  fill::u8(s.d, mask, add);
  return s;
}
Slots<float> slots_f32(size_t n, float scale) {
  Slots<float> s(n);
  fill::f32(s.d, scale);
  return s;
}

const float2* rope_table() {
  static Dev<float2> t;
  if (!t.p) {
    t = Dev<float2>((size_t)(kMaxCtx + 8) * 32);
    attn::rope_table_build(t.p, kMaxCtx + 8, true, g_s);
  }
  return t.p;
}

// ---- the window and the compressor ------------------------------------------------------------------

struct WinSlots {  // a layer's window rings
  Slots<uint8_t> kv = slots_u8((size_t)attn::kRing * kHead, 0x77, 0), sc = slots_u8((size_t)attn::kRing * 16, 7, 120);
  attn::Window all(const Step& st) const {
    attn::Window w{kv.d.p, sc.d.p};
    w.slot = st.table(), w.kv_stride = kv.stride(), w.sc_stride = sc.stride();
    return w;
  }
};
struct WinAlone {
  Dev<uint8_t> kv, sc, kv0;  // kv0: the ring before
  WinAlone(const WinSlots& w, int s) : kv(w.kv.take(s)), sc(w.sc.take(s)), kv0(w.kv.take(s)) {}
  attn::Window win() const { return attn::Window{kv.p, sc.p}; }
  void check(const WinSlots& w, int s, const std::string& t, bool written = true) const {
    same(t + " ring", w.kv.at(s), kv.p, kv.n);
    same(t + " ring scales", w.sc.at(s), sc.p, sc.n);
    if (written) changed(t + " ring", kv.p, kv0.p, kv.n);
  }
};

struct CompSlots {  // a kv-source layer's compressor state
  int ratio, cap;
  Slots<float> ckv_c, csc_c;
  Slots<uint8_t> kv, sc, k, ksc;
  CompSlots(int ratio_)
      : ratio(ratio_), cap(kMaxCtx / ratio_ + 64), ckv_c(slots_f32(16 * kHead, 1.f)), csc_c(slots_f32(16 * kHead, 1.f)),
        kv(slots_u8((size_t)cap * kHead / 2, 0xFF, 0)), sc(slots_u8((size_t)cap * 32, 0x1F, 0x28)),
        k(slots_u8((size_t)cap * 64, 0xFF, 0)), ksc(slots_u8((size_t)cap * 4, 7, 120)) {}
  attn::CompressorCarry carry(const Step& st) const {
    attn::CompressorCarry c{ckv_c.d.p, csc_c.d.p};
    c.slot = st.table(), c.stride = ckv_c.n;
    return c;
  }
  attn::Compressed ckv(const Step& st) const {
    attn::Compressed c{kv.d.p, sc.d.p, cap};
    c.slot = st.table(), c.kv_stride = kv.stride(), c.sc_stride = sc.stride();
    return c;
  }
  attn::IndexKeys ik(const Step& st) const {
    attn::IndexKeys i{k.d.p, ksc.d.p, cap};
    i.slot = st.table(), i.k_stride = k.stride(), i.sc_stride = ksc.stride();
    return i;
  }
};
struct CompAlone {
  Dev<float> ckv_c, csc_c, ckv_c0;  // ..0: the state before
  Dev<uint8_t> kv, sc, k, ksc, kv0, k0;
  int cap;
  CompAlone(const CompSlots& c, int s)
      : ckv_c(c.ckv_c.take(s)), csc_c(c.csc_c.take(s)), ckv_c0(c.ckv_c.take(s)), kv(c.kv.take(s)), sc(c.sc.take(s)),
        k(c.k.take(s)), ksc(c.ksc.take(s)), kv0(c.kv.take(s)), k0(c.k.take(s)), cap(c.cap) {}
  attn::CompressorCarry carry() const { return {ckv_c.p, csc_c.p}; }
  attn::Compressed ckv() const { return {kv.p, sc.p, cap}; }
  attn::IndexKeys ik() const { return {k.p, ksc.p, cap}; }
  // entries: the session completes one (else the entries and keys stay as they were)
  void check(const CompSlots& c, int s, const std::string& t, bool entries) const {
    same(t + " entries", c.kv.at(s), kv.p, kv.n);
    same(t + " entry scales", c.sc.at(s), sc.p, sc.n);
    same(t + " index keys", c.k.at(s), k.p, k.n);
    same(t + " index key scales", c.ksc.at(s), ksc.p, ksc.n);
    same(t + " carry kv", c.ckv_c.at(s), ckv_c.p, ckv_c.n);
    same(t + " carry score", c.csc_c.at(s), csc_c.p, csc_c.n);
    if (entries) changed(t + " entries", kv.p, kv0.p, kv.n), changed(t + " index keys", k.p, k0.p, k.n);
    if (c.ratio > 1) changed(t + " carry", ckv_c.p, ckv_c0.p, ckv_c.n);
  }
};

// --pages: the caches in the shared pool's pages (docs/shared-kv.md §2.3, page_pool.cuh). One pool holds all
// four kv sources at the engine's offsets; the group's own source is 0 (ratio 2) or 3 (ratio 1) and a second
// ratio-2 source is 1, both filled from their contiguous state. The carries are a slot each, as the contiguous
// path's; `image` gathers a source's slot back into the contiguous layout for the comparison. Nothing is
// allocated without --pages/--pages-badlayout.
struct Pages {
  PagePool pool;
  Dev<float> ckv_c[4], csc_c[4];  // the carries (a slot each), as the contiguous path's
  Dev<uint8_t> img[4][4];         // [source][array]: the pages gathered into the contiguous layout
  explicit Pages(int ctx) : pool(paged() ? ctx / 2 + 64 : 0, ctx + 64, kSlots, g_bad) {}
  void seed(int k, const CompSlots& c) {  // the contiguous state's bytes in source k's pages, and the carries
    if (!paged()) return;
    for (int s = 0; s < kSlots; ++s) {
      pool.put(k, 0, s, c.kv.at(s), g_s), pool.put(k, 1, s, c.sc.at(s), g_s), pool.put(k, 2, s, c.k.at(s), g_s),
          pool.put(k, 3, s, c.ksc.at(s), g_s);
    }
    if (!ckv_c[k].p) {
      ckv_c[k] = Dev<float>((size_t)kSlots * 16 * kHead);
      csc_c[k] = Dev<float>((size_t)kSlots * 16 * kHead);
    }
    CUDA_CHECK(cudaMemcpyAsync(ckv_c[k].p, c.ckv_c.d.p, c.ckv_c.d.n * 4, cudaMemcpyDeviceToDevice, g_s));
    CUDA_CHECK(cudaMemcpyAsync(csc_c[k].p, c.csc_c.d.p, c.csc_c.d.n * 4, cudaMemcpyDeviceToDevice, g_s));
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  void seed(int k, int s, const CompAlone& a) {  // slot s's pristine state again: its pages and its carry
    const uint8_t* arr[4] = {a.kv.p, a.sc.p, a.k.p, a.ksc.p};
    for (int i = 0; i < 4; ++i) pool.put(k, i, s, arr[i], g_s);
    CUDA_CHECK(cudaMemcpyAsync(ckv_c[k].p + (size_t)s * 16 * kHead, a.ckv_c.p, a.ckv_c.n * 4, cudaMemcpyDeviceToDevice,
                               g_s));
    CUDA_CHECK(cudaMemcpyAsync(csc_c[k].p + (size_t)s * 16 * kHead, a.csc_c.p, a.csc_c.n * 4, cudaMemcpyDeviceToDevice,
                               g_s));
    CUDA_CHECK(cudaStreamSynchronize(g_s));
  }
  // Source k's slot s back into the contiguous layout: what the kernels wrote through the pages.
  const uint8_t* image(int k, int a, int s) {
    if (!img[k][a].p) img[k][a] = Dev<uint8_t>((size_t)kSlots * pool.cap[k] * PagePool::kArr[a]);
    pool.get(k, a, s, img[k][a].p + (size_t)s * pool.cap[k] * PagePool::kArr[a], g_s);
    CUDA_CHECK(cudaStreamSynchronize(g_s));
    return img[k][a].p + (size_t)s * pool.cap[k] * PagePool::kArr[a];
  }
  attn::Compressed ckv(int k, int s) const { return pool.ckv(k, s, pool.cap[k]); }
  attn::IndexKeys ik(int k, int s) const { return pool.ik(k, s, pool.cap[k]); }
  attn::Compressed ckv(int k, const Step& st) const { return pool.ckv(k, st.table(), pool.cap[k]); }
  attn::IndexKeys ik(int k, const Step& st) const { return pool.ik(k, st.table(), pool.cap[k]); }
  attn::CompressorCarry carry(int k, const Step& st) const {
    return {ckv_c[k].p, csc_c[k].p, st.table(), (size_t)16 * kHead};
  }
  attn::CompressorCarry carry(int k, int s) const {  // slot s's row alone, and its own carry
    return {ckv_c[k].p + (size_t)s * 16 * kHead, csc_c[k].p + (size_t)s * 16 * kHead};
  }
};

// The compressor's weights for emit (norm, wk, k_norm)
struct CompW {
  Dev<uint16_t> norm{kHead}, wk{(size_t)attn::kIdxDim * kHead}, knorm{attn::kIdxDim};
  CompW() {
    fill::bf16_range(norm, 0.5f, 1.5f);
    fill::bf16(wk, 0.05f);
    fill::bf16_range(knorm, 0.5f, 1.5f);
  }
  attn::CompressorWeights cw(int ratio) const { return {ratio, nullptr, nullptr, norm.p, wk.p, knorm.p}; }
};

void test_writes(const Step& st) {
  begin("kv_write + compressor_emit " + st.name());
  const int R = st.rows;
  Dev<uint16_t> kvp((size_t)R * kHead), kvw(kHead), kvo((size_t)R * kHead);
  Dev<float> proj((size_t)R * kHead), score((size_t)R * kHead);
  fill::bf16(kvp, 2.f);
  fill::bf16_range(kvw, 0.5f, 1.5f);
  fill::f32(proj, 1.f);
  fill::f32(score, 1.f);
  CompW w;
  WinSlots win;
  CompSlots c2(2), c2b(2), c1(1);
  Pages pg(kMaxCtx);
  if (paged()) pg.seed(0, c2), pg.seed(1, c2b), pg.seed(3, c1);  // the pool's sources 0, 1 (ratio 2) and 3
  std::vector<WinAlone> wa;
  std::vector<CompAlone> a2, a2b, a1;
  for (const Sess& e : st.ss)
    wa.emplace_back(win, e.slot), a2.emplace_back(c2, e.slot), a2b.emplace_back(c2b, e.slot),
        a1.emplace_back(c1, e.slot);
  // the step
  attn::kv_write(kvp.p, kvw.p, st.pos.p, rope_table(), win.all(st), kvo.p, R, g_s);
  attn::compressor_emit(proj.p, score.p, st.pos.p, R, w.cw(2), c2.carry(st), rope_table(), c2.ckv(st), c2.ik(st),
                        nullptr, 0, g_s);
  if (paged())  // source 1's reference: the same emit on its own contiguous state
    attn::compressor_emit(proj.p, score.p, st.pos.p, R, w.cw(2), c2b.carry(st), rope_table(), c2b.ckv(st), c2b.ik(st),
                          nullptr, 0, g_s);
  attn::compressor_emit(proj.p, nullptr, st.pos.p, R, w.cw(1), c1.carry(st), rope_table(), c1.ckv(st), c1.ik(st),
                        nullptr, 0, g_s);
  if (paged()) {  // the same step with the caches in the pool's pages: sources 0, 1 and 3
    attn::compressor_emit(proj.p, score.p, st.pos.p, R, w.cw(2), pg.carry(0, st), rope_table(), pg.ckv(0, st),
                          pg.ik(0, st), nullptr, 0, g_s);
    attn::compressor_emit(proj.p, score.p, st.pos.p, R, w.cw(2), pg.carry(1, st), rope_table(), pg.ckv(1, st),
                          pg.ik(1, st), nullptr, 0, g_s);
    attn::compressor_emit(proj.p, nullptr, st.pos.p, R, w.cw(1), pg.carry(3, st), rope_table(), pg.ckv(3, st),
                          pg.ik(3, st), nullptr, 0, g_s);
  }
  // slot s's source k's pages against the contiguous step's state, which is what the step and each session alone
  // must equal
  auto pages_same = [&](const std::string& q, int k, int s, const CompSlots& c) {
    same(q + " entries", c.kv.at(s), pg.image(k, 0, s), c.kv.n);
    same(q + " entry scales", c.sc.at(s), pg.image(k, 1, s), c.sc.n);
    same(q + " index keys", c.k.at(s), pg.image(k, 2, s), c.k.n);
    same(q + " index key scales", c.ksc.at(s), pg.image(k, 3, s), c.ksc.n);
  };
  auto pages_same1 = [&](const std::string& q, int s) {  // the ratio-1 source's entries and index keys
    same(q + " entries (r1)", c1.kv.at(s), pg.image(3, 0, s), c1.kv.n);
    same(q + " index keys (r1)", c1.k.at(s), pg.image(3, 2, s), c1.k.n);
  };
  // each session alone
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
    Dev<uint16_t> o((size_t)n * kHead);
    const size_t off = (size_t)f * kHead;
    attn::kv_write(kvp.p + off, kvw.p, st.pos.p + f, rope_table(), wa[i].win(), o.p, n, g_s);
    attn::compressor_emit(proj.p + off, score.p + off, st.pos.p + f, n, w.cw(2), a2[i].carry(), rope_table(),
                          a2[i].ckv(), a2[i].ik(), nullptr, 0, g_s);
    attn::compressor_emit(proj.p + off, nullptr, st.pos.p + f, n, w.cw(1), a1[i].carry(), rope_table(), a1[i].ckv(),
                          a1[i].ik(), nullptr, 0, g_s);
    const std::string t = "session " + std::to_string(i);
    same(t + " dequantised kv", kvo.p + off, o.p, o.n);
    nonzero(t + " dequantised kv", o.p, o.n);
    wa[i].check(win, s, t);
    a2[i].check(c2, s, t + " ratio 2", st.emits(i, 2));
    a1[i].check(c1, s, t + " ratio 1", true);
    if (paged()) {  // source 1 alone, and the step's own pages for all three sources
      attn::compressor_emit(proj.p + off, score.p + off, st.pos.p + f, n, w.cw(2), a2b[i].carry(), rope_table(),
                            a2b[i].ckv(), a2b[i].ik(), nullptr, 0, g_s);
      a2b[i].check(c2b, s, t + " ratio 2 source 1", st.emits(i, 2));
      pages_same(t + " pages", 0, s, c2);
      pages_same(t + " pages source 1", 1, s, c2b);
      pages_same1(t + " pages", s);
      if (st.emits(i, 2)) changed(t + " pages source 1 entries", pg.image(1, 0, s), a2b[i].kv0.p, c2b.kv.n);
    }
  }
  if (paged()) {  // and each session alone through its own slot's row, from its pristine state
    for (size_t i = 0; i < st.ss.size(); ++i) {
      const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
      const size_t off = (size_t)f * kHead;
      pg.seed(0, s, a2[i]), pg.seed(1, s, a2b[i]), pg.seed(3, s, a1[i]);
      attn::compressor_emit(proj.p + off, score.p + off, st.pos.p + f, n, w.cw(2), pg.carry(0, s), rope_table(),
                            pg.ckv(0, s), pg.ik(0, s), nullptr, 0, g_s);
      attn::compressor_emit(proj.p + off, score.p + off, st.pos.p + f, n, w.cw(2), pg.carry(1, s), rope_table(),
                            pg.ckv(1, s), pg.ik(1, s), nullptr, 0, g_s);
      attn::compressor_emit(proj.p + off, nullptr, st.pos.p + f, n, w.cw(1), pg.carry(3, s), rope_table(),
                            pg.ckv(3, s), pg.ik(3, s), nullptr, 0, g_s);
      pages_same("session " + std::to_string(i) + " alone pages", 0, s, c2);
      pages_same("session " + std::to_string(i) + " alone pages source 1", 1, s, c2b);
      pages_same1("session " + std::to_string(i) + " alone pages", s);
    }
  }
  end();
}

void test_prep(const Step& st, int ratio) {
  begin("decode_prep ratio=" + std::to_string(ratio) + " " + st.name());
  const int R = st.rows;
  Dev<uint16_t> qa((size_t)R * attn::kQLora), qnw(attn::kQLora), kvp((size_t)R * kHead), kvw(kHead);
  Dev<uint16_t> x((size_t)R * attn::kDim), wproj((size_t)attn::kIdxHeads * attn::kDim);
  Dev<float> proj((size_t)R * kHead), score((size_t)R * kHead);
  fill::bf16(qa, 2.f);
  fill::bf16_range(qnw, 0.5f, 1.5f);
  fill::bf16(kvp, 2.f);
  fill::bf16_range(kvw, 0.5f, 1.5f);
  fill::bf16(x, 1.f);
  fill::bf16(wproj, 0.02f);
  fill::f32(proj, 1.f);
  fill::f32(score, 1.f);
  CompW w;
  WinSlots win;
  CompSlots cs(ratio), cs2(2);  // the group's source, and the pool's second ratio-2 source (source 1)
  const int k = ratio == 1 ? 3 : 0;  // the group's source in the pool
  Pages pg(kMaxCtx);
  if (paged()) pg.seed(k, cs), pg.seed(1, cs2);
  std::vector<WinAlone> wa;
  std::vector<CompAlone> ca, ca2;
  for (const Sess& e : st.ss) wa.emplace_back(win, e.slot), ca.emplace_back(cs, e.slot), ca2.emplace_back(cs2, e.slot);
  struct Out {
    Dev<uint16_t> qr;
    Dev<uint8_t> qq, qs;
    Dev<float> wp;
    explicit Out(int n)
        : qr((size_t)n * attn::kQLora), qq((size_t)n * attn::kQLora), qs((size_t)n * attn::kQLora / 32),
          wp((size_t)n * attn::kIdxHeads) {}
  };
  auto prep = [&](int rat, const int* pos, int f, attn::Window wn, attn::CompressorCarry carry,
                  attn::Compressed ckv, attn::IndexKeys ik, Out& o, int n) {
    attn::DecodePrep a;
    a.pos = pos;
    a.table = rope_table();
    a.qa = qa.p + (size_t)f * attn::kQLora;
    a.q_norm = qnw.p;
    a.qq = o.qq.p, a.qs = o.qs.p, a.qr = o.qr.p;
    a.kv_pre = kvp.p + (size_t)f * kHead;
    a.kv_norm = kvw.p;
    a.win = wn;
    a.ckv_proj = proj.p + (size_t)f * kHead;
    a.cscore = score.p + (size_t)f * kHead;
    a.cw = w.cw(rat);
    a.carry = carry, a.ckv = ckv, a.ik = ik;
    a.x = x.p + (size_t)f * attn::kDim;
    a.wproj = wproj.p;
    a.wp = o.wp.p;
    attn::decode_prep(a, n, true, g_s);
  };
  Out all(R);
  prep(ratio, st.pos.p, 0, win.all(st), cs.carry(st), cs.ckv(st), cs.ik(st), all, R);
  if (paged()) {  // the same step with the caches in the pool's pages: the group's source and source 1
    Out pall(R), all2(R), out1(R);
    prep(2, st.pos.p, 0, win.all(st), cs2.carry(st), cs2.ckv(st), cs2.ik(st), all2, R);  // source 1's reference
    prep(ratio, st.pos.p, 0, win.all(st), pg.carry(k, st), pg.ckv(k, st), pg.ik(k, st), pall, R);
    prep(2, st.pos.p, 0, win.all(st), pg.carry(1, st), pg.ckv(1, st), pg.ik(1, st), out1, R);
    same("pages qr", all.qr.p, pall.qr.p, pall.qr.n);
    same("pages q", all.qq.p, pall.qq.p, pall.qq.n);
    same("pages head weights", all.wp.p, pall.wp.p, pall.wp.n);
    same("pages source 1 qr", all2.qr.p, out1.qr.p, out1.qr.n);
    same("pages source 1 q", all2.qq.p, out1.qq.p, out1.qq.n);
    same("pages source 1 head weights", all2.wp.p, out1.wp.p, out1.wp.n);
  }
  // slot s's pages against the contiguous step's state, which is what the step and each session alone must equal
  auto pages_same = [&](const std::string& q, int src, int s, const CompSlots& c) {
    same(q + " entries", c.kv.at(s), pg.image(src, 0, s), c.kv.n);
    same(q + " entry scales", c.sc.at(s), pg.image(src, 1, s), c.sc.n);
    same(q + " index keys", c.k.at(s), pg.image(src, 2, s), c.k.n);
    same(q + " index key scales", c.ksc.at(s), pg.image(src, 3, s), c.ksc.n);
  };
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
    Out o(n);
    prep(ratio, st.pos.p + f, f, wa[i].win(), ca[i].carry(), ca[i].ckv(), ca[i].ik(), o, n);
    const std::string t = "session " + std::to_string(i);
    same(t + " qr", all.qr.p + (size_t)f * attn::kQLora, o.qr.p, o.qr.n);
    same(t + " q", all.qq.p + (size_t)f * attn::kQLora, o.qq.p, o.qq.n);
    same(t + " q scales", all.qs.p + (size_t)f * attn::kQLora / 32, o.qs.p, o.qs.n);
    same(t + " head weights", all.wp.p + (size_t)f * attn::kIdxHeads, o.wp.p, o.wp.n);
    nonzero(t + " q", o.qq.p, o.qq.n);
    nonzero(t + " head weights", o.wp.p, o.wp.n);
    wa[i].check(win, s, t);
    ca[i].check(cs, s, t, st.emits(i, ratio));
    if (paged()) {  // source 1: its own alone run, and the step's own pages for both sources
      Out o2(n);
      prep(2, st.pos.p + f, f, wa[i].win(), ca2[i].carry(), ca2[i].ckv(), ca2[i].ik(), o2, n);
      ca2[i].check(cs2, s, t + " source 1", st.emits(i, 2));
      pages_same(t + " pages", k, s, cs);
      pages_same(t + " pages source 1", 1, s, cs2);
    }
  }
  if (paged()) {  // and each session alone through its own slot's row, from its pristine state
    for (size_t i = 0; i < st.ss.size(); ++i) {
      const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
      Out o(n);
      pg.seed(k, s, ca[i]), pg.seed(1, s, ca2[i]);
      prep(ratio, st.pos.p + f, f, wa[i].win(), pg.carry(k, s), pg.ckv(k, s), pg.ik(k, s), o, n);
      prep(2, st.pos.p + f, f, wa[i].win(), pg.carry(1, s), pg.ckv(1, s), pg.ik(1, s), o, n);
      pages_same("session " + std::to_string(i) + " alone pages", k, s, cs);
      pages_same("session " + std::to_string(i) + " alone pages source 1", 1, s, cs2);
    }
  }
  end();
}

// kv_write_sets: two sets, rows [0, count) at pos0 + r, or (per_row) at the rows' own positions with every third
// row left out (B3: several sessions' accepted rows), each row into its slot's window; against each row written
// alone (count 1 at pos0 = its position) into a copy of its slot's windows, the rows of a slot in turn
void test_sets(const Step& st, bool per_row) {
  begin(std::string("kv_write_sets") + (per_row ? " per-row positions " : " ") + st.name());
  const int R = st.rows;
  Dev<uint16_t> kvp0((size_t)R * kHead), kvp1((size_t)R * kHead), kvw(kHead);
  fill::bf16(kvp0, 2.f);
  fill::bf16(kvp1, 2.f);
  fill::bf16_range(kvw, 0.5f, 1.5f);
  WinSlots w0, w1;
  std::vector<WinAlone> a0, a1;
  for (int r = 0; r < R; ++r) a0.emplace_back(w0, st.hslot[r]), a1.emplace_back(w1, st.hslot[r]);
  const int p0 = 4093;
  std::vector<int> hp(R);
  for (int r = 0; r < R; ++r) hp[r] = !per_row ? p0 + r : r % 3 == 2 ? -1 : st.hpos[r];
  Dev<int> count(&R, 1), pos0(&p0, 1), pos(hp.data(), R), one(std::vector<int>{1}.data(), 1);
  attn::KvSets sets;
  sets.n = 2;
  sets.s[0] = attn::KvSet{kvp0.p, kvw.p, w0.all(st)};
  sets.s[1] = attn::KvSet{kvp1.p, kvw.p, w1.all(st)};
  attn::kv_write_sets(sets, attn::kMaxDecodeRows, count.p, per_row ? nullptr : pos0.p, rope_table(), true, g_s,
                      per_row ? pos.p : nullptr);
  std::vector<int> done(R, 0);
  for (int r = 0; r < R; ++r) {
    if (done[r]) continue;
    bool written = false;
    for (int q = r; q < R; ++q) {
      if (st.hslot[q] != st.hslot[r]) continue;
      done[q] = 1;
      if (hp[q] < 0) continue;
      written = true;
      Dev<int> pq_d(&hp[q], 1);
      attn::KvSets one_set;
      one_set.n = 2;
      one_set.s[0] = attn::KvSet{kvp0.p + (size_t)q * kHead, kvw.p, a0[r].win()};
      one_set.s[1] = attn::KvSet{kvp1.p + (size_t)q * kHead, kvw.p, a1[r].win()};
      attn::kv_write_sets(one_set, attn::kMaxDecodeRows, one.p, pq_d.p, rope_table(), true, g_s);
      CUDA_CHECK(cudaStreamSynchronize(g_s));
    }
    a0[r].check(w0, st.hslot[r], "slot " + std::to_string(st.hslot[r]) + " set 0", written);
    a1[r].check(w1, st.hslot[r], "slot " + std::to_string(st.hslot[r]) + " set 1", written);
  }
  end();
}

// ---- sparse attention --------------------------------------------------------------------------------

void test_attn(const Step& st) {
  begin("sparse_attn " + st.name());
  const int R = st.rows, kmax = kTopk, splits = 2 + kmax / 64;
  WinSlots win;
  CompSlots cs(2);
  Dev<uint16_t> q((size_t)R * attn::kHeads * kHead);
  Dev<float> sink(attn::kHeads);
  fill::bf16(q, 1.f);
  fill::f32(sink, 1.f);
  std::mt19937 rng(R * 131 + st.max_last());
  std::vector<int> ht((size_t)R * kmax, -1);
  for (int r = 0; r < R; ++r) {  // each row's entries: distinct, ascending, within what it sees (some rows fewer)
    const int n = (st.hpos[r] + 1) / 2, k = std::min(kmax - 29 * (r % 3), n);
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    std::sort(all.begin(), all.begin() + std::max(k, 0));
    for (int i = 0; i < k; ++i) ht[(size_t)r * kmax + i] = all[i];
  }
  Dev<int> topk(ht.data(), ht.size());
  const int ms = 16;
  Dev<float> po((size_t)R * ms * attn::kHeads * kHead), pml((size_t)R * ms * attn::kHeads * 2);
  const attn::AttnScratch scratch{po.p, pml.p, ms};
  Dev<uint16_t> out((size_t)R * attn::kHeads * kHead);
  attn::sparse_attn(q.p, st.pos.p, {}, R, rope_table(), win.all(st), cs.ckv(st), topk.p, kmax, sink.p, out.p, splits,
                    scratch, g_s, true);
  if (paged()) {  // the same attention with the entries in the pool's pages: the two ratio-2 sources
    CompSlots cs2(2);
    Pages pg(kMaxCtx);
    pg.seed(0, cs), pg.seed(1, cs2);
    Dev<uint16_t> outp((size_t)R * attn::kHeads * kHead), out2((size_t)R * attn::kHeads * kHead);
    Dev<uint16_t> outp2((size_t)R * attn::kHeads * kHead);
    attn::sparse_attn(q.p, st.pos.p, {}, R, rope_table(), win.all(st), pg.ckv(0, st), topk.p, kmax, sink.p, outp.p,
                      splits, scratch, g_s, true);
    same("pages out", out.p, outp.p, out.n);
    attn::sparse_attn(q.p, st.pos.p, {}, R, rope_table(), win.all(st), cs2.ckv(st), topk.p, kmax, sink.p, out2.p, splits,
                      scratch, g_s, true);  // source 1's reference
    attn::sparse_attn(q.p, st.pos.p, {}, R, rope_table(), win.all(st), pg.ckv(1, st), topk.p, kmax, sink.p, outp2.p,
                      splits, scratch, g_s, true);
    same("pages source 1 out", out2.p, outp2.p, out2.n);
    for (size_t i = 0; i < st.ss.size(); ++i) {  // and each session alone through its own slot's row
      const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
      WinAlone wa(win, s);
      Dev<uint16_t> o((size_t)n * attn::kHeads * kHead), o2((size_t)n * attn::kHeads * kHead);
      const size_t qo = (size_t)f * attn::kHeads * kHead;
      attn::sparse_attn(q.p + qo, st.pos.p + f, {}, n, rope_table(), wa.win(), pg.ckv(0, s), topk.p + (size_t)f * kmax,
                        kmax, sink.p, o.p, splits, scratch, g_s, true);
      same("session " + std::to_string(i) + " alone pages", out.p + qo, o.p, o.n);
      attn::sparse_attn(q.p + qo, st.pos.p + f, {}, n, rope_table(), wa.win(), pg.ckv(1, s), topk.p + (size_t)f * kmax,
                        kmax, sink.p, o2.p, splits, scratch, g_s, true);
      same("session " + std::to_string(i) + " alone pages source 1", out2.p + qo, o2.p, o2.n);
    }
  }
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows, s = st.ss[i].slot;
    WinAlone wa(win, s);
    CompAlone ca(cs, s);
    Dev<uint16_t> o((size_t)n * attn::kHeads * kHead);
    const size_t qo = (size_t)f * attn::kHeads * kHead;
    attn::sparse_attn(q.p + qo, st.pos.p + f, {}, n, rope_table(), wa.win(), ca.ckv(), topk.p + (size_t)f * kmax, kmax,
                      sink.p, o.p, splits, scratch, g_s, true);
    same("session " + std::to_string(i) + " out", out.p + qo, o.p, o.n);
    nonzero("session " + std::to_string(i) + " out", o.p, o.n);
  }
  end();
}

// ---- the indexer -------------------------------------------------------------------------------------

struct IdxOut {  // one run's scan keys and selections, rows `kstride` / kTopk / kPool apart
  int kstride;
  Dev<uint16_t> keys, bkeys, tkey, pkey;
  Dev<int> top, pool;
  IdxOut(int rows, int kstride_)
      : kstride(kstride_), keys((size_t)rows * kstride_), bkeys((size_t)rows * kstride_ / 8),
        tkey((size_t)rows * kTopk), pkey((size_t)rows * kPool), top((size_t)rows * kTopk), pool((size_t)rows * kPool) {}
  attn::IndexSel tsel() const { return {kTopk, top.p, tkey.p, kTopk}; }
  attn::IndexSel psel() const { return {kPool, pool.p, pkey.p, kPool}; }
  // rows [f, f + n) of this against all of o; `empty`: o's scan held no entries (part_empty), so o wrote
  // positions -1 and keys 0
  void check(const IdxOut& o, int f, int n, bool bk, bool pl, const std::string& t, bool empty = false) const {
    same(t + " keys", keys.p + (size_t)f * kstride, o.keys.p, (size_t)n * kstride);
    same(t + " top", top.p + (size_t)f * kTopk, o.top.p, (size_t)n * kTopk);
    same(t + " top keys", tkey.p + (size_t)f * kTopk, o.tkey.p, (size_t)n * kTopk);
    if (empty) nonzero(t + " top (empty)", o.top.p, (size_t)n * kTopk);
    else nonzero(t + " top keys", o.tkey.p, (size_t)n * kTopk);
    if (bk) same(t + " block maxima", bkeys.p + (size_t)f * kstride / 8, o.bkeys.p, (size_t)n * kstride / 8);
    if (pl) {
      same(t + " pool", pool.p + (size_t)f * kPool, o.pool.p, (size_t)n * kPool);
      same(t + " pool keys", pkey.p + (size_t)f * kPool, o.pkey.p, (size_t)n * kPool);
      if (empty) nonzero(t + " pool (empty)", o.pool.p, (size_t)n * kPool);
      else nonzero(t + " pool keys", o.pkey.p, (size_t)n * kPool);
    }
  }
};

struct IdxIn {  // a step's indexer inputs: queries, head weights, each row's pool (a block list)
  Dev<uint16_t> iq;
  Dev<float> wp;
  Dev<int> blocks;
  IdxIn(const Step& st, int ratio)
      : iq((size_t)st.rows * attn::kIdxHeads * attn::kIdxDim), wp((size_t)st.rows * attn::kIdxHeads) {
    fill::bf16(iq, 2.f);
    fill::f32(wp, 1.f);
    std::mt19937 rng(st.rows * 17 + ratio);
    std::vector<int> h((size_t)st.rows * kPool, -1);
    for (int r = 0; r < st.rows; ++r) {
      const int nb = ((st.hpos[r] + 1) / ratio + 7) / 8;
      std::vector<int> all(nb);
      std::iota(all.begin(), all.end(), 0);
      std::shuffle(all.begin(), all.end(), rng);
      const int k = std::min(nb, kPool - 3 * r);
      std::sort(all.begin(), all.begin() + k);
      for (int i = 0; i < k; ++i) h[(size_t)r * kPool + i] = all[i];
    }
    blocks = Dev<int>(h.data(), h.size());
  }
  // the queries and head weights from row f on
  const uint16_t* q(int f) const { return iq.p + (size_t)f * attn::kIdxHeads * attn::kIdxDim; }
  const float* w(int f) const { return wp.p + (size_t)f * attn::kIdxHeads; }
};

// index_dense (pool: layer 20's) and index_dense_blocks; cand: the selections' candidate room
void test_dense(const Step& st, int ratio, bool pool, int cand) {
  begin("index_dense ratio=" + std::to_string(ratio) + (pool ? " +pool" : "") + " cand=" + std::to_string(cand) + " " +
        st.name());
  const int R = st.rows, max_scan = kMaxCtx / ratio;
  const int ks = (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
  CompSlots cs(ratio);
  IdxIn in(st, ratio);
  Dev<uint32_t> scratch(attn::index_dense_scratch() / 4);
  IdxOut all(R, ks);
  for (int rep = 0; rep < 2; ++rep)  // the scratch is left zero
    attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, ratio, cs.ik(st), max_scan, all.keys.p, ks,
                      all.bkeys.p, ks / 8, all.tsel(), pool ? all.psel() : attn::IndexSel{}, scratch.p, true, g_s,
                      cand);
  same("scratch left zero", scratch.p, Dev<uint32_t>(scratch.n).p, scratch.n);
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows;
    CompAlone ca(cs, st.ss[i].slot);
    IdxOut o(n, ks);
    attn::index_dense(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, ratio, ca.ik(), max_scan, o.keys.p, ks,
                      o.bkeys.p, ks / 8, o.tsel(), pool ? o.psel() : attn::IndexSel{}, scratch.p, true, g_s, cand);
    all.check(o, f, n, true, pool, "session " + std::to_string(i));
  }
  if (paged()) {  // the same scans with the keys in the pool's pages: the group's source and source 1
    const int max_scan2 = kMaxCtx / 2;
    const int ks2 = (max_scan2 + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    const int k = ratio == 1 ? 3 : 0;
    CompSlots cs2(2);
    Pages pg(kMaxCtx);
    pg.seed(k, cs), pg.seed(1, cs2);
    IdxOut pall(R, ks);
    for (int rep = 0; rep < 2; ++rep)  // the scratch is left zero
      attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, ratio, pg.ik(k, st), max_scan, pall.keys.p, ks,
                        pall.bkeys.p, ks / 8, pall.tsel(), pool ? pall.psel() : attn::IndexSel{}, scratch.p, true, g_s,
                        cand);
    all.check(pall, 0, R, true, pool, "pages");
    for (size_t i = 0; i < st.ss.size(); ++i) {
      const int f = st.first[i], n = st.ss[i].rows;
      IdxOut o(n, ks);
      attn::index_dense(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, ratio, pg.ik(k, st.ss[i].slot), max_scan,
                        o.keys.p, ks, o.bkeys.p, ks / 8, o.tsel(), pool ? o.psel() : attn::IndexSel{}, scratch.p, true,
                        g_s, cand);
      pall.check(o, f, n, true, pool, "pages session " + std::to_string(i));
    }
    same("pages scratch left zero", scratch.p, Dev<uint32_t>(scratch.n).p, scratch.n);
    // source 1 (ratio 2) in the same pages: a distinct state, the ratio-2 scan
    IdxOut all2(R, ks2), pall2(R, ks2);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, cs2.ik(st), max_scan2, all2.keys.p, ks2,
                        all2.bkeys.p, ks2 / 8, all2.tsel(), attn::IndexSel{}, scratch.p, true, g_s, cand);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, pg.ik(1, st), max_scan2, pall2.keys.p, ks2,
                        pall2.bkeys.p, ks2 / 8, pall2.tsel(), attn::IndexSel{}, scratch.p, true, g_s, cand);
    all2.check(pall2, 0, R, true, false, "pages source 1");
  }
  end();
  if (ratio != 1) return;
  begin("index_dense_blocks cand=" + std::to_string(cand) + " " + st.name());
  IdxOut ball(R, kPool * 8);
  attn::index_dense_blocks(in.q(0), in.w(0), st.pos.p, rope_table(), R, 1, cs.ik(st), in.blocks.p, kPool, ball.keys.p,
                           kPool * 8, ball.tsel(), scratch.p, true, g_s, cand);
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows;
    CompAlone ca(cs, st.ss[i].slot);
    IdxOut o(n, kPool * 8);
    attn::index_dense_blocks(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, 1, ca.ik(),
                             in.blocks.p + (size_t)f * kPool, kPool, o.keys.p, kPool * 8, o.tsel(), scratch.p, true,
                             g_s, cand);
    ball.check(o, f, n, false, false, "session " + std::to_string(i));
  }
  if (paged()) {  // the same block scans with the keys in the pool's pages: the ratio-1 source and source 1
    const int max_scan2 = kMaxCtx / 2;
    const int ks2 = (max_scan2 + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    CompSlots cs2(2);
    Pages pg(kMaxCtx);
    pg.seed(3, cs), pg.seed(1, cs2);
    IdxOut pball(R, kPool * 8);
    attn::index_dense_blocks(in.q(0), in.w(0), st.pos.p, rope_table(), R, 1, pg.ik(3, st), in.blocks.p, kPool, pball.keys.p,
                             kPool * 8, pball.tsel(), scratch.p, true, g_s, cand);
    ball.check(pball, 0, R, false, false, "pages");
    for (size_t i = 0; i < st.ss.size(); ++i) {
      const int f = st.first[i], n = st.ss[i].rows;
      IdxOut o(n, kPool * 8);
      attn::index_dense_blocks(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, 1, pg.ik(3, st.ss[i].slot),
                               in.blocks.p + (size_t)f * kPool, kPool, o.keys.p, kPool * 8, o.tsel(), scratch.p, true,
                               g_s, cand);
      pball.check(o, f, n, false, false, "pages session " + std::to_string(i));
    }
    // source 1 (ratio 2) in the same pages: a distinct state, its dense scan (a block list is ratio 1's)
    IdxOut all2(R, ks2), pall2(R, ks2);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, cs2.ik(st), max_scan2, all2.keys.p, ks2,
                        all2.bkeys.p, ks2 / 8, all2.tsel(), attn::IndexSel{}, scratch.p, true, g_s, cand);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_dense(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, pg.ik(1, st), max_scan2, pall2.keys.p, ks2,
                        pall2.bkeys.p, ks2 / 8, pall2.tsel(), attn::IndexSel{}, scratch.p, true, g_s, cand);
    all2.check(pall2, 0, R, true, false, "pages source 1");
  }
  same("scratch left zero", scratch.p, Dev<uint32_t>(scratch.n).p, scratch.n);
  end();
}

// index_fused: part `part` of a dense scan with the pool, or a block list; then index_merge2 of the rows'
// selections as gathered candidates
void test_fused(const Step& st, int ratio, int part, bool blocks) {
  const std::string scan_name = blocks ? " blocks" : " part=" + std::to_string(part) + " +pool";
  begin("index_fused ratio=" + std::to_string(ratio) + scan_name + " " + st.name());
  const int R = st.rows;
  CompSlots cs(ratio);
  IdxIn in(st, ratio);
  attn::IndexScan scan{};
  int max_scan;
  if (blocks) {
    scan.blocks = in.blocks.p, scan.nblocks = kPool, max_scan = kPool * 8;
  } else {
    scan.part = part, scan.nparts = kParts;
    max_scan = ((kMaxCtx / ratio + kParts - 1) / kParts + 63) / 64 * 64;
  }
  const int ks = (max_scan + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
  Dev<int> counters(attn::kMaxDecodeRows);
  IdxOut all(R, ks);
  const bool pool = !blocks;
  for (int rep = 0; rep < 2; ++rep)  // the counters are left zero
    attn::index_fused(in.q(0), in.w(0), st.pos.p, rope_table(), R, ratio, cs.ik(st), scan, max_scan, all.keys.p, ks,
                      pool ? all.bkeys.p : nullptr, ks / 8, all.tsel(), pool ? all.psel() : attn::IndexSel{},
                      counters.p, true, g_s);
  same("counters left zero", counters.p, Dev<int>(counters.n).p, counters.n);
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows;
    CompAlone ca(cs, st.ss[i].slot);
    attn::IndexScan sc = scan;
    if (blocks) sc.blocks = in.blocks.p + (size_t)f * kPool;
    IdxOut o(n, ks);
    attn::index_fused(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, ratio, ca.ik(), sc, max_scan, o.keys.p, ks,
                      pool ? o.bkeys.p : nullptr, ks / 8, o.tsel(), pool ? o.psel() : attn::IndexSel{}, counters.p,
                      true, g_s);
    all.check(o, f, n, pool, pool, "session " + std::to_string(i), pool && part_empty(st.ss[i].last, ratio, part));
  }
  if (paged()) {  // the same scans with the keys in the pool's pages: the group's source and source 1
    const int k = ratio == 1 ? 3 : 0;
    CompSlots cs2(2);
    Pages pg(kMaxCtx);
    pg.seed(k, cs), pg.seed(1, cs2);
    bool empty_all = pool;
    for (const Sess& e : st.ss) empty_all = empty_all && part_empty(e.last, ratio, part);
    IdxOut pall(R, ks);
    for (int rep = 0; rep < 2; ++rep)  // the counters are left zero
      attn::index_fused(in.q(0), in.w(0), st.pos.p, rope_table(), R, ratio, pg.ik(k, st), scan, max_scan, pall.keys.p, ks,
                        pool ? pall.bkeys.p : nullptr, ks / 8, pall.tsel(), pool ? pall.psel() : attn::IndexSel{},
                        counters.p, true, g_s);
    all.check(pall, 0, R, pool, pool, "pages", empty_all);
    for (size_t i = 0; i < st.ss.size(); ++i) {
      const int f = st.first[i], n = st.ss[i].rows;
      attn::IndexScan sc = scan;
      if (blocks) sc.blocks = in.blocks.p + (size_t)f * kPool;
      IdxOut o(n, ks);
      attn::index_fused(in.q(f), in.w(f), st.pos.p + f, rope_table(), n, ratio, pg.ik(k, st.ss[i].slot), sc, max_scan,
                        o.keys.p, ks, pool ? o.bkeys.p : nullptr, ks / 8, o.tsel(),
                        pool ? o.psel() : attn::IndexSel{}, counters.p, true, g_s);
      pall.check(o, f, n, pool, pool, "pages session " + std::to_string(i),
                 pool && part_empty(st.ss[i].last, ratio, part));
    }
    same("pages counters left zero", counters.p, Dev<int>(counters.n).p, counters.n);
    // source 1 (ratio 2) in the same pages: a distinct state, part 1 of its dense scan
    const int max_scan2 = ((kMaxCtx / 2 + kParts - 1) / kParts + 63) / 64 * 64;
    const int ks2 = (max_scan2 + attn::kScoreAlign - 1) / attn::kScoreAlign * attn::kScoreAlign;
    attn::IndexScan scan2{};
    scan2.part = 1, scan2.nparts = kParts;
    bool empty2 = true;
    for (const Sess& e : st.ss) empty2 = empty2 && part_empty(e.last, 2, 1);
    IdxOut all2(R, ks2), pall2(R, ks2);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_fused(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, cs2.ik(st), scan2, max_scan2, all2.keys.p, ks2,
                        all2.bkeys.p, ks2 / 8, all2.tsel(), all2.psel(), counters.p, true, g_s);
    for (int rep = 0; rep < 2; ++rep)
      attn::index_fused(in.q(0), in.w(0), st.pos.p, rope_table(), R, 2, pg.ik(1, st), scan2, max_scan2, pall2.keys.p, ks2,
                        pall2.bkeys.p, ks2 / 8, pall2.tsel(), pall2.psel(), counters.p, true, g_s);
    all2.check(pall2, 0, R, true, true, "pages source 1", empty2);
  }
  end();
  if (blocks) return;
  // index_merge2 of every row's top and pool as candidates into half as many, the step against each session
  begin("index_merge2 " + st.name());
  Dev<int> mtop((size_t)R * kTopk), mpool((size_t)R * kPool);
  Dev<uint16_t> mtk((size_t)R * kTopk), mpk((size_t)R * kPool);
  const attn::IndexSel mt{kTopk / 2, mtop.p, mtk.p, kTopk}, mp{kPool / 2, mpool.p, mpk.p, kPool};
  attn::index_merge2(R, all.top.p, all.tkey.p, kTopk, mt, all.pool.p, all.pkey.p, kPool, mp, true, g_s);
  for (size_t i = 0; i < st.ss.size(); ++i) {
    const int f = st.first[i], n = st.ss[i].rows;
    Dev<int> ot((size_t)n * kTopk), op((size_t)n * kPool);
    Dev<uint16_t> otk((size_t)n * kTopk), opk((size_t)n * kPool);
    attn::index_merge2(n, all.top.p + (size_t)f * kTopk, all.tkey.p + (size_t)f * kTopk, kTopk,
                       attn::IndexSel{kTopk / 2, ot.p, otk.p, kTopk}, all.pool.p + (size_t)f * kPool,
                       all.pkey.p + (size_t)f * kPool, kPool, attn::IndexSel{kPool / 2, op.p, opk.p, kPool}, true, g_s);
    const std::string t = "session " + std::to_string(i);
    same(t + " top", mtop.p + (size_t)f * kTopk, ot.p, ot.n);
    same(t + " top keys", mtk.p + (size_t)f * kTopk, otk.p, otk.n);
    same(t + " pool", mpool.p + (size_t)f * kPool, op.p, op.n);
    same(t + " pool keys", mpk.p + (size_t)f * kPool, opk.p, opk.n);
    if (part_empty(st.ss[i].last, ratio, part)) nonzero(t + " top (empty)", ot.p, ot.n);  // all -1
    else nonzero(t + " top keys", otk.p, otk.n);
  }
  end();
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--no-table")) g_no_table = true;
    else if (!std::strcmp(argv[i], "--pages")) g_pages = true;
    else if (!std::strcmp(argv[i], "--pages-badlayout")) g_bad = true;
    else die("slot_tests: unknown argument %s", argv[i]);
  }
  if (g_no_table && paged()) die("slot_tests: --no-table and the page modes are exclusive");
  CUDA_CHECK(cudaStreamCreate(&g_s));  // blocking: fill.cuh fills on the legacy stream
  // B2: four one-row sessions, odd and even positions, short and long, in shuffled slots
  const Step b2({{1, 40011, 3}, {1, 517, 0}, {1, 9002, 4}, {1, 37, 1}});
  // B3's shapes: an anchor and drafts per session; a session starting at an odd position right after another
  const Step b3({{3, 12000, 2}, {1, 301, 4}, {4, 2048, 0}});
  const Step two({{2, 6001, 1}, {2, 70001, 3}});
  const Step mixed({{1, 70, 4}, {2, 25001, 0}, {1, 3, 2}, {3, 18432, 1}});
  // B4: two groups of 8 (a long session, a short one starting at an odd position), four sessions of 4 at B2's
  // places (two starting odd), B3's shapes at 12 rows, and a group of 8 beside one row (the group outside slot
  // 0: test_sets leaves the one row out, and without a table the rest would land in their own slot)
  const Step two8({{8, 70001, 1}, {8, 6002, 3}});
  const Step four4({{4, 40011, 3}, {4, 520, 0}, {4, 9002, 4}, {4, 37, 1}});
  const Step b3x12({{6, 12000, 2}, {1, 301, 4}, {5, 2048, 0}});
  const Step nine({{8, 25001, 2}, {1, 3, 0}});
  for (const Step* st : {&b2, &b3, &two, &mixed, &two8, &four4, &b3x12, &nine}) {
    test_writes(*st);
    if (!g_no_table || st->rows <= attn::kDecodeGroup) {
      test_prep(*st, 2);
      test_prep(*st, 1);
    }
    test_sets(*st, false);
    test_sets(*st, true);
    test_attn(*st);
    test_dense(*st, 2, false, -1);
    test_dense(*st, 1, true, -1);
    test_fused(*st, 2, 1, false);
    test_fused(*st, 1, 3, false);
    test_fused(*st, 1, 0, true);
  }
  for (const Step* st : {&b2, &two, &two8}) {  // little or no candidate room: the selections' other paths
    test_dense(*st, 1, true, 0);
    test_dense(*st, 2, false, 1500);
  }
  std::printf("%s%s: %d checks, %d differ\n",
              g_no_table ? "without the tables (must fail): "
              : g_bad   ? "with the bad layout (must fail): "
                        : "",
              g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
