// Sessions' saved state (model.h state_*, state.h). Every copy is queued on the model's stream, so it
// lands in order with the steps around it, and a block freed on the host can be handed out again at
// once: whatever reads it later is queued after whatever wrote it before. Blocks come from the pool
// reserved at startup (state.h), so taking a snapshot inside the decode loop costs no allocation.
#include "model/state.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iterator>
#include <set>

#include "common/keepfile.h"
#include "model/spec.h"
#include "model/spill.h"

namespace model {

using namespace detail;

namespace {

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }

// A block's checksum for the disk tier's round trip: every 8-byte word mixed with its index, summed.
// kSumCtas CTAs each sum a stride of it; the last one to finish adds their partial sums in order and
// stores the result in *out (pinned memory) as one store, so nothing is zeroed first and the sum does
// not depend on which CTA finished when. Launches are serialized on the model's stream, so they share
// the partials and the counter.
constexpr int kSumCtas = 16;
__device__ uint64_t g_sum_part[kSumCtas];
__device__ unsigned g_sum_done;

__global__ void block_sum(const uint64_t* p, size_t n, uint64_t* out) {
  __shared__ uint64_t part[32];
  uint64_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t x = p[i] + i * 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    acc += x ^ (x >> 27);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x >= 32) return;
  acc = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x) return;
  g_sum_part[blockIdx.x] = acc;
  __threadfence();
  if (atomicAdd(&g_sum_done, 1u) != gridDim.x - 1) return;
  __threadfence();  // the last CTA: every partial sum is in
  uint64_t s = 0;
  for (unsigned j = 0; j < gridDim.x; ++j) s += ((volatile uint64_t*)g_sum_part)[j];
  *out = s;
  g_sum_done = 0;
}

// over the bytes the disk tier copies: the block rounded to the pool's grain
void sum_block(const void* p, size_t bytes, uint64_t* out, cudaStream_t st) {
  block_sum<<<kSumCtas, 1024, 0, st>>>((const uint64_t*)p, Pool::round(bytes) / 8, out);
}

}  // namespace

Store::~Store() {
  if (dirfd >= 0) close(dirfd);
}

Store& Model::Impl::store_of() {
  if (!store_s) store_s = std::make_unique<Store>();
  return *store_s;
}

// ---- copies: one index file per session per rank ---------------------------------------------------
//
// A session's last complete copy in the disk tier's file (Sess::Copy) is named by one small index file,
// <base>.d/<h>.<seq>.c: a header, then its blocks' offsets and the checksums they went out with. Every
// finished turn's session is written through to one (state_copy), so a stop, a watchdog restart or a
// power cut loses at most the turn in flight; the next start reads the index files (state_recover) and
// its sessions come back as spilled ones. keepfile.h's rules apply: mode 0600, written whole or not at
// all, in a directory of mode 0700.

namespace {

constexpr char kCopyMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'C', '1'};

struct CopyHeader {
  char magic[8];
  uint64_t run_id;
  int32_t rank, h;
  uint64_t seq;
  int32_t saved, end_pos;
  uint64_t used;
  int64_t written;  // unix time, seconds
  uint64_t file_bytes, page_bytes, snap_bytes, end_bytes;  // the layout it was written with
  uint32_t n_pages, n_snaps;
  uint64_t body_bytes, body_hash;
};

// <base>.d/<h>.<seq>.c
std::string copy_name(const Store& s, int h, uint64_t seq) {
  return s.cdir + "/" + std::to_string(h) + "." + std::to_string(seq) + ".c";
}

// whether `name` is one of ours (the scan ignores anything else in the directory)
bool copy_file(const char* name) {
  const std::string n = name;
  const size_t dot = n.find('.');
  if (dot == std::string::npos || n.size() < 3 || n.compare(n.size() - 2, 2, ".c") != 0) return false;
  const size_t dot2 = n.find('.', dot + 1);
  if (dot2 == std::string::npos || dot2 == dot + 1 || dot2 + 2 != n.size()) return false;
  for (size_t i = 0; i < dot; ++i)
    if (n[i] < '0' || n[i] > '9') return false;
  for (size_t i = dot + 1; i < dot2; ++i)
    if (n[i] < '0' || n[i] > '9') return false;
  return true;
}

// The bytes of a copy's index file: its header, then the blocks' offsets and their checksums. The sums are
// the disk tier's pinned ones as each block went out (the writer has waited for the copy's event, and a
// carried block's sum stays in its slot until its offset is freed, which waits for this index to go).
void copy_index(SpillFile& disk, const Store::Sess::Copy& c, uint64_t run_id, int rank, int h, bool sp,
                std::vector<uint8_t>& f) {
  using keepfile::put;
  f.assign(sizeof(CopyHeader), 0);
  for (const auto& p : c.pages) put<uint64_t>(f, p.first), put<uint64_t>(f, *disk.sum(p.first, false));
  for (const auto& [pos, os] : c.snaps)
    put<int32_t>(f, pos), put<uint32_t>(f, 0), put<uint64_t>(f, os.first), put<uint64_t>(f, *disk.sum(os.first, false));
  put<uint64_t>(f, c.end.first), put<uint64_t>(f, c.end.first != Pool::kNone ? *disk.sum(c.end.first, false) : 0);
  CopyHeader hd{};
  std::memcpy(hd.magic, kCopyMagic, sizeof kCopyMagic);
  hd.run_id = run_id, hd.rank = rank, hd.h = h, hd.seq = c.seq;
  hd.saved = c.saved, hd.end_pos = c.end_pos, hd.used = c.used, hd.written = c.written;
  hd.file_bytes = disk.size(), hd.page_bytes = kPageBytes, hd.snap_bytes = snap_bytes(sp), hd.end_bytes = end_bytes(sp);
  hd.n_pages = (uint32_t)c.pages.size(), hd.n_snaps = (uint32_t)c.snaps.size();
  hd.body_bytes = f.size() - sizeof hd;
  hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
  std::memcpy(f.data(), &hd, sizeof hd);
}

// One index file read and checked: magic, run id, rank, the layout's sizes, the file's size, the body's
// hash, the offsets (on the grain, inside the file) and the age. False with why: the file is dropped.
bool read_copy(const std::string& path, uint64_t run_id, int rank, bool sp, size_t file_bytes, int max_ctx,
               double max_age_s, int* h_out, Store::Sess::Copy* c, std::string* why) {
  std::vector<uint8_t> f;
  if (!keepfile::read_private(path, 1u << 20, &f, why)) return false;
  if (f.size() < sizeof(CopyHeader)) return *why = path + ": too short", false;
  CopyHeader hd;
  std::memcpy(&hd, f.data(), sizeof hd);
  const double age = (double)std::time(nullptr) - (double)hd.written;
  if (std::memcmp(hd.magic, kCopyMagic, sizeof kCopyMagic) != 0) return *why = path + ": not a copy index", false;
  if (hd.run_id != run_id) return *why = path + ": of another engine binary, weights or options", false;
  if (hd.rank != rank) return *why = path + ": of another rank", false;
  if (age > max_age_s || age < -600) return *why = path + ": written " + std::to_string((long long)age) + " s ago", false;
  if (hd.page_bytes != kPageBytes || hd.snap_bytes != snap_bytes(sp) || hd.end_bytes != end_bytes(sp))
    return *why = path + ": of another layout", false;
  if (hd.file_bytes != file_bytes) return *why = path + ": not of its file", false;
  if (hd.body_bytes != f.size() - sizeof hd || keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash)
    return *why = path + ": incomplete or damaged", false;
  const int max_pages = max_ctx / kPagePos + 1;
  if (hd.h <= 0 || !hd.seq || hd.n_pages > (uint32_t)max_pages || hd.n_snaps > 4096u || hd.saved < 0 ||
      hd.saved > (int64_t)hd.n_pages * kPagePos || hd.end_pos < -1 || hd.end_pos > max_ctx)
    return *why = path + ": a bad header", false;
  auto block = [&](size_t off, size_t n) {
    return off % kPoolGrain == 0 && off < file_bytes && Pool::round(n) <= file_bytes - off;
  };
  keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
  c->seq = hd.seq, c->saved = hd.saved, c->end_pos = hd.end_pos, c->used = hd.used, c->written = hd.written;
  for (uint32_t i = 0; i < hd.n_pages && r.ok; ++i) {
    const size_t off = r.get<uint64_t>();
    const uint64_t sum = r.get<uint64_t>();
    if (!block(off, kPageBytes)) return *why = path + ": a block outside the file", false;
    c->pages.emplace_back(off, sum);
  }
  for (uint32_t i = 0; i < hd.n_snaps && r.ok; ++i) {
    const int pos = r.get<int32_t>();
    (void)r.get<uint32_t>();
    const size_t off = r.get<uint64_t>();
    const uint64_t sum = r.get<uint64_t>();
    if (pos < kSnapRows || pos > max_ctx || !block(off, snap_bytes(sp)) ||
        !c->snaps.emplace(pos, std::make_pair(off, sum)).second)
      return *why = path + ": a bad snapshot record", false;
  }
  if (r.ok) {
    const size_t off = r.get<uint64_t>();
    const uint64_t sum = r.get<uint64_t>();
    if (off != Pool::kNone && !block(off, end_bytes(sp))) return *why = path + ": a block outside the file", false;
    if (off != Pool::kNone && hd.end_pos < 0) return *why = path + ": an end block with no position", false;
    c->end = {off, sum};
  }
  if (!r.ok || r.at != r.n) return *why = path + ": a bad body", false;
  *h_out = hd.h;
  return true;
}

}  // namespace

// Blocks freed while an index file may still name them, and the index files to remove: every queued
// write lands, the files go, the directory is synced, and only then are the blocks free to hand out
// again. That order is what makes the reuse of an offset safe.
void Model::Impl::settle() {
  Store& s = store_of();
  if (s.pending.empty() && s.unlink_pending.empty()) return;
  s.disk->drain();
  for (const std::string& p : s.unlink_pending) unlink(p.c_str());
  if (s.dirfd >= 0) fsync(s.dirfd);
  for (const auto& [off, n] : s.pending) s.dpool.give_off(off, n), s.dbytes -= Pool::round(n);
  s.pending.clear();
  s.unlink_pending.clear();
}

namespace {

// whether ss's copy is the session as it stands now
bool copy_current(const Store::Sess& ss) {
  if (!ss.copy.seq || ss.cvalid < ss.saved || ss.copy.saved != ss.saved || ss.copy.end_pos != ss.end_pos) return false;
  for (const auto& sn : ss.snaps)
    if (!ss.copy.snaps.count(sn.first)) return false;
  return true;
}

// A copy's blocks and its index file released: the offsets wait for the index to be gone (settle).
void copy_release(Store& s, int h, const Store::Sess::Copy& c, bool sp) {
  for (const auto& [off, sum] : c.pages) s.pending.emplace_back(off, kPageBytes);
  for (const auto& [pos, os] : c.snaps) s.pending.emplace_back(os.first, snap_bytes(sp));
  if (c.end.first != Pool::kNone) s.pending.emplace_back(c.end.first, end_bytes(sp));
  if (c.seq) s.unlink_pending.push_back(copy_name(s, h, c.seq));
}

}  // namespace

// Session h's state written through to a complete copy in the disk tier's file, named by its index file
// (state.h). A no-op when the session is the copy already, or the copy is current. The blocks the old
// copy held unchanged below `cvalid` are carried; everything else is taken and written, and the index
// goes behind them in the writer's queue, so a crash leaves the old copy whole and the new one absent.
void Model::Impl::state_copy(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || !s.disk) return;
  Store::Sess& ss = it->second;
  if (ss.disk || copy_current(ss)) return;
  const bool sp = spec != nullptr;
  settle();
  const Store::Sess::Copy old = ss.copy;
  Store::Sess::Copy c;
  c.saved = ss.saved, c.end_pos = ss.end_pos, c.used = ss.used, c.written = (int64_t)std::time(nullptr);
  const size_t keep_pages = (size_t)std::max(0, std::min(ss.cvalid, old.saved)) / kPagePos;
  std::vector<std::pair<size_t, size_t>> taken;  // this copy's new blocks, freed when it cannot be finished
  bool room = true;
  auto take = [&](size_t bytes) {
    const size_t off = disk_take(h, bytes);
    if (off != Pool::kNone) taken.emplace_back(off, bytes);
    return off;
  };
  std::vector<size_t> newp;  // the page indices this copy writes
  std::vector<int> news;     // and the snapshot positions
  for (size_t i = 0; room && i < ss.pages.size(); ++i) {
    if (i < keep_pages) {
      c.pages.emplace_back(old.pages[i]);
      continue;
    }
    const size_t off = take(kPageBytes);
    if ((room = off != Pool::kNone)) c.pages.emplace_back(off, 0), newp.push_back(i);
  }
  for (auto i = ss.snaps.begin(); room && i != ss.snaps.end(); ++i) {
    const auto o = old.snaps.find(i->first);
    if (i->first <= ss.cvalid && o != old.snaps.end()) {
      c.snaps.emplace(i->first, o->second);  // unchanged, and the file holds it as it is
      continue;
    }
    const size_t off = take(snap_bytes(sp));
    if ((room = off != Pool::kNone)) c.snaps.emplace(i->first, std::make_pair(off, 0)), news.push_back(i->first);
  }
  if (room && ss.end) {  // the end block is always written again: it is what a turn leaves behind
    const size_t off = take(end_bytes(sp));
    if ((room = off != Pool::kNone)) c.end = {off, 0};
  }
  if (!room) {  // no room for the copy: the old one stays, and the offsets it took go back
    for (const auto& [off, n] : taken) s.dpool.give_off(off, n), s.dbytes -= Pool::round(n);
    static double warned = 0;
    if (rank == 0 && wall_s() - warned > 60) {
      warned = wall_s();
      std::printf("state: no room in the disk tier for session %d's copy\n", h);
    }
    return;
  }
  size_t wrote = 0;
  s.disk->begin(st);
  for (const size_t i : newp) {
    sum_block(ss.pages[i], kPageBytes, s.disk->sum(c.pages[i].first, false), st);
    c.wseq = s.disk->write(ss.pages[i], kPageBytes, c.pages[i].first, st);
    wrote += Pool::round(kPageBytes);
  }
  for (const int pos : news) {
    const size_t off = c.snaps.at(pos).first;
    sum_block(ss.snaps.at(pos), snap_bytes(sp), s.disk->sum(off, false), st);
    c.wseq = s.disk->write(ss.snaps.at(pos), snap_bytes(sp), off, st);
    wrote += Pool::round(snap_bytes(sp));
  }
  if (ss.end) {
    sum_block(ss.end, end_bytes(sp), s.disk->sum(c.end.first, false), st);
    c.wseq = s.disk->write(ss.end, end_bytes(sp), c.end.first, st);
    wrote += Pool::round(end_bytes(sp));
  }
  s.disk->flush(st);
  // What the old copy held and this one does not: those blocks and its index file wait for the index to
  // be gone (settle); the blocks this one carries stay where they are.
  for (size_t i = keep_pages; i < old.pages.size(); ++i) s.pending.emplace_back(old.pages[i].first, kPageBytes);
  for (const auto& [pos, os] : old.snaps)
    if (!c.snaps.count(pos)) s.pending.emplace_back(os.first, snap_bytes(sp));
  if (old.end.first != Pool::kNone) s.pending.emplace_back(old.end.first, end_bytes(sp));
  if (old.seq) s.unlink_pending.push_back(copy_name(s, h, old.seq));
  c.seq = ++s.copy_seq;
  ss.copy = c;
  ss.cvalid = ss.saved;
  const Store::Sess::Copy built = c;
  SpillFile* const disk = s.disk.get();
  s.disk->index(copy_name(s, h, c.seq), c.seq, st,
                [built, disk, id = s.run_id, r = rank, h, sp](std::vector<uint8_t>& f) {
                  copy_index(*disk, built, id, r, h, sp, f);
                });
  ++s.stats.copies;
  s.stats.copy_bytes += wrote;
}

// What session h holds from position q on is about to be rewritten: its copy does not cover it any more.
void Model::Impl::state_inval(int h, int q) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  it->second.cvalid = std::max(0, std::min(it->second.cvalid, q));
}

// Every copy written so far in the file and on the disk: a planned stop's last act (serve.cpp flush).
void Model::Impl::state_sync() {
  Store& s = store_of();
  if (!s.disk) return;
  settle();
  s.disk->sync();
}


// The disk tier opened: base.spill of `bytes` (reuse: the file a crash or a stop left, as it is) and
// base.d, the directory of the sessions' index files. Its copies are installed by state_recover_take.
void Model::Impl::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  Store& s = store_of();
  if (s.disk) die("state: the disk tier is opened once");
  bytes = bytes / kPoolGrain * kPoolGrain;
  s.base = base, s.run_id = run_id;
  if (!bytes) return;
  const double t0 = wall_s();
  s.disk = std::make_unique<SpillFile>(base, bytes, staging, reuse);
  s.dpool.size = s.disk->size();
  s.dpool.free.emplace(0, s.dpool.size);
  s.disk->keep();  // every finished turn's copy lives in it: it outlives this run, a stop or a crash
  if (rank == 0)
    std::printf("state: the disk tier's file %s, %.2f GiB, in %.1f ms\n", s.disk->path().c_str(),
                s.dpool.size / 1073741824.0, 1e3 * (wall_s() - t0));
}

// ---- recovery at start -----------------------------------------------------------------------------
//
// A start's index files, one per session per rank, name a complete copy of each session in the spill
// file. They are read here (main.cpp does the allgathers that make the ranks agree on which to keep)
// and the sessions they name come back as spilled ones. Anything that does not check out is removed, so
// the next start drops it; a bad copy costs one start, never a restart loop.

// The spill file's size when base.d holds at least one index file of this run, else 0. Makes the
// directory (mode 0700) and opens it for the later calls.
size_t Model::Impl::state_recover_begin(const std::string& base, uint64_t run_id) {
  Store& s = store_of();
  s.base = base, s.run_id = run_id, s.cdir = base + ".d";
  if (mkdir(s.cdir.c_str(), 0700) != 0 && errno != EEXIST)
    die("state: mkdir %s: %s", s.cdir.c_str(), std::strerror(errno));
  s.dirfd = open(s.cdir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (s.dirfd < 0) die("state: open %s: %s", s.cdir.c_str(), std::strerror(errno));
  struct stat st{};
  if (stat((base + ".spill").c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
      (st.st_mode & 0777) != 0600)
    return 0;
  size_t bytes = 0;
  DIR* d = opendir(s.cdir.c_str());
  for (struct dirent* e; d && (e = readdir(d));) {
    if (!copy_file(e->d_name)) continue;
    std::vector<uint8_t> f;
    std::string why;
    if (!keepfile::read_private(s.cdir + "/" + e->d_name, 1u << 20, &f, &why) || f.size() < sizeof(CopyHeader)) continue;
    CopyHeader hd;
    std::memcpy(&hd, f.data(), sizeof hd);
    if (std::memcmp(hd.magic, kCopyMagic, sizeof kCopyMagic) != 0 || hd.run_id != run_id || hd.rank != rank) continue;
    if (hd.file_bytes != (uint64_t)st.st_size) continue;
    bytes = (size_t)st.st_size;
    break;
  }
  if (d) closedir(d);
  return bytes;
}

// A start that keeps nothing: the spill file, every file in the index directory, and the legacy index.
void Model::Impl::state_recover_clean() {
  Store& s = store_of();
  unlink((s.base + ".spill").c_str());
  unlink((s.base + ".index").c_str());
  unlink((s.base + ".index.new").c_str());
  if (s.cdir.empty()) return;
  if (DIR* d = opendir(s.cdir.c_str())) {
    for (struct dirent* e; (e = readdir(d));)
      if (e->d_name[0] != '.') unlink((s.cdir + "/" + e->d_name).c_str());
    closedir(d);
  }
  if (s.dirfd >= 0) fsync(s.dirfd);
}

// Every index file under base.d read and checked: the whole ones of this run and this layout, young
// enough. *.tmp goes, and so does every file that fails. Returns the (h, seq) pairs that are left.
std::vector<std::pair<int, uint64_t>> Model::Impl::state_recover_scan(double max_age_s) {
  Store& s = store_of();
  const bool sp = spec != nullptr;
  std::vector<std::pair<int, uint64_t>> pairs;
  struct stat st{};
  const bool have = stat((s.base + ".spill").c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_uid == getuid() &&
                    (st.st_mode & 0777) == 0600;
  DIR* d = opendir(s.cdir.c_str());
  if (!d) return pairs;
  int dropped = 0;
  for (struct dirent* e; (e = readdir(d));) {
    const std::string name = e->d_name;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
      unlink((s.cdir + "/" + name).c_str());
      continue;
    }
    if (!copy_file(e->d_name)) continue;
    const std::string path = s.cdir + "/" + name;
    int h = 0;
    Store::Sess::Copy c;
    std::string why;
    if (!have || !read_copy(path, s.run_id, rank, sp, (size_t)st.st_size, opt.max_ctx, max_age_s, &h, &c, &why)) {
      unlink(path.c_str());
      ++dropped;
      if (rank == 0 && !why.empty()) std::printf("state: %s dropped (%s)\n", path.c_str(), why.c_str());
      continue;
    }
    pairs.emplace_back(h, c.seq);
  }
  closedir(d);
  if (dropped) fsync(s.dirfd);
  if (pairs.size() > 262144) {  // a directory this size is not ours: a clean start
    if (rank == 0) std::printf("state: %zu index files: too many to recover, a clean start\n", pairs.size());
    pairs.clear();
  }
  return pairs;
}

// The copies every rank agreed on: installed as spilled sessions (their blocks taken in the file), and
// every index file that is not one of them removed.
void Model::Impl::state_recover_take(const std::vector<std::pair<int, uint64_t>>& keep) {
  Store& s = store_of();
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  std::map<int, uint64_t> want(keep.begin(), keep.end());
  struct stat st{};
  if (stat((s.base + ".spill").c_str(), &st) != 0) die("state: %s is gone", (s.base + ".spill").c_str());
  DIR* d = opendir(s.cdir.c_str());
  int dropped = 0;
  for (struct dirent* e; d && (e = readdir(d));) {
    if (!copy_file(e->d_name)) continue;
    const std::string path = s.cdir + "/" + e->d_name;
    int h = 0;
    Store::Sess::Copy c;
    std::string why;
    if (!read_copy(path, s.run_id, rank, sp, (size_t)st.st_size, opt.max_ctx, 1e18, &h, &c, &why) ||
        !want.count(h) || want.at(h) != c.seq) {
      unlink(path.c_str());
      ++dropped;
      continue;
    }
    want.erase(h);
    auto take = [&](size_t off, size_t n, uint64_t sum) {
      if (!s.dpool.take_at(off, n)) die("state: a recovered block at %zu is not free", off);
      s.dbytes += Pool::round(n);
      *s.disk->sum(off, false) = sum;
    };
    for (const auto& [off, sum] : c.pages) take(off, kPageBytes, sum);
    for (const auto& [pos, os] : c.snaps) take(os.first, snap_bytes(sp), os.second);
    if (c.end.first != Pool::kNone) take(c.end.first, end_bytes(sp), c.end.second);
    Store::Sess& ss = s.sess[h];
    ss.disk = true;
    ss.saved = c.saved, ss.end_pos = c.end_pos, ss.used = c.used;
    c.recovered = true;
    ss.copy = c;
    s.clock = std::max(s.clock, c.used);
    s.copy_seq = std::max(s.copy_seq, c.seq);
  }
  if (d) closedir(d);
  if (dropped) fsync(s.dirfd);
  ++s.clock;
  ++s.copy_seq;
  s.stats.kept_loaded += s.sess.size();
  s.stats.recovered += s.sess.size();
  if (rank == 0 && !s.sess.empty())
    std::printf("state: %zu sessions recovered from %s (%.2f of %.2f GiB) in %.1f ms; %zu index files dropped\n",
                s.sess.size(), s.cdir.c_str(), s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0,
                1e3 * (wall_s() - t0), (size_t)dropped);
}


void Model::Impl::state_reserve(size_t bytes) {
  Store& s = store_of();
  if (s.kv.pool.base) die("state: the store's pool is reserved once");
  bytes = bytes / kPoolGrain * kPoolGrain;
  if (!bytes) return;
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  s.kv.pool.base = static_cast<uint8_t*>(p);
  s.kv.pool.size = bytes;
  s.kv.pool.free.emplace(0, bytes);
  if (shared_kv) reserve_pages(bytes);
}

// --shared-kv: the pool's zero page and the slot rows' images (docs/shared-kv.md §2.3; the slot table
// itself is the model's, made with it). The pool is zeroed once, so a page a session is given holds what
// a fresh contiguous cache held.
void Model::Impl::reserve_pages(size_t bytes) {
  Store& s = store_of();
  CUDA_CHECK(cudaMemset(s.kv.pool.base, 0, bytes));
  s.kv.zero = s.kv.pool.take(kPageBytes);
  if (!s.kv.zero) die("state: a pool of %.2f GiB cannot hold the zero page", bytes / 1073741824.0);
  s.kv.reset(nslots, max_pages, s.kv.zero);
  for (int sl = 0; sl < nslots; ++sl) write_slot_row(sl);  // every row the zero page until a session takes it
  use_slot(cur_slot);  // the eager paths' page views
}

// Session h takes slot s (§2.6): the slot's row is the session's page list, written now (h = 0: the
// slot has no session and the row is all zero pages).
void Model::Impl::state_slot_row(int s, int h) {
  if (!shared_kv) return;
  if (s < 0 || s >= nslots) die("state: slot %d of %d", s, nslots);
  // The golden and bench modes' slots hold handles -(s + 1) for the whole run (main.cpp): a command acting
  // on such a slot with no server session (h = 0, apply_ops) keeps it. The server's handles are >= 1.
  if (h == 0 && slot_sess.of(s) < 0) return;
  slot_sess.set(s, h);
  if (store_of().kv.zero) write_slot_row(s);  // before the pool exists, reserve_pages writes every row
}

// Slot s's table row: its session's pages, then the zero page (§2.3). The whole row, one copy; the
// entries above the session's list never hold a stale page a reader could dereference.
void Model::Impl::write_slot_row(int s) {
  Store& s0 = store_of();
  const auto it = s0.sess.find(slot_sess.of(s));
  const std::vector<uint8_t*>* pages = it == s0.sess.end() ? nullptr : &it->second.pages;
  s0.kv.list(s, pages ? (int)pages->size() : 0, pages ? pages->data() : nullptr);
  CUDA_CHECK(cudaMemcpyAsync(slot_tab + (size_t)s * max_pages, s0.kv.row(s), s0.kv.row_bytes(), cudaMemcpyHostToDevice, st));
}

// The slot session h is live in (-1: none).
int Model::Impl::slot_of(int h) const { return slot_sess.slot_of(h); }

// The least recently used session that session h's operation may evict: not h, not spilled, not pinned,
// and under the flag not live in a slot (a live session's pages are its working state). The flag's own
// page allocations ask it; store_alloc's loop has the same rule without the flag's clause (T4 adds it
// there, where the live-session refusal in state_spill belongs).
int Model::Impl::evictable(int h) {
  const Store& s = store_of();
  int victim = -1;
  uint64_t oldest = UINT64_MAX;
  for (const auto& [k, ss] : s.sess)
    if (k != h && !ss.disk && ss.used < oldest && std::find(s.pinned.begin(), s.pinned.end(), k) == s.pinned.end() &&
        (!shared_kv || slot_of(k) < 0))
      victim = k, oldest = ss.used;
  return victim;
}

// A block for session h. While none is free, the least recently used session that the current command
// does not use is evicted (every rank evicts the same): spilled to the disk tier, or freed without one;
// nullptr when none is left to evict.
uint8_t* Model::Impl::store_alloc(int h, size_t bytes) {
  Store& s = store_of();
  for (;;) {
    if (uint8_t* p = s.kv.pool.take(bytes)) {
      s.bytes += Pool::round(bytes);
      return p;
    }
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess)
      if (k != h && !ss.disk && ss.used < oldest && std::find(s.pinned.begin(), s.pinned.end(), k) == s.pinned.end() &&
          (!shared_kv || slot_of(k) < 0))  // §2.5: a live session's pages are its working state
        victim = k, oldest = ss.used;
    if (victim < 0) {
      if (rank == 0)
        std::printf("state: no room for %.1f MiB of session %d's state (%.2f of %.2f GiB in use)\n",
                    bytes / 1048576.0, h, s.bytes / 1073741824.0, s.kv.pool.size / 1073741824.0);
      return nullptr;
    }
    state_spill(victim);
  }
}

// --shared-kv: a page for session h's copy-on-write (§2.7), inside a state operation that cannot fail: the
// pool's (evicting), else the spare page held back for this (SharedKv::spare). Null only when both are gone,
// which session_room prevents: a session grows only once the spare is back.
uint8_t* Model::Impl::cow_page(int h) {
  Store& s = store_of();
  if (uint8_t* p = store_alloc(h, kPageBytes)) {
    ++s.stats.cow;
    return p;
  }
  uint8_t* p = s.kv.spare;
  s.kv.spare = nullptr;
  if (p) s.bytes += Pool::round(kPageBytes), ++s.stats.cow;
  return p;
}

// ---- the disk tier --------------------------------------------------------------------------------

// Room in the file for a block of session h: while there is none, the least recently used session with
// a copy that the current command does not use gives its copy up (a spilled one goes entirely: state_free
// and state_evicted; a stored one keeps its blocks in the pool). kNone when none is left.
size_t Model::Impl::disk_take(int h, size_t bytes) {
  Store& s = store_of();
  const bool sp = spec != nullptr;
  settle();
  for (;;) {
    const size_t off = s.dpool.take_off(bytes);
    if (off != Pool::kNone) {
      s.dbytes += Pool::round(bytes);
      return off;
    }
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess)
      if (k != h && ss.copy.seq && ss.used < oldest && std::find(s.pinned.begin(), s.pinned.end(), k) == s.pinned.end())
        victim = k, oldest = ss.used;
    if (victim < 0) return Pool::kNone;
    if (s.sess.at(victim).disk) {
      if (rank == 0) std::printf("state: session %d dropped from the disk tier to make room\n", victim);
      ++s.stats.disk_drops;
      state_free(victim);
      s.evicted.push_back(victim);
    } else {  // its copy goes; its blocks stay in the pool
      Store::Sess& v = s.sess.at(victim);
      copy_release(s, victim, v.copy, sp);
      v.copy = Store::Sess::Copy{};
      v.cvalid = 0;
    }
    settle();
  }
}

// Session h's blocks into the disk tier: its copy first (state_copy), then the pool's blocks released,
// as today. Without a copy to come back from (no tier, or no room for one) h is freed (state_evicted).
void Model::Impl::state_spill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  // --shared-kv (§2.5): a live session's pages are its working state, so it is never spilled; the server
  // lets its slot go first (persist_slot(s, true)), which makes it a stored session like any other.
  if (shared_kv && slot_of(h) >= 0)
    die("state: session %d is live in slot %d", h, slot_of(h));
  Store::Sess& ss = it->second;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  state_copy(h);
  if (!copy_current(ss)) {  // no copy of what it holds now: it is not worth keeping
    state_free(h);
    s.evicted.push_back(h);
    return;
  }
  const size_t b0 = s.bytes;
  const size_t spilled = ss.pages.size() * Pool::round(kPageBytes) + ss.snaps.size() * Pool::round(snap_bytes(sp)) +
                         (ss.end ? Pool::round(end_bytes(sp)) : 0);
  for (uint8_t* p : ss.pages)
    if (!shared_kv || s.kv.release(p)) store_release(p, kPageBytes);  // §2.7: shared
  for (const auto& sn : ss.snaps) store_release(sn.second, snap_bytes(sp));
  store_release(ss.end, end_bytes(sp));
  ss.pages.clear();
  ss.snaps.clear();
  ss.end = nullptr;
  ss.disk = true;
  ++s.stats.spills;
  s.stats.spill_bytes += spilled;
  if (rank == 0)
    std::printf("state: session %d spilled to disk (%.1f MiB; %.2f of %.2f GiB of the file in use; host %.2f ms)\n", h,
                (b0 - s.bytes) / 1048576.0, s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0,
                1e3 * (wall_s() - t0));
}

// Session h back from its copy (after its writes), as far as the store has room: pages in order (saved
// shrinks to what came back), the end block, then window snapshots from the latest down; what does not
// fit is dropped, as a save that finds no room would. Every block that comes back is checked against the
// checksum it went out with; a mismatch removes the copy's index file and stops the engine, so that no
// turn continues from a wrong state and the next start drops the session instead of looping.
void Model::Impl::state_unspill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || !it->second.disk) return;
  Store::Sess& ss = it->second;
  const Store::Sess::Copy c = ss.copy;  // the pool holds none of its blocks: the copy is the session
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  s.disk->wait(c.wseq);
  const double t1 = wall_s();
  size_t got = 0;
  std::vector<size_t> back;  // the file offsets of the blocks that came back
  const auto read = [&](uint8_t* dst, size_t bytes, size_t off) {
    s.disk->read(dst, bytes, off, st);
    sum_block(dst, bytes, s.disk->sum(off, true), st);
    back.push_back(off);
    got += Pool::round(bytes);
  };
  s.pinned.push_back(h);  // its own blocks' allocations never evict it
  for (size_t i = 0; i < c.pages.size(); ++i) {
    uint8_t* p = store_alloc(h, kPageBytes);
    if (!p) break;
    read(p, kPageBytes, c.pages[i].first);
    ss.pages.push_back(p);
  }
  ss.saved = std::min(c.saved, (int)ss.pages.size() * kPagePos);
  if (c.end.first != Pool::kNone && c.end_pos <= ss.saved)
    if ((ss.end = store_alloc(h, end_bytes(sp)))) read(ss.end, end_bytes(sp), c.end.first);
  if (!ss.end) ss.end_pos = -1;
  for (auto i = c.snaps.rbegin(); i != c.snaps.rend(); ++i) {
    uint8_t* b = store_alloc(h, snap_bytes(sp));
    if (!b) break;
    read(b, snap_bytes(sp), i->second.first);
    ss.snaps.emplace(i->first, b);
  }
  s.pinned.pop_back();
  CUDA_CHECK(cudaStreamSynchronize(st));
  const double t2 = wall_s();
  size_t bad = 0;
  uint64_t fp = 0;  // the checksums' xor, logged: a check that compared nothing would show as 0
  for (const size_t o : back) bad += *s.disk->sum(o, false) != *s.disk->sum(o, true), fp ^= *s.disk->sum(o, true);
  if (bad) {  // the copy is wrong: its index goes, and so does the process (a bad copy costs one crash)
    const std::string path = copy_name(s, h, c.seq);
    unlink(path.c_str());
    if (s.dirfd >= 0) fsync(s.dirfd);
    die("state: session %d's copy %llu came back different from the disk (rank %d): its index is removed, so the "
        "next start drops it", h, (unsigned long long)c.seq, rank);
  }
  ss.disk = false;
  ss.cvalid = std::min(c.saved, ss.saved);
  ++s.stats.unspills;
  s.stats.unspill_bytes += got;
  if (rank == 0)
    std::printf("state: session %d back from disk: %.1f MiB in %.1f ms (%.2f GB/s; %.1f ms waiting for its writes; "
                "%zu blocks checked, %016llx)\n", h, got / 1048576.0, 1e3 * (t2 - t0), got / std::max(1e-9, t2 - t1) / 1e9,
                1e3 * (t1 - t0), back.size(), (unsigned long long)fp);
}

namespace {

// session h (made if new) for an operation that needs its blocks: back from the disk tier first
Store::Sess& use_blocks(Model::Impl& m, int h) {
  Store& s = m.store_of();
  const auto it = s.sess.find(h);
  if (it != s.sess.end() && it->second.disk) m.state_unspill(h);
  return s.use(h);
}

}  // namespace

void Model::Impl::store_release(uint8_t* p, size_t bytes) {
  if (!p) return;
  Store& s = store_of();
  s.kv.pool.give(p, bytes);
  s.bytes -= Pool::round(bytes);
}

// Compressed caches of positions [from, to): entries [from / ratio, to / ratio) of every kv source
// (an entry is complete once its last position is in). Without room for every page, as far as the
// pages go.
void Model::Impl::state_save(int h, int from, int to) {
  Store::Sess& ss = use_blocks(*this, h);
  if (from > ss.saved) die("state: session %d saves [%d, %d) with %d saved", h, from, to, ss.saved);
  if (shared_kv) {  // the entries are already in the session's pages: saved only grows
    if (to > (int)ss.pages.size() * kPagePos)
      die("state: session %d saves to %d without its page %d", h, to, (int)ss.pages.size());
    ss.saved = std::max(ss.saved, to);
    return;
  }
  while ((int)ss.pages.size() * kPagePos < to) {
    uint8_t* p = store_alloc(h, kPageBytes);
    if (!p) break;
    ss.pages.push_back(p);
  }
  to = std::min(to, (int)ss.pages.size() * kPagePos);
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio;
    const KvSrc& src = kvs[k];
    uint8_t* const arr[4] = {src.ckv, src.csc, src.ik, src.iksc};
    for (int j = from / ratio, j1 = to / ratio; j < j1;) {
      const int pi = j / E, n = std::min(j1, (pi + 1) * E) - j;
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemcpyAsync(ss.pages[pi] + page_off(k, a) + (size_t)(j - pi * E) * kArrBytes[a], arr[a] + (size_t)j * kArrBytes[a],
                                   (size_t)n * kArrBytes[a], cudaMemcpyDeviceToDevice, st));
      j += n;
    }
  }
  ss.saved = std::max(ss.saved, to);
}

void Model::Impl::state_load(int h, int to) {
  if (!store_of().sess.count(h)) die("state: session %d loads with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (to > ss.saved) die("state: session %d loads %d positions of %d", h, to, ss.saved);
  if (shared_kv) {  // no copy: the entries are the session's pages, so the slot's row points at them
    const int s = slot_of(h);
    if (s >= 0) write_slot_row(s);
    return;
  }
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio;
    const KvSrc& dst = kvs[k];
    uint8_t* const arr[4] = {dst.ckv, dst.csc, dst.ik, dst.iksc};
    for (int j = 0, j1 = to / ratio; j < j1;) {
      const int pi = j / E, n = std::min(j1, (pi + 1) * E) - j;
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemcpyAsync(arr[a] + (size_t)j * kArrBytes[a], ss.pages[pi] + page_off(k, a) + (size_t)(j - pi * E) * kArrBytes[a],
                                   (size_t)n * kArrBytes[a], cudaMemcpyDeviceToDevice, st));
      j += n;
    }
  }
}

namespace {

// positions [p0, p1) between a ring and a linear buffer (row bytes per position), at most 256 of them
void ring_rows(uint8_t* ring, uint8_t* lin, size_t row, int p0, int p1, bool to_lin, cudaStream_t st) {
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    uint8_t* r = ring + (size_t)slot * row;
    uint8_t* l = lin + (size_t)(q - p0) * row;
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? l : r, to_lin ? r : l, (size_t)n * row, cudaMemcpyDeviceToDevice, st));
    q += n;
  }
}

// the draft module's windows, positions [p0, p1) or whole (p0 < 0), to or from b
void spec_windows(const Spec& sp, uint8_t* b, int p0, int p1, bool to_buf, cudaStream_t st) {
  const size_t n = p0 < 0 ? attn::kRing : (size_t)(p1 - p0);
  for (int k = 0; k < dspark::kBlocks; ++k) {
    const attn::Window w = sp.draft->window(k);
    uint8_t* kv = b + (size_t)k * n * kSpecSlot;
    uint8_t* sc = kv + n * kHead;
    if (p0 < 0) {
      CUDA_CHECK(cudaMemcpyAsync(to_buf ? kv : w.kv, to_buf ? w.kv : kv, n * kHead, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(to_buf ? sc : w.sc, to_buf ? w.sc : sc, n * kHead / 32, cudaMemcpyDeviceToDevice, st));
    } else {
      ring_rows(w.kv, kv, kHead, p0, p1, to_buf, st);
      ring_rows(w.sc, sc, kHead / 32, p0, p1, to_buf, st);
    }
  }
}

}  // namespace

// A snapshot's block (the caller fills it), or nullptr when the store has no room for it.
uint8_t* Model::Impl::state_snap_buf(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  const auto it = ss.snaps.find(pos);
  if (it != ss.snaps.end()) return it->second;
  uint8_t* b = store_alloc(h, snap_bytes(spec != nullptr));
  if (b) ss.snaps.emplace(pos, b);
  return b;
}

void Model::Impl::state_snap(int h, int pos) {
  if (pos < kSnapRows) die("state: snapshot at %d", pos);
  uint8_t* b = state_snap_buf(h, pos);
  if (!b) return;
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(ring_kv[L], snap_kv(b, L), kHead, pos - kSnapRows, pos, true, st);
    ring_rows(ring_sc[L], snap_sc(b, L), kHead / 32, pos - kSnapRows, pos, true, st);
  }
  if (spec) spec_windows(*spec, b + kSnapBytes, pos - kSnapRows, pos, true, st);
}

// The draft module's windows into the snapshot at pos, which the prefill took (none: not taken).
void Model::Impl::state_snap_spec(int h, int pos) {
  const Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  const auto b = it->second.snaps.find(pos);
  if (b != it->second.snaps.end()) spec_windows(*spec, b->second + kSnapBytes, pos - kSnapRows, pos, true, st);
}

void Model::Impl::state_restore(int h, int pos) {
  if (!store_of().sess.count(h)) die("state: session %d restores with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.snaps.count(pos)) die("state: session %d has no snapshot at %d", h, pos);
  uint8_t* b = ss.snaps.at(pos);
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(ring_kv[L], snap_kv(b, L), kHead, pos - kSnapRows, pos, false, st);
    ring_rows(ring_sc[L], snap_sc(b, L), kHead / 32, pos - kSnapRows, pos, false, st);
  }
  if (spec) spec_windows(*spec, b + kSnapBytes, pos - kSnapRows, pos, false, st);
}

void Model::Impl::state_snap_end(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.end) ss.end = store_alloc(h, end_bytes(spec != nullptr));
  if (!ss.end) {
    ss.end_pos = -1;
    return;
  }
  ss.end_pos = pos;
  uint8_t* b = ss.end;
  for (int L = 0; L < kLayers; ++L, b += kRingBytes) {
    CUDA_CHECK(cudaMemcpyAsync(b, ring_kv[L], (size_t)attn::kRing * kHead, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(b + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32,
                               cudaMemcpyDeviceToDevice, st));
  }
  for (int k = 0; k < kKvSlots; ++k, b += kCarryBytes) {
    CUDA_CHECK(cudaMemcpyAsync(b, kvs[k].carry_kv, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(b + kCarryBytes / 2, kvs[k].carry_sc, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
  }
  if (spec) spec_windows(*spec, ss.end + kEndBytes, -1, 0, true, st);
}

void Model::Impl::state_restore_end(int h) {
  if (!store_of().sess.count(h)) die("state: session %d restores its end with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.end) die("state: session %d has no end snapshot", h);
  const uint8_t* b = ss.end;
  for (int L = 0; L < kLayers; ++L, b += kRingBytes) {
    CUDA_CHECK(cudaMemcpyAsync(ring_kv[L], b, (size_t)attn::kRing * kHead, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ring_sc[L], b + (size_t)attn::kRing * kHead, (size_t)attn::kRing * kHead / 32,
                               cudaMemcpyDeviceToDevice, st));
  }
  for (int k = 0; k < kKvSlots; ++k, b += kCarryBytes) {
    CUDA_CHECK(cudaMemcpyAsync(kvs[k].carry_kv, b, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(kvs[k].carry_sc, b + kCarryBytes / 2, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
  }
  if (spec) spec_windows(*spec, ss.end + kEndBytes, -1, 0, false, st);
}

// What session h holds from position `from` on no longer belongs to its history: snapshots above it,
// the end snapshot if it is above, and saved positions from it on.
void Model::Impl::state_drop(int h, int from) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  if (ss.disk) state_unspill(h);  // its blocks are in its copy: back first (state_unspill)
  ss.cvalid = std::max(0, std::min(ss.cvalid, from));  // its copy does not cover what follows `from`
  for (auto i = ss.snaps.upper_bound(from); i != ss.snaps.end();) {
    store_release(i->second, snap_bytes(sp));
    i = ss.snaps.erase(i);
  }
  if (ss.end_pos > from) {
    store_release(ss.end, end_bytes(sp));
    ss.end = nullptr;
    ss.end_pos = -1;
  }
  ss.saved = std::min(ss.saved, from);
  if (shared_kv) {  // the pages above `from`, and the live slot's row's entries, back to the zero page
    // A live session's pages hold its entries up to its written length, beyond `saved` (the last save
    // only marks what the store had when it left a slot), so the pages below `from` stay whatever `saved` is.
    const int sl = slot_of(h);
    s.kv.drop(sl, ss.pages, s.bytes, from);  // a page a fork shares stays (§2.7)
    // The page holding `from` is written again from `from` on: if a fork shares it, this session takes its
    // own copy first (one page, once), so the other holders keep their entries (§2.7).
    const int pi = from / kPagePos;
    if (from % kPagePos && pi < (int)ss.pages.size() && s.kv.holders(ss.pages[pi]) > 1) {
      uint8_t* p = cow_page(h);
      if (!p) die("state: session %d's restart at %d has no page for its copy (the shared KV pool is full)", h, from);
      CUDA_CHECK(cudaMemcpyAsync(p, ss.pages[pi], kPageBytes, cudaMemcpyDeviceToDevice, st));
      s.kv.release(ss.pages[pi]);  // the others keep it
      ss.pages[pi] = p;
    }
    if (sl >= 0) write_slot_row(sl);  // the row belongs to the slot: the pages, then the zero page
    return;
  }
  while ((int)ss.pages.size() > (ss.saved + kPagePos - 1) / kPagePos) {
    store_release(ss.pages.back(), kPageBytes);
    ss.pages.pop_back();
  }
}

void Model::Impl::state_free(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  if (shared_kv)  // a session live in a slot leaves it: the row is the slot's, written by its next take
    for (int j = 0; j < nslots; ++j)
      if (slot_sess.of(j) == h) slot_sess.set(j, 0);
  for (uint8_t* p : ss.pages)
    if (!shared_kv || s.kv.release(p)) store_release(p, kPageBytes);  // §2.7: a fork may share them
  for (auto& sn : ss.snaps) store_release(sn.second, snap_bytes(spec != nullptr));
  store_release(ss.end, end_bytes(spec != nullptr));
  // the copy's blocks in the file, and its index: both go once the index is gone (settle)
  copy_release(s, h, ss.copy, spec != nullptr);
  s.sess.erase(it);
}

// src's snapshots up to upto into dst (src is kept while they are copied); those without room are not.
void Model::Impl::state_copy_snaps(int src, int dst, int upto) {
  Store& s = store_of();
  const auto it = s.sess.find(src);
  if (it == s.sess.end() || src == dst) return;
  if (it->second.disk) {
    s.pinned.push_back(dst);
    state_unspill(src);
    s.pinned.pop_back();
  }
  it->second.used = ++s.clock;
  // --shared-kv (§2.7): the fork's (dst's) own page list. It takes src's pages below `upto`'s page —
  // shared read-only, neither session writes them again — and copies the page holding `upto` (the fork
  // writes from `upto` on in its copy, src above its own saved in its own): one page, one copy, once.
  // The copy is queued on the model's stream after src's queued work. dst is the fork's new session
  // (empty list); the row of the slot it takes is written from this list by cmd.cpp's apply_ops.
  if (shared_kv) {
    Store::Sess& d = s.use(dst);
    const std::vector<uint8_t*>& sp = it->second.pages;
    const int pi = upto / kPagePos, keep = std::min(pi, (int)sp.size());
    d.pages.assign(sp.begin(), sp.begin() + keep);
    s.kv.hold(d.pages.data(), keep);  // the pages are the fork's now too: shared read-only
    d.saved = std::max(0, std::min(upto, it->second.saved));
    if (upto % kPagePos && pi < (int)sp.size()) {
      s.pinned.push_back(src);  // its pages while the copy allocates
      uint8_t* p = cow_page(dst);
      s.pinned.pop_back();
      if (!p) die("state: session %d's fork of %d at %d has no page (the shared KV pool is full)", dst, src, upto);
      for (int k = 0; k < kKvSlots; ++k) {
        const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio;
        const int n = std::min(E, it->second.saved / ratio - pi * E);
        for (int a = 0; a < 4 && n > 0; ++a)
          CUDA_CHECK(cudaMemcpyAsync(p + page_off(k, a), sp[pi] + page_off(k, a), (size_t)n * kArrBytes[a],
                                     cudaMemcpyDeviceToDevice, st));
      }
      d.pages.push_back(p);
    }
  }
  const std::vector<std::pair<int, uint8_t*>> from(it->second.snaps.begin(), it->second.snaps.upper_bound(upto));
  s.pinned.push_back(src);
  for (const auto& sn : from)
    if (uint8_t* b = state_snap_buf(dst, sn.first))
      CUDA_CHECK(cudaMemcpyAsync(b, sn.second, snap_bytes(spec != nullptr), cudaMemcpyDeviceToDevice, st));
  s.pinned.pop_back();
}

void Model::Impl::state_drop_snap(int h, int pos) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  it->second.used = ++s.clock;
  if (it->second.disk) state_unspill(h);  // its blocks are in its copy: back first
  const auto i = it->second.snaps.find(pos);
  if (i == it->second.snaps.end()) return;
  store_release(i->second, snap_bytes(spec != nullptr));
  it->second.snaps.erase(i);
}

void Model::state_reserve(size_t bytes) { d_->state_reserve(bytes); }
size_t Model::state_capacity() const { return d_->store_s ? d_->store_s->kv.pool.size : 0; }
void Model::state_slot_row(int s, int h) { d_->state_slot_row(s, h); }
bool Model::shared_kv_room(int slot, int upto) { return d_->shared_kv_room(slot, upto); }
bool Model::shared_kv_room(const int* slot, const int* pos, int rows) { return d_->shared_kv_room(slot, pos, rows); }
void Model::state_pin(const std::vector<int>& h) { d_->store_of().pinned = h; }
std::vector<int> Model::state_evicted() {
  std::vector<int> v;
  if (d_->store_s) v.swap(d_->store_s->evicted);
  return v;
}
void Model::state_save(int h, int from, int to) { d_->state_save(h, from, to); }
void Model::state_load(int h, int to) { d_->state_load(h, to); }
void Model::state_snap(int h, int pos) { d_->state_snap(h, pos); }
void Model::state_snap_end(int h, int pos) { d_->state_snap_end(h, pos); }
void Model::state_restore(int h, int pos) { d_->state_restore(h, pos); }
void Model::state_restore_end(int h) { d_->state_restore_end(h); }
void Model::state_drop(int h, int from) { d_->state_drop(h, from); }
void Model::state_free(int h) { d_->state_free(h); }
void Model::state_copy_snaps(int src, int dst, int upto) { d_->state_copy_snaps(src, dst, upto); }
void Model::state_drop_snap(int h, int pos) { d_->state_drop_snap(h, pos); }
uint64_t Model::state_recover_begin(const std::string& base, uint64_t run_id) { return d_->state_recover_begin(base, run_id); }
std::vector<std::pair<int, uint64_t>> Model::state_recover_scan(double max_age_s) { return d_->state_recover_scan(max_age_s); }
void Model::state_recover_take(const std::vector<std::pair<int, uint64_t>>& keep) { d_->state_recover_take(keep); }
void Model::state_recover_clean() { d_->state_recover_clean(); }
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->state_disk(base, bytes, staging, reuse, run_id);
}
void Model::state_copy(int h) { d_->state_copy(h); }
void Model::state_inval(int h, int q) { d_->state_inval(h, q); }
void Model::state_sync() { d_->state_sync(); }
std::vector<int> Model::state_sessions() const {
  std::vector<int> v;
  if (d_->store_s)
    for (const auto& kv : d_->store_s->sess) v.push_back(kv.first);
  return v;
}
size_t Model::state_disk_capacity() const { return d_->store_s && d_->store_s->disk ? d_->store_s->dpool.size : 0; }
size_t Model::state_disk_bytes() const { return d_->store_s ? d_->store_s->dbytes : 0; }
std::string Model::state_disk_path() const { return d_->store_s && d_->store_s->disk ? d_->store_s->disk->path() : ""; }
uint64_t Model::state_run_id() const { return d_->store_s ? d_->store_s->run_id : 0; }
void Model::state_spill(int h) { d_->state_spill(h); }
void Model::state_unspill(int h) { d_->state_unspill(h); }
Model::StateInfo Model::state_info(int h) const {
  StateInfo r;
  if (!d_->store_s) return r;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return r;
  const Store::Sess& ss = it->second;
  const bool spec = d_->spec != nullptr;
  r.saved = ss.saved;
  r.copy_seq = ss.copy.seq;
  r.on_disk = ss.disk;
  if (ss.disk) {  // the copy is the session: its blocks and their file offsets
    r.end_pos = ss.copy.end.first != Pool::kNone ? ss.end_pos : -1;
    for (const auto& sn : ss.copy.snaps) r.snaps.push_back(sn.first);
    r.disk_bytes = ss.copy.pages.size() * Pool::round(kPageBytes) + ss.copy.snaps.size() * Pool::round(snap_bytes(spec)) +
                   (ss.copy.end.first != Pool::kNone ? Pool::round(end_bytes(spec)) : 0);
    return r;
  }
  r.end_pos = ss.end ? ss.end_pos : -1;
  for (const auto& sn : ss.snaps) r.snaps.push_back(sn.first);
  r.bytes = ss.pages.size() * Pool::round(kPageBytes) + ss.snaps.size() * Pool::round(snap_bytes(spec)) +
            (ss.end ? Pool::round(end_bytes(spec)) : 0);
  return r;
}
size_t Model::state_bytes() const { return d_->store_s ? d_->store_s->bytes : 0; }
StoreStats Model::state_stats() const { return d_->store_s ? d_->store_s->stats : StoreStats{}; }
void Model::drain() { d_->drain(); }

}  // namespace model
