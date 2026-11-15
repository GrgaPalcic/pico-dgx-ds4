# Shared KV: one pool, page tables, no copy between a slot and the store

The design every later shared-KV task follows. The feature is the engine flag `--shared-kv`
(`dsgb10_rank`, [main.cpp](../engine/server/main.cpp)), off by default: with it off every code path,
kernel launch, memory layout, log line and output hash is byte-for-byte as before.

What it changes, in one paragraph. Today a live session's compressed caches are per-slot contiguous
buffers (up to ~890 MiB a slot at 1M positions, reserved for `--slots` slots whether used or not), the
session store is a second, separate pool of ~12 GiB, and moving a session between a slot and the store
copies its caches (`state_save`, `state_load`), so a returning session sits in memory twice. With
`--shared-kv` the slot's compressed caches, the index keys and the store's pages are one pool: a page
holds 4096 positions of all four kv sources, exactly as the store's pages do today, and a session owns
a page table whose entries are the pool pages it holds. Kernels read and write the caches through that
table; a page is written in place; a session takes or leaves a slot by pointing a table, with no copy;
a fork shares its base's full pages and copies only the partial tail page. The store's page format, the
disk tier's file format and the kept-sessions index are unchanged.

Paths below are relative to `ds41f-4xgx10-switchless/`. A bare file name is under `engine/` and unique
there, so `state.h` is `engine/model/state.h`, `attn.h` is `engine/attn/attn.h`, `indexer.cu` is
`engine/attn/indexer.cu`, `test_pattn.cu` is `engine/kernels/tests/test_pattn.cu` and so on; where a
name is not unique it is written out (`model/prefill.cu`). Line numbers are of this commit (`f211791`).

## 1. Today, verified in source

### 1.1 The per-slot working state

- **A kv source's caches.** `engine/model/impl.h:77-95`: `KvSrc { uint8_t *ckv, *csc, *ik, *iksc; int
  cap; float *carry_kv, *carry_sc; }`. Entry sizes are 256 B (compressed KV, e2m1 pairs), 32 B (its
  e4m3 scales, one per 16), 64 B (the index key, e2m1 pairs transposed) and 4 B (its ue8m0 scales, one
  per 32) — `state.h:38` `kEntryBytes = 256 + 32 + 64 + 4`. `cap` is `unit_ctx / ratio + 64`
  (`model.cu:99`), 64 extra entries so a slot's caches cover one more group than its context.
- **Slot s's buffers are slot 0's + s × stride** (`impl.h:85-88`): `ckv_stride() = cap * 256`,
  `csc_stride() = cap * 32`, `ik_stride() = cap * 64`, `iksc_stride() = cap * 4`; `at(int s)`
  (`impl.h:90-94`) adds them, and `KvSrc::kCarryStride = 16 * kHead` floats (`impl.h:89`) the carries'
  stride (the carry is a 16-row ring by position & 15, not a per-position cache).
- **Whole caches per slot, contiguous.** `model.cu:94-105`: per unit, `ring_kv0[L]`/`ring_sc0[L]` for
  the 40 window rings and, for the four kv sources (`kv_slot(L)`), `ckv/csc/ik/iksc = dalloc(nunits ×
  stride)` and the two carries. Nothing is allocated per session.
- **Units and big slots.** `config.h:25-39`: `kMaxSlots = 16`, `kMaxUnits = 64`, `kUnitMul = 8`. A big
  slot is 8 units of the small slot's size laid end to end "so every kernel's slot × stride addressing
  stays uniform"; `model.cu:78-92` sets `unit_of[s]` and `slot_ctx_[s]` (slots `[0, big_slots)` cover
  `max_ctx`, the rest `max_ctx / kUnitMul`), `model.cu:93` dies past `kMaxUnits` units. So 16 × 1M is
  128 units > 64: the two are mutually exclusive today.
- **The live slot.** `model.cu:431-439` `use_slot(s)`: `cur_unit = unit_of[s]`, `ring_kv[L] = ring_kv0[L]
  + u * kRingKvBytes`, `kvs[k] = kvs0[k].at(u)`, and the Engram history is swapped per unit
  (`hists[u]`).
- **A step's view of the state.** `model.cu:441-462`: `step_win(L)`/`step_ckv(k)`/`step_keys(k)`/
  `step_carry(k)` hand the kernels slot 0's pointers plus `slot = in_d->slot` (the step's per-row unit)
  and the strides above, when `nunits > 1`; with one unit the table is null and every row is at the
  pointers. `attn/attn_common.cuh:134-170` (`win_at`, `ckv_at`, `ik_at`, `carry_at`) adds
  `slot[r] * stride` per row on the device.
- **Sizes at `--max-ctx 1048576`** (recomputed from the source; the handoff's numbers came from the
  running service's log lines, `engine.md` §Serving):

  | per slot, 1M positions | ratio-2 source (×3) | ratio-1 source (20) | 40 rings | carries | total |
  |---|---|---|---|---|---|
  | ckv | 134,234,112 (128.0 MiB) | 268,451,840 (256.0 MiB) | | | |
  | csc | 16,779,264 (16.0) | 33,556,480 (32.0) | | | |
  | ik | 33,558,528 (32.0) | 67,112,960 (64.0) | | | |
  | iksc | 2,097,408 (2.0) | 4,194,560 (4.0) | | | |
  | subtotal | 186,669,312 (178.0 MiB) | 373,315,840 (356.0 MiB) | 5,406,720 (5.16) | 262,144 (0.25) | **895.4 MiB** |

  (`cap = 524352` for the ratio-2 sources and `1048640` for layer 20; rings 40 × 256 × 528 B.) At the
  service's `--slots 16 --big-slots 4`: 4 × 895.4 + 12 × 116.7 = 4,982 MiB ≈ 4.87 GiB, the handoff's
  "~5 GB of slot reservation" (a small slot is `max_ctx / 8`: 116.7 MiB). At `--slots 16` with no big
  slots it would be 16 × 895.4 = 14.0 GiB.
- **`kstride` is not a cache stride.** `model.cu:120-121`: `kstride = max(max_ctx, kPoolBlocks * 8)`
  rounded to `attn::kScoreAlign` — the stride of the indexer's *scan output* buffers (`keys`, `bkeys`,
  `model.cu:161-162`), and the largest scan a captured graph covers (`impl.h:281 scan_ctx`). It stays
  as it is under the flag.

### 1.2 The store

- **One pool, reserved once.** `state.h:47-58` `Pool`: `base`, `size`, a map of free ranges, best fit
  (`state.cu:87-104` `take_off`/`take`), coalescing (`state.cu:125-141` `give_off`), `take_at` for a
  kept file's offsets (`state.cu:106-118`), all in `kPoolGrain = 64 << 10` blocks (`state.h:43`).
  `state.cu:336-343` `state_reserve` mallocs it once; `main.cpp:735-741` sizes it (`--session-gib`,
  default 12 GiB, capped by the smallest rank's free device memory less `--session-margin-gib` and its
  available host memory less the margin and the Engram row cache). `state.cu:351-372` `store_alloc`
  hands out a block of session `h`, evicting the least recently used session the command does not use
  (`state.h:100` `pinned`) — spilled to disk, or freed without a disk tier — and returns `nullptr` when
  there is none left: "a snapshot not taken, pages not saved, no end snapshot. Nothing in the store
  fails a request" (`state.h:7-12`).
- **A stored session.** `state.h:63-79` `Sess`: `pages` (compressed caches, page i = positions `[i,
  i+1) * kPagePos`), `saved` (positions `[0, saved)` are in the pages), `snaps` (window snapshots by
  position, the draft module's windows after them), `end`/`end_pos` (every layer's whole ring and the
  carries at a position), `used` (the LRU clock), and, spilled, the same blocks' offsets in the disk
  tier's file.
- **The page format.** `state.h:37-42`: `kPagePos = 4096`, `kPageBytes = (4096/2 * 3 + 4096) * 356 =
  3,645,440` B — one page holds all four kv sources for 4096 positions: 2048 entries of each ratio-2
  source (layers 2, 8, 14) and 4096 of layer 20. `state.cu:34-42` `page_off(k, a)` gives the byte
  offset of source `k`'s array `a` (kv, sc, k, ksc) inside a page. A page is allocated rounded to the
  grain: `Pool::round(3,645,440) = 3,670,016` B = 56 × 64 KiB = 3.5 MiB.
- **The store's copies.** `state.cu:555-577` `state_save(h, from, to)` allocates pages up to `to` and
  copies entries `[from/ratio, to/ratio)` of every source from the live slot's caches into them;
  `state.cu:579-593` `state_load(h, to)` copies them back into the live slot's caches and keeps the
  pages (a returning session is in memory twice). `state.cu:639-710` `state_snap`/`state_snap_end`
  copy ring rows and the carries into a snapshot block; `state.cu:662-680` `state_restore`/
  `state_restore_end` copy them back; `state.cu:711-750` `state_drop` frees what is above a position;
  `state.cu:751-763` `state_free`; `state.cu:764-781` `state_copy_snaps` (a fork's snapshots).
  Every copy is a `cudaMemcpyAsync` on the model's stream, so it lands in order with the steps around
  it (`state.cu:1-5`).
- **The disk tier.** `spill.h:28-88` `SpillFile`: one preallocated O_DIRECT file per rank, offsets and
  sizes on the grain, writes through pinned staging after the work queued on the stream
  (`spill.h:42-56`), per-block checksums in pinned memory (`spill.h:58`), a writer thread
  (`spill.cpp:178-222`). `state.cu:414-476`
  `state_spill` copies a session's blocks out and frees them at once; `state.cu:478-533` `state_unspill`
  reads them back as far as the store has room, checking every block; `state.cu:391-412` `disk_take`
  makes room in the file, dropping the least recently used spilled sessions.
- **Kept sessions.** `state.cu` `state_recover_begin`/`state_recover_scan` validate the copies' index
  files under `base.d/` (magic, rank, run id, layout, age, every block on the grain and inside the
  file); `state_recover_take` opens the file as it is, its sessions spilled ones, each block's checksum
  seeded from the index; `state_copy` writes a finished turn's copy and its index, and `state_sync` puts
  every copy in the file and on the disk. The `CopyHeader`'s `page_bytes`, `snap_bytes` and `end_bytes`
  must equal this build's layout, so a layout change invalidates the file (and `run_id` includes the
  binary's bytes, `main.cpp` `run_id`).

### 1.3 The moves between a slot and the store

- `engine/server/serve.cpp:677-687` `persist_ops(x, v)`: for the slot's session `x`, a `kStSave` of
  `[info.saved, history)` and a `kStSnapEnd` at the history's end (and it records the exact ranges
  `end_lo/end_clo/end_w/end_dec_lo`).
- `serve.cpp:688-704` `persist_slot(s, free)`: the slot's session's state into the store, in a command
  acting on slot `s`; `free` also lets the slot go (`slot_[s] = 0`, `x->slot = -1`).
- **Every call site**: `serve.cpp:610` (a planned stop, every slot), `:696` (inside `persist_slot`),
  `:755` (evicting a session live in a slot, before spilling it), `:955` (a session that no longer fits
  its slot moves to a big one), `:958` (a fork's base in flight keeps its slot; its state goes to the
  store and the fork starts from there), `:996` (`go`: the slot's previous session keeps its state in
  the store before another session takes the slot).
- The ops travel as `Cmd::ops` (`cmd.h:48-50` `StateOp`, `:30-46` the op codes) and are applied on
  every rank by `cmd.cpp:57-73` `apply_state` → `cmd.cpp:75-88` `apply_ops`, which calls
  `Model::state_pin` (`model.h:270`) with the command's sessions, its handle and the slots' live
  sessions (`cmd.h:78-81`) for the duration of the command: those are the sessions the store may not
  evict while it runs.
- A prefill chunk's snapshots are taken on the way (`model/prefill.cu:656-763` `Model::Impl::prefill`), and a
  decode loop's by `Cmd::snap_every` (`cmd.h:99`).

### 1.4 Every hit of the addressing grep

`grep -rn -E "ckv_stride|ik_stride|slot_tab|slot \* |unit_of|kstride" engine`. "Cache addressing" = the
hit is about where a session's compressed caches, index keys or carries live, so the shared-KV work
must handle it; "unrelated" = another `slot` (a window ring's 256-row ring, a carry's 16-row ring, an
MoE workspace, a test's own buffer, a scan buffer's stride).

| hit | class | why |
|---|---|---|
| `attn/compressor.cuh:144-145` `carry.kv[slot * kHead + c]` | unrelated | the carry's ring slot, `position & 15` |
| `attn/prep.cu:85-86` `win.kv + slot * kHead` | unrelated | the window ring, `(p - base) & mask` |
| `attn/sparse_attn.cu:140,142` | unrelated | window ring reads |
| `attn/tests/attn_tests.cu:399-471`, `bench_attn.cu:76,205-293`, `bench_index.cu:62-286`, `fused_tests.cu:423-823`, `slot_tests.cu:451-468` | unrelated | `kstride` there is the indexer's scan-buffer stride |
| `attn/tests/bench_step.cu:144` `ik_k.p + slot * (ik_k.n / g_nslots)` | cache addressing | the bench's per-slot keys (test-side; needs a table row) |
| `dspark/dspark.cu:44` `mh[r * kMainK + slot * kDim + k]` | unrelated | the main-hidden buffer's kv-source slot |
| `dspark/tests/dspark_tests.cu:407-408` | unrelated | a window's ring slot in a test |
| `engram/tools/iobench.cpp:64-65` | unrelated | an iovec's slot count |
| `kernels/gemv16.cu:614`, `kernels/gemv_fp8.cu:313` | unrelated | a weight staging ring (a software pipeline stage) |
| `kernels/moe.cu:548-549,797-798,891,952,1047-1050,1102-1105` | unrelated | a routed expert's slot in the MoE workspace |
| `kernels/prefill_attn.cu:89,91,118` | unrelated | window ring slots |
| `kernels/prefill_comp.cu:190-191` | unrelated | the carry's ring slot |
| `kernels/tests/test_pattn.cu:131` | unrelated | a ring slot in a test |
| `model/config.h:36` | cache addressing | the comment on `kUnitMul` ("slot × stride addressing stays uniform") |
| `model/config.h:83` `kKvSlots = 4` | cache addressing | how many kv sources there are |
| `model/impl.h:85-88,90-94` `ckv_stride()`…`at(s)` | cache addressing | the per-slot strides |
| `model/impl.h:139` `unit_of[]`, `:141 kstride` | both | `unit_of` is cache addressing; `kstride` is a scan buffer (1.1) |
| `model/model.cu:87` `unit_of[s] = nunits` | cache addressing | unit assignment |
| `model/model.cu:100-105` `dalloc(nunits * stride)` | cache addressing | the per-unit caches and carries |
| `model/model.cu:120-121,161-162` | unrelated | `kstride` and the scan buffers |
| `model/model.cu:433` `unit_of[s]`, `:449,455` step strides | cache addressing | `use_slot`, the step views |
| `model/model.cu:549,555` | unrelated | `index_dense`'s `keys`/`kstride` |
| `model/model.cu:847,917` `in.slot[r] = unit_of[…]` | cache addressing | the step's per-row unit |
| `model/prefill.cu:97-98,388-391` | unrelated | the prefill index buffers and their strides |
| `model/prefill.cu:405-406` `ring_kv[L] + slot * kv` | unrelated | `ring_copy`, window ring rows |
| `model/prefill.cu:618` `p.ced_rows + slot * kCedRow` | unrelated | CED's kept rows per slot |
| `model/prefill.h:63` | unrelated | a comment on `keys` |
| `model/spec.cu:1179,1229,1268` `unit_of[b.slot[k]]` | cache addressing | the spec steps' per-entry unit |
| `model/state.cu:603` `ring + slot * row` | unrelated | `ring_rows`, snapshot rows |

`state.h`/`state.cu`'s own `pages`, `kPageBytes` and `page_off` are the store's (1.2), and the flag
makes them the slot's too.

## 2. Design

### 2.1 The pool, and what a page holds

One pool per rank, reserved once at startup, the store's pool (`state.h:47-58`, `state.cu:336-343`)
with the slot's caches moved into it. Allocation, freeing, coalescing and the LRU eviction are the
existing best-fit `Pool`; a page is `Pool::round(kPageBytes) = 3,670,016` B (3.5 MiB) of it.

A page is exactly the store's page: 4096 positions of all four kv sources at `state.cu:34-42`'s
offsets — 2048 entries each of the ratio-2 sources (2, 8, 14) at 729,088 B and 4096 of layer 20 at
1,458,176 B. The store's page format, its `kPageBytes`, its checksums, its disk blocks and the kept
index's layout are therefore unchanged, and a page that a slot writes is already in the store's format:
saving a session is bookkeeping, not a copy (2.6).

The pool's size is the store's size plus what the flag no longer reserves (§2.9).

### 2.2 What stays per slot

| what | size at 1M, per slot | why it stays |
|---|---|---|
| window rings, 40 layers | 5.16 MiB | a 256-row ring by position; a stored session's window is a snapshot |
| compressor carries, 4 sources | 0.25 MiB | a 16-row ring by position & 15 |
| the drafter's windows, 4 blocks | 0.52 MiB (`dspark.cu:919`, `dspark.cu:963-968`) | a 256-row ring by position, per block |
| the slot table's row (one per slot) | 2,056 B (`257 × 8`, `max_ctx/4096 + 1`) | 2.3 |
| **total** | **5.9 MiB** (was 895.4 MiB) | |

The rings and the carries keep their slot addressing (`attn_common.cuh:134-170`) and their snapshots
unchanged; a session's window state moves between a slot and the store exactly as today, by the copies
`state_snap`/`state_restore` already make (2.9 MiB and 6.1 MiB a session, `engine.md` §Serving).

### 2.3 The page table, and how a step reaches it

**One table row per slot, in device memory**: `uint64_t slot_tab[kMaxSlots][max_pages]`
(`max_pages = max_ctx / kPagePos + 1`), 2,056 B a row at 1M and 32.9 KB for the sixteen slots,
allocated at startup with the pool. Page i of a session holds all four kv sources' entries for
positions `[4096 i, 4096 (i+1))`, so one row of page pointers serves every source, each source reading
it at its own `page_off` offsets and entries-per-page.

**`ops::StepIn` does not change.** Under the flag the units collapse to the slots (`unit_of[s] = s`,
`nunits = nslots`), so the `in.slot[r]` a step already carries (`ops.h:44-46`; filled at
`model.cu:847,917`, `spec.cu:1179,1229,1268`) names each row's slot, and the table is indexed exactly
as the window rings and the carries already are: `tab = slot_tab[0] + slot[r] * max_pages`. A captured
graph reads the tables from device memory it already reads (`in_d->slot`), so a replay needs no update
and no new graph parameter, and the eager, batch and spec paths all reach the tables the same way — by
the slot each of their rows carries.

**The host owns the page lists.** A session's pages are `Store::Sess::pages` (`state.h:65`) as today,
live or stored; the device row is a mirror written on the model's stream:

- a session takes a slot (`use_slot`, `model.cu:431-439`): the whole row is written — its pages at
  `[0, pages.size())` and the zero page (below) at `[pages.size(), max_pages)`, so a row never holds a
  stale entry that a reader could dereference;
- `shared_kv_room` adds a page to the session live in that slot: one 8-byte host-to-device copy into
  its entry;
- a session leaves a slot: nothing; the next session to take the slot rewrites the row.

A **zero page** — one 3.5 MiB block, zeroed once at reserve and never handed out by the allocator — is
what every entry above a session's list holds. An out-of-list read then returns zeros, which is what
today's fresh caches hold (`impl.h:31-39` `dalloc` zeroes every slot buffer), and the host's row write
is the only place that has to know about it.

The device-side view, one struct shared by `Compressed` and `IndexKeys` (both of one kv source use the
same row and the same page layout):

```cpp
// engine/attn/pages.cuh
namespace attn {
// A kv source's four caches (ckv 256 B, csc 32, ik 64, iksc 4 per entry) in the shared pool's pages
// (page i: positions [i, i+1) * 4096, state.cu's page_off), or the slot's contiguous caches. `tab` is
// null for the contiguous path (--shared-kv off, and the drafter's own windows): the helpers below are
// then exactly the pointer arithmetic of before. `slot` gives each step row its slot (ops::StepIn::slot),
// `stride` the slot table's entries.
struct KvPages {
  const uint64_t* tab = nullptr;  // [stride] page base pointers; null: contiguous
  const int* slot = nullptr;      // step rows -> slots; null: `tab` is this slot's row
  uint32_t stride = 0;            // entries per slot (max_pages)
  uint32_t off[4] = {};           // bytes from a page's base: ckv, csc, ik, iksc (page_off)
  int shift = 11;                 // log2(entries per page): 11 (ratio 2) or 12 (ratio 1)
  __device__ bool paged() const { return tab != nullptr; }
};
// The row's own table (one load per step row, not per entry): row r's slot's.
__device__ __forceinline__ KvPages pages_at(KvPages p, int r) {
  if (p.slot) { p.tab += (size_t)p.slot[r] * p.stride; p.slot = nullptr; }
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
__device__ __forceinline__ uint8_t* page_ent_m(const KvPages& p, uint8_t* base, int e, int a, size_t bytes);
}  // namespace attn
```

`Compressed` and `IndexKeys` (`attn.h:47-60`) each gain one `KvPages pages;` field (null `tab` by
default) and thin accessors so a kernel call site stays one expression:

```cpp
struct Compressed {
  // … as today: kv, sc, cap, slot, kv_stride, sc_stride
  KvPages pages;
  __device__ const uint8_t* kv_at(int e) const;   // page_ent<kPaged>(pages, kv, e, 0, 256)
  __device__ const uint8_t* sc_at(int e) const;   // (…, sc, e, 1, 32)
  __device__ uint8_t* kv_at_w(int e); __device__ uint8_t* sc_at_w(int e);
};
struct IndexKeys { /* k, sc, cap, slot, k_stride, sc_stride, pages, k_at(e), sc_at(e) */ };
```

`ckv_at`/`ik_at` (`attn_common.cuh:143-160`) advance `pages.tab` by `slot[r] * pages.stride` exactly as
they advance `kv`/`k` today, so the per-row indirection is unchanged. The step's views are:

```cpp
attn::Compressed Model::Impl::step_ckv(int k) const {
  attn::Compressed c = kvs0[k].compressed();          // the flag-off path: contiguous, as today
  if (shared_kv) { c.pages = page_view(k); c.pages.slot = in_d->slot; }   // flag on
  else if (nunits > 1) { c.slot = in_d->slot; c.kv_stride = …; }
  return c;
}
```

`page_view(k)` is `KvPages{slot_tab, nullptr, max_pages, {off_ckv, off_csc, off_ik, off_iksc}, shift}`
(the slot table's base and the source's `page_off` offsets); `step_win`, `step_carry` and the drafter's
windows are untouched (they were already per unit, and a unit is a slot under the flag). `next_in`
(`model.cu:764-783`), `launch_batch` (`model.cu:817-871`) and the spec fills (`spec.cu`) need no
change at all: they already write `in.slot[r] = unit_of[slot]`, which is the row's slot under the flag.

The **eager paths** (a prefill chunk, the state operations, the snapshots) act on one slot at a time:
`use_slot(s)` (`model.cu:431-439`) sets `kvs[k].pages.tab = slot_tab[s]` with `slot = nullptr`, so those
kernels address one session's pages directly. `model/prefill.cu:367-398`'s
`index_p`, `:400-433`'s `ring_copy` and `:435-580`'s `attention_p` read `kvs[…]` as they do today.

**Flag off**, `shared_kv` is false, no `KvPages` is ever filled, `ops::StepIn` is byte-for-byte the
struct it is today (the flag's tables live in `Model::Impl`, not in the step ring), and every kernel
takes its `<false>` instantiation: the address arithmetic is the same expression as today (§4).

**Built.** `pages_at`/`page_ent`/`page_ent_m` live in `engine/attn/pages.cuh` (T1), not in
`attn_common.cuh`; `attn.h` includes it. `KvPages::off` is `uint32_t[4]`, `shift` defaults to 11 and
`page_view(k, s)` fills them (`page_off(k, a)`: each source its own block of the page); `paged()` is `__host__ __device__`, because the
host picks the instantiation. `Compressed`/`IndexKeys`' write accessors are non-const. A step wider than
`kDecodeRows` (wide64, `DSGB10_WIDE`, which the service sets) runs the indexer and the compressor's emit in
groups of 16 rows (`model.cu`); each group offsets `KvPages::slot` by its first row, as it offsets the
contiguous `slot` pointer (fc41169; before it, rows 16.. read and wrote rows 0..15's sessions' pages).
`sparse_attn` takes the step's rows in one call and needs no offset.

### 2.4 The address helper

`attn::page_ent<kPaged>` (2.3) is the only address helper; every kernel that touches a cache uses it
through the accessors above. Its cost per entry is one 8-byte table load (L1-resident: a page covers
2048 or 4096 entries, and a kernel walks entries in order), one shift, one mask and one multiply-add.
A kernel that already walks a whole page's entries pays it once per 4096 positions per lane; the dense
index scan pays it once per 16-position tile per lane (§2.10).

### 2.5 Allocation, pinning, eviction, and the full pool

- **Pages are allocated on the host, never in a kernel**, and in the same order on every rank (the
  store's invariant, `state.h:7-12`): the pool's blocks and the sessions' page lists are the same on all
  four ranks, so a full pool is hit at the same moment everywhere. The model exposes

  ```cpp
  // model.h: room for the sessions in `slot`/`slots` up to `upto` positions (or each row's pos[r] + 1),
  // allocated now. False: the pool has no room and nothing can be evicted; the caller must not launch
  // (below). With --shared-kv off this is a constant true and allocates nothing.
  bool shared_kv_room(int slot, int upto);
  bool shared_kv_room(const int* slot, const int* pos, int rows);
  ```

  and it is called **inside the loops every rank runs**, immediately before the launch whose rows it
  covers, so the step is never launched without its pages and every rank decides alike:
  - `cmd.cpp`'s `decode_loop` (`:345-…`), before `m.launch(...)` for the next position;
  - `cmd.cpp`'s `spec_loop` (`:165-…`), before `m.launch_verify(...)`, for `pos_bound + 1 +
    kMaxSessRows` positions (a verify step's rows are the device's choice, so the bound is what the loop
    knows);
  - `cmd.cpp`'s `batch_loop` (`:405-…`) and `spec_batch_loop` (`:577-…`), before the step, per entry: an
    entry that fails is stopped (`kEndPool`, below) and dropped from the step's live set, the others run;
  - `cmd.cpp`'s `run_prefill` (`:118-124`), before `m.prefill(...)`, for `c.pos0 + c.rows` positions.
  A step never allocates inside a captured graph, and no kernel needs an allocator. A fork's COW page
  (§2.7) and a session's list are allocated the same way, from the state operations' host code.
- **A live slot's pages are pinned, and a live session is never *spilled*.** A session live in a slot
  cannot be spilled to disk: its pages are its working state. `state_spill` refuses one
  (`die("state: session %d is live in slot %d")`), and the server's `evict` (`serve.cpp:737-768`)
  skips such sessions; a session that must be spilled first lets its slot go (the server's
  `persist_slot(s, true)`, a no-op under the flag, `serve.cpp:755`), which makes it a stored session
  like any other. `state_free` *may* be called on a live session (a RELEASE, `serve.cpp:704-713`): under
  the flag it returns the session's pages to the pool (its row belongs to the slot). That is safe, and stays safe,
  because **no step that names a session is in flight when its slot, its row or its pages change
  hands**: a decode or spec loop waits for the step queued behind the one that stopped before it
  returns (`cmd.cpp:307-311`), the prefill path drains the model's stream before the request leaves
  `go` (`serve.cpp:1105,1128`) and before an `unspill` returns (`serve.cpp:778`), and a slot is only
  handed to another session after the loop that held it has returned. T4 must keep this invariant (and
  state it where the flag's paths rely on it).
- **Eviction is the store's, unchanged**: `store_alloc` (`state.cu:351-372`) takes the least recently
  used session the command does not use, spills it to the disk tier (`state.cu:414-476`) or frees it,
  and frees its pages into the pool. A session that leaves a slot first (the server's `persist_slot(s,
  true)`, a no-op under the flag) is evictable like any stored one.
- **When the room is missing** (a live session needs a page and nothing is evictable) the step or the
  chunk is never launched and the request ends with an error, never a crash and never a silent short
  write:
  - the loop stops at that step boundary on every rank: `DecodeEnd::why = kEndPool`, a new value of the
    enum at `cmd.h:137`, and for a batch loop that entry's `why`, the others going on. The tokens
    already sent stand; the session's state is consistent at its last written position (its pages hold
    exactly the entries of the positions computed so far, so continuing, spilling or freeing it is
    valid);
  - rank 0's server maps `kEndPool` to `kDone` reason `kError` with the text
    `the shared KV pool is full (%.2f of %.2f GiB in use; session %016llx needs a page for position %d)`
    — the place where `DecodeEnd::why` becomes the DONE reason today is `serve.cpp:1333-1336` (one
    flight) and `:1432-1436` (a batch) — and logs `serve: request %llu: <text>`;
  - **dsfront** reports it as it reports every engine error (`frontend/src/server.rs:571-575`): a
    non-streaming request gets HTTP **500** with
    `{"error": {"message": "<the text>", "type": "server_error", "param": null, "code": 500}}`
    (`server.rs:121-135`, `:863`); a streaming request has already had HTTP 200, so the error is its
    last SSE event, the same JSON, then `data: [DONE]` (`server.rs:962-964`, `:984`). Either way the
    front-end abandons its session-table entry and releases the session, so the engine frees its state
    and the conversation's next turn starts fresh (a cold prefill), as after any engine error;
  - the model never dereferences a missing page: a launch that finds one dies
    (`model: step at %d of session %d without its page (shared_kv_room first)`), so a bug cannot read a
    null table entry;
  - the evictions a `shared_kv_room` inside a loop makes are the store's usual ones, and the server collects
    them with `forget_evicted()` (`serve.cpp:725-736`) right after each loop returns — the flag's task
    adds that call where it is missing (after `decode_one`'s loop, `serve.cpp:1305-1330`, and after
    `decode_batch`'s, `:1400-1440`).
- The disk tier's own outcome is unchanged: when the file is full, its least recently used sessions are
  dropped (`state.cu:391-412`).

**Built.** The host side's entry point is `SharedKv::room(pages, bytes, upto, evict)` (the session's
list, the store's byte counter and a callback into the store's LRU choice), not `room(slot, upto)`; the
model's API is by slot, as §2.5's callers need. The room calls also sit in `Model::step`
(`model.cu:1317`), which §2.5's list does not name: it is the one eager launch the golden and bench
modes make outside `cmd.cpp`, and TASK.md's T4 asked for it. Outside serve the modes give every slot a
session of its own (the handles `-1 … -nslots`), so their room calls find one. A `spec_loop`'s last
step (the step that feeds a stopped request's last token) keeps the request's own stop reason when its
page is missing rather than `kEndPool`: the request was ending anyway.

### 2.6 Taking and leaving a slot, and the tail page

- **Taking a slot** for a session whose state is in the store: `kStLoad`/`kStRestore`/`kStRestoreEnd` as
  today (`serve.cpp:1004-1014`), except that `state_load(h, q)` under the flag only *points* the slot's
  table row at the session's pages (§2.7) — no cache copy. The row itself is written by `apply_ops`
  (`cmd.cpp:75-88`) from `Cmd::live[]` (`cmd.h:78-81`, the handles of the sessions live in the slots,
  which every command already carries), after the command's state operations have run: the session's
  host list into `[0, pages.size())` and the zero page above it, 2,056 B at 1M. So a slot take costs one
  2 KB copy, and every later page is one 8-byte copy into its entry (§2.3). `state_restore` (a window
  snapshot into the ring) is unchanged.
- **Leaving a slot**: `persist_ops` (`serve.cpp:677-687`) issues `kStSave` and `kStSnapEnd`; under the
  flag `kStSave` only extends the session's `saved` to the history's end (the entries are already in its
  pages; the pages exist because the step path allocated them — a page missing there is an invariant
  violation and dies, `state: session %d saves to %d without its page %d`), and `kStSnapEnd` copies the
  rings and carries as today. Nothing is written for the device row: it belongs to the slot, and the
  next session to take the slot rewrites it. So a slot change copies 6.1 MiB of window state, not the
  caches.
- **The tail page is written in place.** A page that no other session shares is written by the
  compressor's emit (the decode step's `attn::compressor_emit`, `attn.h:105-110`; the prefill chunk's
  `dsk::compressor_emit_p`, `prefill_comp.cu:203-216`) directly at its entries, exactly as the
  contiguous caches were written. Pages below the session's written length are full and never written
  again; the entry above `saved` in the tail page is written by the next group that completes there.

**Built.** The row's writer is `apply_ops` from `Cmd::live[]`, as §2.6 says; `use_slot` does not write
one. A command that frees a session still has its handle in `live[]`, so that slot's row is written
all-zero (`write_slot_row` finds no pages) and the freed session is registered in `SlotSessions` again.
Benign: the server's `slot_[]` is 0 there, handles are never reused, and the next take of the slot
overwrites the registration.

### 2.7 Forks: copy-on-write of the partial tail page

A fork's session starts from its base's state at a restart point `q` (`serve.cpp:953-1012`). Under the
flag:

- the fork takes its own slot, and that slot's row takes the base's pages for every page below
  `q / 4096` (part of the 2,056-byte row write of §2.6, from the fork's host list, queued on the model's
  stream after the base's queued work);
- if `q` is not page-aligned, the page holding `q` is *shared and partial*: the fork allocates a new
  page, copies the base's entries `[page_start, saved)` of all four sources into it (up to 3.5 MiB, one
  copy, once), and puts the copy in its row. Entries from `q`'s group on are then written in place in
  the copy;
- full pages (below the tail) are never written by either session again, so they are shared read-only;
  the base and the fork write disjoint entries even where they share a tail page (the base appends above
  its `saved`, the fork from `q` on, and `q ≤ saved` at the fork). The copy is queued on the model's
  stream, after the base's queued steps, so it cannot observe a torn entry the fork would read: the fork
  rewrites every entry from `q / ratio` on;
- the fork's snapshots are copied as today (`state_copy_snaps`, `state.cu:764-781`); a fork of a fork
  shares the same full pages.

`state_drop(h, from)` (`state.cu:711-750`) on the live session frees the pages above `from` from its host
list and puts the zero page in the corresponding row entries (the session keeps its slot, so the row is
the one to fix); the entries above `from` in the tail page are abandoned (never read: §3).

**Built.** The COW lives in `state_copy_snaps` (`state.cu:787-812`), not `state_load`: `kStCopySnaps` is
the one state operation a fork's command always carries with both handles, while `kStLoad` is issued
only when the fork does not take its base's slot. The shared full pages are *reference-counted*
(`SharedKv::refs`, `hold`/`release` in `pages.h`), so a page still held by one session is never handed
out again and the order the sessions are freed in does not matter. The COW's page comes from
`store_alloc` (the store's LRU eviction); if the pool has nothing to evict, `state_copy_snaps` dies — a
fork cannot start without its tail page, and the state operations have no error path (§2.5's text
covers the loops, not a fork).

### 2.8 The disk tier and the copies

Unchanged in format, by construction: a page is the store's page, so

- a session's **copy** in the file is the same bytes at the same offsets in the same file format
  (`spill.h` `SpillFile`, `state.cu` `sum_block`/`state_copy`), and its checksums are taken over whole
  rounded blocks as before; one small index file per session per rank (`<base>.d/<h>.<seq>.c`,
  `state.cu` `copy_index`/`read_copy`) names those blocks and their checksums, and the server's history
  file (`server.<tag>.d/<id>.<h>.<seq>.s`) names the same copy `seq`;
- reading a copy back (`state_unspill`) allocates *new* pages from the pool and rewrites the session's
  **host list** for them; a spilled session is never live (2.5), so no device row is touched — when it
  next takes a slot, §2.6 writes the row from the list;
- a copy's `page_bytes`/`snap_bytes`/`end_bytes` are the same as the kept index's were, and the recovery
  checks are the ones that index had (magic, run id, rank, the layout, the file's size, the body's hash,
  the offsets), with `state_recover_scan`/`state_recover_take` in place of `state_kept_check`/`state_disk`;
- **the flag is part of the run id only when it is on.** `run_id` (`main.cpp`) hashes the
  binary, the weight files and the options array `o[]`, which has no shared-KV field; T1 adds one line
  before the return — `if (a.shared_kv) h = keepfile::fnv("shared-kv", 9, h);` — so a flag-off run's id
  is bit for bit what it is today (its kept files stay readable, and a flag-off build's run id is
  unchanged by this work), and a flag-on run's id differs: a flag-on run never reads a flag-off file
  (whose sessions' caches are not pool pages) and a flag-off run never reads a flag-on one. Both
  directions start clean, which is the rule for a changed build (`main.cpp` `state_recover_begin`);
- a page's bytes above the session's written length are never read by a kernel (§3), but they are copied
  to disk and checked on the way back: the round trip is byte-exact whatever they hold.

The copies change *when* blocks reach the file, never what they hold: a finished turn's session is
written through (`state_copy`) instead of only at a planned stop, so a `kill -9` loses at most the turn
in flight. A spill of a session with no copy of what it holds now frees it, as before.

### 2.9 Startup sizing, `--slots`, `--big-slots`, refusals

- **`--shared-kv`** (a flag, default off) turns all of it on. Nothing else changes by default.
- **The pool's size.** With the flag off, `main.cpp:735-741` sizes the store from the smallest rank's
  free device memory less `--session-margin-gib` and its available host memory less the margin and the
  Engram row cache, at most `--session-gib` (12 GiB by default). With the flag on, the per-slot caches
  are not allocated, so the same measurement sees `slots_bytes` more free memory (895.4 MiB a big slot,
  116.7 a small, at 1M), and `--session-gib` caps the *pool*:

  ```
  cap  = --session-gib given ? --session-gib : 12 GiB + slots_bytes
  pool = min(offer, cap)                      # offer as today, measured after the fixed state
  ```

  so the same command line keeps the same total KV memory and now shares it, while an explicit
  `--session-gib` (a small one for a test, a large one for a long-context service) means what it says.
  At the service's flags (`--slots 16 --big-slots 4 --max-ctx 1048576`, no `--session-gib`):
  12 + 4.87 = 16.87 GiB → 4,935 pages ≈ 20,213,760 positions (the handoff's ~17 GiB / ~4,800 pages /
  ~19.6M positions; it read the running service's logs, this is recomputed from the source, where a big
  slot is 895.4 MiB, not 939). Rank 0's log line, after the existing `session store:` line:

  ```
  shared KV: 16.87 GiB on every rank (at most 12.00 + the slots' 4.87; 4935 pages of 3.5 MiB, 20213760
  positions; 16 slot rows of 257, and one 3.5 MiB zero page)
  ```
- **Outside `--mode serve`** there is no store (`main.cpp:735`), and the golden and bench modes use the
  live slot's caches: `main.cpp` reserves the pool in every mode when the flag is on, at the same point
  (after the weights are loaded and `prefill_init` has run, step 5-6), sized for the run —
  `pages = max_ctx / 4096 + 8` (24 pages, 84 MiB, at the default 64k; 264 pages, 924 MiB, at 1M) — and
  `--session-gib` is ignored. So `--mode tf/decode/bench --big --shared-kv` work as they do today.
- **`--slots N`** keeps its meaning: N live sessions, each with its own slot table row, rings, carries and
  drafter windows, N ≤ `kMaxSlots`. Under the flag every slot can hold `--max-ctx` positions
  (`slot_ctx(s) = max_ctx`), so `fits()` (`serve.cpp:1189-1191`) is always true and the server's small
  slot / big slot logic (`serve.cpp:953-958`) never triggers.
- **`--big-slots B`** (B > 0) is refused: the units exist so that one slot's caches can be a multiple of
  another's, and the pool makes a slot's size independent of its reservation.
- **Refused combinations, exact text** (checked in `main.cpp` after the existing validation, before the
  model is built):

  | command line | text |
  |---|---|
  | `--shared-kv --big-slots B` | `--shared-kv with --big-slots %d: every slot holds the whole context` |
  | `--shared-kv --session-gib 0` | `--shared-kv with --session-gib 0: the pool holds the sessions' state` |
  | `--shared-kv --max-ctx %d`, `%d % 4096` | `--shared-kv with --max-ctx %d: a multiple of %d` (4096) |
  | `--shared-kv --slots %d` with `%d > 16` | the existing `--slots %d --slot %d: 1 to %d slots, a below their number` |
  | `--shared-kv` in a build without the kernels' page support | `--shared-kv: this build does not read pages yet` (removed by T4, when the flag is whole, §6) |

**Built**, with the two differences the code makes, both in the cap's arithmetic (`pages.h`,
`slot_caches_bytes`/`shared_kv_cap`, `main.cpp`):

- the flag's own blocks are inside the cap: `pool = min(offer, cap + shared_kv_overhead(slots, max_ctx))`
  (3,735,552 B at 1M and 16 slots), so the sessions' pages are the cap itself and a test's small
  `--session-gib` still holds the pages it names;
- the reservation the cap replaces is the code's, per unit: a slot with no `--big-slots` (what the flag
  allows) covers `--max-ctx` in one unit, 938,992,640 B = 895.4 MiB at 1M, so at `--slots 16 --max-ctx
  1048576` the cap is 12 + 16 × 895.4 MiB ≈ 26.0 GiB → 7,604 pages, 31,145,984 positions, and rank 0
  logs `shared KV: 26.00 GiB on every rank (at most 12.00 + the slots' 13.99; 7604 pages of 3.5 MiB,
  31145984 positions; 16 slot rows of 257, and one 3.5 MiB zero page)`. The example above (4.87 GiB,
  16.87 GiB, 4,935 pages) is the service's *flag-off* layout with `--big-slots 4`: 4 × 8 units of
  `max_ctx / 8` and 12 of one = 5.02 GiB by this arithmetic, where the design's per-slot table counted a
  big slot's rings and carries once rather than per unit. The last refusal row is gone (T4); a
  `--session-gib` at or below 0 gets that row's text, not only 0.

### 2.10 Prefill, the dense scan, the drafter

- **Prefill** (`model/prefill.cu:656-763`, `attention_p` at `:435-…`): the window is a linear buffer plus the
  ring (`ring_copy`, `:400-433`) and is unchanged; the compressed emit
  (`dsk::compressor_emit_p`, `prefill_comp.cu:203-216`) and the indexer's key writes go through
  `page_ent_m<kPaged>`. `index_p` (`model/prefill.cu:367-398`) scans through the same row. Nothing else
  changes: a chunk's rows are consecutive positions of one session, so `shared_kv_room(c.slot, c.pos0 +
  c.rows)` in `run_prefill` covers them, and a chunk's emit writes pages that the same call allocated.
- **The dense index scan.** `attn::index_dense` (`indexer.cu:984-1010`, kernel `:706-830`) loads each
  16-position tile's keys in `load_tile` (`indexer.cu:150-176`) with two 16-byte `__ldg`s and one scale
  word per lane per tile, and the prefill scan (`kernels/prefill_index.cu:88-100`) does the same with
  `__ldg` on `a.k + p * 64`. Through pages each of those loads becomes `page_ent<true>`: one extra
  8-byte table load per tile per lane (the tile's page entry is the same for 128 consecutive tiles, so
  it is an L1 hit), a shift, a mask and a multiply-add. The scan is MMA-bound, reads 4.5 MB of keys at
  layers 2/8/14 and 8.9 MB at layer 20 per step at 128k (`engine.md` §Long contexts), and the table
  costs 8 B per 4096 positions per lane: the keys' DRAM traffic does not change. **Estimated cost:
  +2–5 % on the eight index kernels (157–398 µs a step at 1k–128k, `engine.md` §Long contexts), i.e.
  under 0.02 ms a step at 1k and under 0.02 ms at 128k; to be measured** by
  `engine/build/attn/bench_index --pages` (new: the same cases with the keys in pool pages) before the
  task that changes the scan is accepted.
- **The drafter** (`dspark/`) needs no page work: its attention passes `attn::Compressed{}` and reads
  its own windows (`dspark.cu:1064,1242,1345`), which stay per slot (§2.2). What the flag changes for
  DSpark is only the main model's step it wraps: the spec step's rows' slots (`in.slot[r]`, which
  `spec.cu:1179,1229,1268` already fill), the pages its verify rows may write (`shared_kv_room` up to
  `pos_bound + 1 + kMaxSessRows`, called in `cmd.cpp`'s spec loops, §2.5) and the captured spec graphs,
  which read the tables through `in_d->slot` as every other step does (`spec.cu:779-1080`). Nothing in
  `spec.cu` or `dspark/` changes. `Options::slots`/the draft module's window count are unchanged.

**Built.** The instantiation is chosen by `pages.paged()` (a non-null `tab`), not by the process flag,
which is what the kernel tests need (they have no flag); with the flag off no `KvPages` is ever filled,
so the `<false>` instance is always the one launched. `model/prefill.cu` and `prefill.h` needed no
statement: the prefill path hands `kvs[...]` to the kernels, and `use_slot`'s page views (T1) are what
they read. `test_pattn`'s `pages` argument runs the cases that need no reference data
(`~/dsgb10-data/attntests` is absent on these boxes); the reference-data cases keep skipping as today.

## 3. Exactness

**The claim.** With `--shared-kv` on, the engine computes the same bits as with it off for the same
requests: the same tokens, logits and logprobs, hashed by the golden modes and compared by the serve
tests. There are no exceptions to this; the list below is of the mechanisms, then of what does differ
and why none of it is an output.

1. **Same arithmetic, same order.** No kernel's computation changes: the same fp32/bf16 operations, the
   same MMA tiles, the same split-K and reduction order, the same collectives in the same order on the
   same streams. `page_ent` (2.4) computes a different *address*; the value loaded or stored at that
   address is the same byte. In particular the dense scan's key loads (`indexer.cu:150-176`,
   `prefill_index.cu:88-100`), the sparse attention's compressed gathers (`sparse_attn.cu:158-175`), the
   prefill attention's (`prefill_attn.cu:100-112`) and the compressor's entry writes
   (`compressor.cuh:120-142`, `prefill_comp.cu:160-193`) keep their expressions, their lane mapping and
   their order.
2. **Same bytes at the same offsets.** A page's layout is the store's (`state.cu:34-42`), so entry `j` of
   kv source `k` sits at the same relative offset in a page as in a contiguous cache; a page's writes and
   reads touch the same bytes as before, and the tail page's entries are written in place (2.6).
3. **Same read set, and what a read past it can see.** Every kernel that reads a compressed entry or an
   index key is bounded by the position it computes, or masks what it reads; the table goes through each
   one. The masks are *selects* on the loaded value (the sort key or the probability is replaced, never
   multiplied by a garbage scale), so a stale or zero byte cannot reach an output.

   | reader | how far it reads | what it does past the session's written length |
   |---|---|---|
   | `indexer.cu:150-176` `load_tile`, dense (`blist == nullptr`) | `k.p0 = i0 + g < n ? lo + i0 + g : -1`, `n = (pos_last + 1) / ratio` of the group's session (`indexer.cu:35-40` `dense_range`, `:706-720`, `:771-772`) | nothing: the load is guarded by `p0 >= 0` (`:160-171`) |
   | `indexer.cu:150-176` `load_tile`, a block list (`blist`: the Reindex layers, on layer 20's pool) | block `b = blist[i0 >> 3]`'s eight entries; `b` comes from layer 20's pool selection, so `b * 8 < n20` | up to 7 entries past the length (the block holding the last position is chosen whole), inside the last allocated page (a block's span never crosses a page: 512 blocks of 8 a page); `score_tile_row` (`:210-215`) then *selects* each key to 0 for `p0 >= len`, so the value is discarded |
   | `prefill_index.cu:88-100` | `if (p0 < n)`, `n = (pos[rows - 1] + 1) / ratio` | nothing |
   | `sparse_attn.cu:158-172` (the compressed gather) | `kp = topk[r][…]`; the top-k's padding is -1 (`attn.h:145-148`) | nothing: `kp >= 0` selects the load and the scale |
   | `prefill_attn.cu:100-112` | `ok = kp >= 0`, `pa::cp16(…, ok)` | nothing: the copy is predicated |
   | the compressor's reads (`compressor.cuh:39-40`, `prefill_comp.cu:68-69`) | the carry, a 16-row ring by `position & 15`, never a page | n/a (per slot) |
   | the window readers (`prep.cu:85-86`, `sparse_attn.cu:132-148`, `prefill_attn.cu:85-92`) | the ring, `(p - base) & mask` | a reused slot's ring is stale, exactly as today |

   So the only read past a session's written length is the block path's, at most 7 entries, and its
   value cannot reach an output. Two things make such a read *safe* rather than merely harmless: every
   entry above a session's list holds the zero page (§2.3), and **the pool is zeroed once at reserve**
   (`state_reserve`, `state.cu:336-343`: one `cudaMemset`; today's store pool is not zeroed because its
   blocks are always written before they are read). A page a session has just been given therefore holds
   zeros, exactly like a fresh contiguous cache (`impl.h:31-39` `dalloc`); a page it *reuses* after
   another session freed it holds that session's entries — no worse than a reused *slot* today, whose
   contiguous caches are not cleared either, and the same masking is what makes both safe.
4. **The copies that disappear are copies of the same bytes.** `kStSave` (`state.cu:555-577`) copies the
   slot's caches into the store's pages; with the flag the entries are already in the pages, so the
   operation only extends `saved`. A fork's COW copy (2.7) copies the base's bytes; a disk round trip
   copies a page out and back and checksums it. None of them changes a value the next kernel reads.
5. **Everything outside the compressed caches is untouched**: the window rings, the compressor carries,
   the drafter's windows, the snapshots, the Engram path, the MoE, the head, the sampler, the graphs'
   structure and the step ring (`ops::StepIn` is byte-for-byte the struct it is today, §2.3).
6. **Ordering.** Every page allocation, table write and COW copy is a `cudaMemcpyAsync` or a host-side
   bookkeeping step queued on the model's stream between the same steps as today's copies (`state.cu:1-5`),
   so the device's view of the caches is the same at every point.

**What differs, and why it is not an output:**

| difference | why it cannot change a token, logit or hash |
|---|---|
| a page's bytes above the session's written length hold zeros (a page never used since reserve) or a previous session's entries (a reused page) | the only reader that goes past the length masks what it reads (3); the bytes are copied to disk and back as they are, so the checksum still matches; a reused slot's contiguous caches are stale in exactly the same way today |
| the flag's log lines (`shared KV: …`, the store's line, the eviction lines) | logs, and no token id, prompt or response is in any of them |
| `kStSave`'s `cudaMemcpyAsync`s are gone; page allocation, the slot rows' writes and COW copies are added | timing only |
| a flag-on run whose pool runs out fails a request (2.5) that a flag-off run would have served out of its 12 GiB store plus its per-slot reservation | capacity, not bits; the failure is an error the front end reports |
| the spec gate's measured ms per step (`spec_ms`, `cmd.h:120-123`) differ with the speed, so a flag-on run verifies a different number of drafts and takes a different number of steps | speculative decoding is exact: a row computes what it computes alone, a rejected draft's KV is written again with the same values, and the sampler's noise is a function of the seed and the position (`engine.md` §Scheduling, "Exact"), so the tokens and logprobs are the same; `golden/README.md`'s `fp8`/`bf16` rows (`k1` with drafts on and off, the same hash) exist to pin exactly this |
| `--slots`/`--big-slots` lay the state out differently (5.9 MiB a slot instead of up to 895.4) | memory layout only; the slot table's values are unchanged |

**The gates.** All of these must hold before the flag is offered to the owner (the runs of §6 in the
orchestrator's window form; the expected hashes are the official checkpoint's, which is what the fleet
runs):

```
golden/refhash.sh -- --shared-kv
# tf c7605e0b6ba8d9d2 · k1 = nol2pf = fp8 = bf16 7941647b162df1a8 · temp 02ef3e0858d7eb45
# ced16 8790ee9388f4abd8 · ced28 de99bfce6ea96ba8
# (EXPECT=official, golden/README.md "Checking a build"; on dealignai's variant, in parentheses:
#  tf 842278b28d87dad0, k1/nol2pf/fp8/bf16 febc7dc54ee40a39, temp a8f7a08f8cd1c5e8, ced16 81f772ab79a950e4,
#  ced28 8f08a153d415c5c8, exact28 87f42fa9f94f13de)
python3 engine/server/tests/serve_test.py --ced --socket ~/test.sock   # 17 of 17, --shared-kv on the server
python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
    --nonce N --out A.json                       # the served gate, on a --shared-kv lane
python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
    --nonce N --out B.json                       # the same lane, the flag off
python3 bench/share_gate.py compare B.json A.json    # every case id equal (temperature 0, sha256)
python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
    --nonce N --cases fork --out F.json          # a freshly started engine: the fork case alone
python3 bench/share_gate.py compare A.json F.json    # its fork replies equal the full run's
python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
    --nonce N --cases alone --out L.json         # restarted again: each concurrent prompt alone, cold
python3 bench/share_gate.py compare A.json L.json    # every concurrent reply equals its prompt's alone one
```

`refhash.sh` passes its `--` arguments to every run (`golden/refhash.sh:24-28`, `:139-146`), so one
command covers all runs; `serve_test.py --ced`'s checks are the engine's own exact-reuse, CED, fork and
snapshot comparisons against cold prefills (`engine/server/tests/serve_test.py:1-12`); the share gate
compares served replies across a fresh session, a resumed one, concurrent levels, forks and 16 grown
sessions (`bench/share_gate.py`, the sibling task's tool — run as it is, with its new `--cases`). Both
gate runs use the **same `--nonce`** so the two reports' prompts are identical; the fork case and the
concurrent replies are the ones that must equal a *freshly started* engine's, so the fork case and the
`alone` case each run against a just-started engine and are compared with the full run's. An alone
request sent in the same engine after its concurrent twin would resume the twin's state, a regenerate
path that matches the cold one to numerics, not bits (`engine.md` "Serving"); for the same reason a
`resumed` warm reply that differs from its cold one is a note in the report, not a failure.

**The gate runs carry no `DSGB10_OEA`** (measured 2026-11-14): under the lane's production arguments
(`DSGB10_OEA=4,12,9`, the fork's lossy, batch-aware expert re-routing, `engine/kernels/moe.cu:1452`) a
request's reply depends on what else is in its step from 9 rows on, so "concurrent = alone" does not
hold there with or without `--shared-kv`. The exactness gates therefore run with the production
arguments minus `DSGB10_OEA`.

## 4. Flag-off invariance

`shared_kv` is a process-wide constant (the flag). Nothing about it is per session, per row or per step,
so every changed kernel can be split at compile time:

| file | mechanism |
|---|---|
| `attn/attn.h` | `Compressed`/`IndexKeys` gain one `KvPages pages;` field (null `tab` by default) and the accessors of 2.3; every entry point's signature is unchanged and picks the kernel instance by `pages.paged()` |
| `attn/attn_common.cuh` | new helpers only (`pages_at`, `page_ent`); `win_at`/`carry_at`/`same_session`/`session_first`/`check_slots` are unchanged; `ckv_at`/`ik_at` add `pages.tab += slot[r] * pages.stride`, a no-op with a null table (a uniform branch outside the entry loop) |
| `attn/prep.cu`, `compressor.cu`, `sparse_attn.cu`, `indexer.cu` | each kernel that touches a cache is templated on `bool kPaged` (`indexer.cu`'s already carry `bool kSlots`); the false instantiation's address arithmetic is the pre-change expression `base + e * bytes`, with no branch and no table load |
| `kernels/prefill_attn.cu`, `prefill_comp.cu`, `prefill_index.cu` | the same |
| `model/*.cu`, `server/*.cpp` | host-side: `shared_kv` false means no pool, no table, no `ensure`, no bookkeeping; every new statement is inside `if (shared_kv)` or a new function, and the flag-off paths are the current code |

There is **no runtime branch inside a hot loop**: the only per-entry branch is `if constexpr (!kPaged)`
inside `page_ent`, removed at compile time, and the host's `pages.paged()` chooses the instantiation at
launch (and therefore what a captured graph contains). The cost is a second instantiation of each
cache-touching kernel: compile time, and the flag-off binary's `.text` grows while its executed code
does not change.

The flag-off *hashes* are checked too, not just argued: `golden/refhash.sh` without `--shared-kv` must
still print the table of §3 (and `--no-l2pf`, `--spec`, `--draft-bf16` their equalities), and the flag-off
kernel tests (`slot_tests`, `fused_tests`, `test_pattn`, `test_kernels`) must pass unchanged.

## 5. Change list per file

| file | what changes | task |
|---|---|---|
| `engine/model/pages.h`, `pages.cpp` (new) | the host side, no CUDA: `Pool` moved here from `state.h`, the page geometry (`kPagePos`, `kPageBytes`, `kPageBlock`, `kPageShift`), `layout_of(ratio)` (`page_off` as a table), `SharedKv`: `room(slot, upto)`, `list(slot, pages)`, `add(slot, page)`, `drop(slot, from)`, the zero page, the LRU victim choice | T1 |
| `engine/model/tests/page_tests.cpp` (new), `engine/model/tests/CMakeLists.txt` | host-only unit tests (pure C++, `g++ -std=c++20`): allocation, coalescing, best fit, page indices, the slot rows, the zero page, drop, the full-pool outcome | T1 |
| `engine/attn/pages.cuh` (new) | `KvPages`, `pages_at`, `page_ent`, `page_ent_m` (2.3) | T1 |
| `engine/attn/attn.h` | the `pages` field of `Compressed`/`IndexKeys` and their accessors (2.3); every entry point's signature unchanged | T1 |
| `engine/model/state.h`, `state.cu` | `Pool`/geometry move to `pages.h`; pages as pool blocks of `kPageBlock`; the zero page and the one-time memset in `state_reserve`; the slot rows' writes (`state_slot_row`); `state_save`/`state_load`/`state_drop`/`state_free` as bookkeeping over the pages; the store's log lines | T1, T4 |
| `engine/model/impl.h`, `model.h` | `shared_kv`, `slot_tab`, `shared_kv_room(...)`, `state_slot_row(s, h)`, `shared_kv_capacity/bytes` (2.5) | T1 |
| `engine/model/model.cu` | the flag's plumbing: `use_slot`'s row, the step views (2.3), and the temporary refusal of §2.9 | T1, T4 |
| `engine/server/main.cpp` | `--shared-kv` parsing, the refusals of §2.9, the pool's sizing, `run_id`'s one line (§2.8) and the log line | T1, T5 |
| `engine/attn/attn_common.cuh` | `pages_at`/`page_ent`, `ckv_at`/`ik_at` | T2 |
| `engine/attn/prep.cu`, `compressor.cu`, `sparse_attn.cu`, `indexer.cu` | the decode path's reads and writes through the pages | T2 |
| `engine/attn/tests/slot_tests.cu`, `fused_tests.cu`, `bench_index.cu` | `--pages` (new): the same checks with the caches in pool pages, and the scan's cost | T2 |
| `engine/kernels/prefill_attn.cu`, `prefill_comp.cu`, `prefill_index.cu`, `prefill_attn.h` | the prefill path's reads and writes through the pages | T3 |
| `engine/model/prefill.cu`, `prefill.h` | the prefill path's views (`use_slot`'s row) | T3 |
| `engine/kernels/tests/test_pattn.cu` | a third argument `pages` (new): the same cases with the caches in pool pages | T3 |
| `engine/server/cmd.cpp` | `shared_kv_room` in the loops and `run_prefill`, `kEndPool` (`cmd.h:137`), the slot rows' write in `apply_ops` | T4 |
| `engine/server/serve.cpp` | `kEndPool` → `kError` and its text, `forget_evicted()` after the loops, eviction skips live sessions, the budget (`budget_`, `serve.cpp:465,1553`) under the flag, the pool's log line | T4 |
| `engine/model/state.cu`, `state.h` | the fork's COW (`state_load`), `state_unspill`'s host-list rewrite, the live-session refusal in `state_spill` | T4 |
| `engine/server/tests/pool_test.py` (new) | the full pool's defined outcome (2.5) | T4 |
| `engine/server/tests/serve_test.py`, `snap_test.py`, `slice_test.py`, `batch_test.py`, `disk_test.py`, `kept_test.py` | flag-on runs | T4, T5 |
| `engine/model/spec.cu` | nothing: the spec fills already carry the row's slot, and the spec graphs read `in_d->slot` (2.10) | — |
| `engine/dspark/dspark.cu`, `dspark.h` | nothing: the draft pass passes `attn::Compressed{}` (2.10) | — |
| `engine/server/main.cpp`, `serve.cpp` | the final sizing rule, the refusals, the log lines | T5 |
| `docs/engine.md`, `docs/shared-kv.md`, `README.md`, `golden/README.md` | the flag's documentation, the flag-on rows of the hash table | T5 |
| `engine/server/tests/soak.py`, `revisit.py` | flag-on runs | T5 |
| `bench/share_gate.py` | nothing: run as it is against a `--shared-kv` lane (its own task), with its new `--cases` | T5 (checks only) |

## 6. Task cut

Five tasks. T1 lands the pool, the pages, the tables and the store's page bookkeeping and leaves the flag
refused (§2.9); T2 (the decode kernels), T3 (the prefill kernels) and T4 (the store's fork and disk work,
the server, the loops) then run at the same time — their files are disjoint — and T5 closes the flag's
sizing and documentation. The flag is *correct* only when T2, T3 and T4 are all in (every kernel and both
loop paths must read pages), so the flag-on end-to-end checks belong to T4 and T5; T1's, T2's and T3's own
GPU checks are the kernel tests — which build their own pool and pages and need no flag — plus the
flag-off runs. Every task is ≤ ~600 changed lines and builds and runs on its own. `engine/build/` is the
CMake build directory (`engine/CMakeLists.txt` puts each subdirectory's binaries under
`engine/build/<dir>/`: `engine/build/attn/slot_tests`, `engine/build/kernels/test_pattn`, …).

### T1 — the pool, the pages, the tables and the store's page bookkeeping

**Files owned.** `engine/model/pages.h`, `engine/model/pages.cpp` (new), `engine/model/tests/page_tests.cpp`
(new), `engine/model/tests/CMakeLists.txt`, `engine/attn/pages.cuh` (new), `engine/attn/attn.h`,
`engine/model/state.h`, `engine/model/state.cu`, `engine/model/impl.h`, `engine/model/model.h`,
`engine/model/model.cu`, `engine/server/main.cpp`, `engine/model/CMakeLists.txt`.
**Depends on.** Nothing.
**Estimate.** ~600 changed lines.
**Deliverable.** `--shared-kv` parses, the pool, the slot table and the zero page are reserved at the point
the store's pool is reserved today, the pool is zeroed once, a slot's row is written from its session's
page list, `shared_kv_room` allocates pages, `state_save`/`state_load`/`state_drop`/`state_free` are
bookkeeping over the pages, `run_id` hashes the flag when it is on (§2.8), and the flag is refused with
the exact text of §2.9 until T4 removes that one line. With the flag off nothing at all changes.
**Host checks.**
```
g++ -std=c++20 -O1 -I engine engine/model/pages.cpp engine/model/tests/page_tests.cpp -o /tmp/page_tests && /tmp/page_tests
# page_tests: NN checks, 0 failed   (best fit, coalescing, page indices, the slot rows, the zero page, drop, no room)
cuda-compile-check.sh        # from the worktree root: CUDA COMPILE OK
engine/build/model/spec_batch_test                            # host-only, no GPU: PASSED
```
**GPU checks.** Flag off, every one exactly as before:
```
engine/build/attn/slot_tests
engine/build/attn/fused_tests
golden/refhash.sh k1                                          # EXPECT=official: 7941647b162df1a8
```
**Acceptance.** The host tests pass; the flag-off GPU checks pass unchanged; `--shared-kv` is refused with
`--shared-kv: this build does not read pages yet`; `--shared-kv --big-slots 1` and `--shared-kv
--session-gib 0` die with their texts; no allocation happens after startup; the flag-off `run_id` is
unchanged.

### T2 — the decode kernels read the caches through the pages

**Files owned.** `engine/attn/attn_common.cuh`, `engine/attn/prep.cu`, `engine/attn/compressor.cu`,
`engine/attn/sparse_attn.cu`, `engine/attn/indexer.cu`, `engine/attn/tests/slot_tests.cu`,
`engine/attn/tests/fused_tests.cu`, `engine/attn/tests/bench_index.cu`.
**Depends on.** T1.
**Estimate.** ~260 changed lines.
**Deliverable.** `pages_at`/`page_ent` and the decode kernels' reads and writes through them (2.3, 2.10);
the `--pages` modes of the three tests, which build a pool and a slot table themselves and check the same
bit-for-bit equalities as the contiguous path. The refusal line stays until T4.
**Host checks.**
```
cuda-compile-check.sh        # CUDA COMPILE OK
```
**GPU checks.** Flag off, then the new page modes (the argument `--pages` is new in all three):
```
engine/build/attn/slot_tests
engine/build/attn/slot_tests --pages
engine/build/attn/fused_tests
engine/build/attn/fused_tests --pages
engine/build/attn/bench_index --iters 3 --kinds 1
engine/build/attn/bench_index --pages --iters 3 --kinds 1     # within 5 % of the line above
golden/refhash.sh k1                                          # flag off: 7941647b162df1a8
```
**Acceptance.** `slot_tests --pages` and `fused_tests --pages` pass bit for bit (every group), the flag-off
runs are unchanged, and `bench_index --pages` is within 5 % of the contiguous baseline.

### T3 — the prefill kernels read and write the caches through the pages

**Files owned.** `engine/kernels/prefill_attn.cu`, `engine/kernels/prefill_comp.cu`,
`engine/kernels/prefill_index.cu`, `engine/kernels/prefill_attn.h`, `engine/model/prefill.cu`,
`engine/model/prefill.h`, `engine/kernels/tests/test_pattn.cu`.
**Depends on.** T1.
**Estimate.** ~230 changed lines.
**Deliverable.** The prefill attention's compressed reads, the compressor's entry writes and the prefill
index scan through the pages (2.10); `test_pattn`'s third argument `pages` runs the same cases with the
caches in pool pages.
**Host checks.**
```
cuda-compile-check.sh        # CUDA COMPILE OK
```
**GPU checks.** Flag off, then the new `pages` argument:
```
engine/build/kernels/test_pattn '' index
engine/build/kernels/test_pattn '' index pages
engine/build/kernels/test_pattn '' attn pages
engine/build/kernels/test_pattn '' comp pages
golden/refhash.sh k1                                          # flag off: 7941647b162df1a8
```
**Acceptance.** Every `pages` case passes bit for bit, and the flag-off prefill runs are unchanged.

### T4 — the store's fork and disk work, the server, the loops, and the flag turned on

**Files owned.** `engine/server/cmd.cpp`, `engine/server/cmd.h`, `engine/server/serve.cpp`,
`engine/model/state.cu`, `engine/model/state.h`, `engine/model/model.cu` (the refusal's removal only),
`engine/server/tests/pool_test.py` (new), `engine/server/tests/serve_test.py`, `snap_test.py`,
`slice_test.py`, `batch_test.py`, `disk_test.py`, `kept_test.py`.
**Depends on.** T1, T2, T3 (its flag-on checks are only meaningful with both kernel tasks merged).
**Estimate.** ~420 changed lines.
**Deliverable.** The fork's COW (`state_load`) and the slot row's page list (2.7), the disk tier's and kept
sessions' round trip through pool pages (2.8), `unspill`'s host-list rewrite, the live-session refusal in
`state_spill`, `shared_kv_room` in `cmd.cpp`'s loops and `run_prefill` with `kEndPool` and the DONE
mapping (2.5), `forget_evicted()` after the loops, eviction skipping live sessions, the server's budget
under the flag, `pool_test.py`, and the removal of T1's refusal line — the flag is whole here.
**Host checks.**
```
cuda-compile-check.sh        # CUDA COMPILE OK
python3 -m py_compile engine/server/tests/pool_test.py engine/server/tests/serve_test.py
```
**GPU checks.** The flag on, end to end (the official checkpoint's hashes):
```
golden/refhash.sh -- --shared-kv
# tf c7605e0b6ba8d9d2 · k1 = nol2pf = fp8 = bf16 7941647b162df1a8 · temp 02ef3e0858d7eb45
# ced16 8790ee9388f4abd8 · ced28 de99bfce6ea96ba8
```
```
ENGINE: --shared-kv --slots 4 --session-gib 4
TEST: python3 engine/server/tests/serve_test.py --ced --tokens ~/golden/long-tokens/tokens.npy
TEST: python3 engine/server/tests/snap_test.py --ced --tokens ~/golden/synth/tokens.npy
TEST: python3 engine/server/tests/slice_test.py --tokens ~/golden/synth/tokens.npy
TEST: python3 engine/server/tests/batch_test.py --slots 4 --tokens ~/golden/synth/tokens.npy
# disk_test: two engines, both with the flag, then the same pair with it off and a --compare
ENGINE: --shared-kv --disk-gib 0                  # (a) the store keeps everything
ENGINE: --shared-kv --session-gib 0.25            # (b) the store spills to the disk tier
TEST: python3 engine/server/tests/disk_test.py --tokens ~/golden/synth/tokens.npy --out A.json    # against (a)
TEST: python3 engine/server/tests/disk_test.py --tokens ~/golden/synth/tokens.npy --out B.json    # against (b)
TEST: python3 engine/server/tests/disk_test.py --compare A.json B.json
#   and the same two servers without --shared-kv, their reports compared with the flag-on ones
# kept_test: a stop-start pair, the same engine args both times, then the same pair with the flag off
ENGINE: --shared-kv --slots 4 --session-gib 4 --disk-gib 8
TEST: python3 engine/server/tests/kept_test.py --tokens ~/golden/synth/tokens.npy --phase before \
      --state ~/test-spill/kept --out B.json      # then SIGTERM rank 0 and start the same engine again
TEST: python3 engine/server/tests/kept_test.py --phase after --state ~/test-spill/kept --out A.json
TEST: python3 engine/server/tests/kept_test.py --compare REF.json A.json    # REF: the flag-off pair's after-run
# the full pool: a prompt past it ends with kError and the text of 2.5, and the engine stays up
ENGINE: --shared-kv --session-gib 0.02 --slots 2 --disk-gib 0
TEST: python3 engine/server/tests/pool_test.py
```
**Acceptance.** Every flag-on hash equals `golden/README.md`'s official table; `serve_test.py --ced` 17 of
17 and `snap_test.py --ced` 19 of 19; `disk_test.py --compare` and `kept_test.py --compare` equal bit for
bit; `slice_test.py` and `batch_test.py` pass; `pool_test.py` shows the defined full-pool error, the
tokens already sent stand and the engine stays up; the flag-off runs unchanged.

### T5 — sizing, refusals, documentation, and the gates

**Files owned.** `engine/server/main.cpp`, `engine/server/serve.cpp`, `docs/engine.md`,
`docs/shared-kv.md`, `README.md`, `golden/README.md`, `engine/server/tests/soak.py`,
`engine/server/tests/revisit.py`, `engine/server/tests/serve_test.py`, `disk_test.py`, `kept_test.py`.
**Depends on.** T1–T4.
**Estimate.** ~200 changed lines.
**Deliverable.** The final sizing rule of §2.9 (the `--session-gib` default under the flag, the pool's log
line), the refusals with their exact texts, the flag's sections in `docs/engine.md` (§Memory per rank,
§Serving, §Scheduling's slot paragraphs), the flag-on rows of `golden/README.md`'s table, and the flag-on
soak.
**Host checks.**
```
cuda-compile-check.sh        # CUDA COMPILE OK
bash -n engine/server/run4.sh engine/server/service.sh        # if either changed
```
**GPU checks.** The gates of §3, in full:
```
golden/refhash.sh -- --shared-kv                              # the official table, every run
golden/refhash.sh                                             # flag off: unchanged
ENGINE: --shared-kv --max-ctx 1048576 --slots 16
TEST: python3 engine/server/tests/serve_test.py --ced --tokens ~/golden/long-tokens/tokens.npy
# the served gate: a test dsfront at 127.0.0.1:18890 (never :8888), the same --nonce on both runs
TEST: python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
      --nonce N --out A.json                                  # the --shared-kv lane
TEST: python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
      --nonce N --out B.json                                  # the same lane, the flag off
TEST: python3 bench/share_gate.py compare B.json A.json
TEST: python3 bench/share_gate.py run --base http://127.0.0.1:18890/v1 --key-file ~/ds4-kv-share/test-key \
      --nonce N --cases fork --out F.json                     # a freshly started engine, the fork case alone
TEST: python3 bench/share_gate.py compare A.json F.json
TEST: python3 engine/server/tests/soak.py --url http://127.0.0.1:18890 --key-file ~/ds4-kv-share/test-key
TEST: python3 engine/server/tests/revisit.py --key-file ~/ds4-kv-share/test-key
```
**Acceptance.** Every gate passes; the pool's log line matches the rule; the flag-off path is unchanged;
`docs/engine.md` and `golden/README.md` describe the flag as it behaves.

## 7. Memory and speed

**Memory, at the service's flags** (`--slots 16 --big-slots 4 --max-ctx 1048576 --session-gib 12`):

| | today | `--shared-kv` |
|---|---|---|
| per-slot caches, 16 slots | 4.87 GiB (4 × 895.4 MiB + 12 × 116.7) | 95 MiB (16 × 5.9 MiB) |
| store / pool | 12.00 GiB | 16.87 GiB = 4,935 pages × 3.5 MiB = 20,213,760 positions |
| KV memory in all | 16.87 GiB | 16.87 GiB, all of it shareable |
| 16 live sessions at 1M | impossible (4.87 GiB reserved for 4 big + 12 small) | 4,096 pages = 14.0 GiB, leaving 839 pages = 3.4M positions for other sessions |
| the slot table + zero page | — | 32.9 KB (16 slots × 257 × 8 B) + one 3.5 MiB zero page at 1M; 2.2 KB + 3.5 MiB at 64k |

The resident footprint at startup is the same to within a few MiB (the pool grows by exactly what the
slots released); what changes is that a short chat no longer holds 895.4 MiB of reserved cache and a long
one is no longer capped by its slot's size. Beyond the pool, the disk tier is unchanged: 32 GiB per rank,
about 170 thinned 128k sessions or 110 of 250k (`engine.md` §Serving).

**Built** (T5): the projection above is off where the code differs from the design's per-slot table.
`--big-slots` is refused under the flag, so the flag's own command line has no big slots: at `--slots 16
--max-ctx 1048576` the cap is 12 GiB + 16 × 895.4 MiB ≈ 26.0 GiB (7,604 pages, 31,145,984 positions),
and the same command line with `--big-slots 4` reserves 5.02 GiB for the slots, not 4.87 (a big slot is
8 units, so its rings and carries are 8 too). §2.9's **Built** note has the arithmetic and the numbers
the log line prints.

**Speed.**

- **A decode step**: the indexer's dense scan pays the table indirection (2.10): +2–5 % of the eight index
  kernels, which are 157 µs of a step at 1k and 398 µs at 128k (`engine.md` §Long contexts) — under
  0.02 ms of a 19–39 ms step. The compressed attention's gather adds two L1-resident loads per 16-key
  block per lane. Everything else is unchanged. Expected: within 0.5 % of a step, to be measured by
  `engine/build/attn/bench_step` (the same cases before and after) in T2.
- **A turn that resumes a stored session** (a slice turn, a revisit, a fork of a stored base): no cache
  copy at all — the slot's row is written and the pages are already the store's — where today
  `state_load` copies up to 890 MiB (`engine.md` §Scheduling: "its prompt and the tokens sent so far,
  restarted at its end snapshot (one row, ~25 ms, the state's load included)"). A 1M-position session's
  resume drops from hundreds of ms to a table write.
- **Leaving a slot** (`persist_slot`): 6.1 MiB of window state instead of the caches too.
- **Spilling a live session** (a planned stop, or an eviction of a session whose slot was taken): the
  save before the spill disappears.
- **Prefill**: unchanged except the same scan indirection (the index scan is 5–10 % of a chunk) and the
  emit's stores (the same count, one page's worth of addresses more).
- **The allocator**: best fit over 3.5 MiB blocks with coalescing (`state.cu:87-141`), the same code as
  today, over at most a few thousand blocks; a page is allocated or freed whole, once per 4096 positions
  per session, on the host.

## 8. Questions only the owner can answer

1. **Should the service run `--shared-kv` once the gates pass?** The note's gates are the golden hashes,
   the serve tests, the share gate and a soak; a deploy is the owner's call, and the service's
   `DSGB10_ENGINE_ARGS` is the switch. *Recommendation*: land the flag, run the soak on a dev lane, then
   add it to the service's arguments in a deploy of its own; the flag-off path stays for A/B.
2. **The dense indexer's page indirection**: if T2 measures worse than +5 % on the dense scan, do we
   accept it or move the index keys to their own page layout (which would change the store's page format,
   the disk format and the copies' index files)? *Recommendation*: accept up to +5 %; a separate key layout is a
   larger change and the formats' stability is worth more than 0.02 ms a step.
3. **Kept sessions across the flag.** T1 hashes the flag into `run_id` only when it is on (§2.8), so a
   flag-off run's id and its kept files are untouched, and the two directions never read each other's
   files (each starts clean, as after any build change). The alternative — the flag always in the id —
   would invalidate every flag-off kept file on the deploy that adds the flag. *Recommendation*: as
   specified; `service.sh forget` stays the way to drop kept sessions.
4. **`--session-gib`'s default under the flag** (12 GiB plus what `--slots` would have reserved, §2.9) —
   or should the flag default to the whole free-memory offer? *Recommendation*: keep the cap: it keeps
   the store's eviction behavior predictable and a small `--session-gib` a usable test knob.
5. **`--slots 16` with a pool that cannot hold 16 sessions at `--max-ctx`**: the server admits up to
   `--slots` requests and a session that outgrows the pool ends with the error of §2.5. *Recommendation*:
   as it is: `--slots` is the live-session limit, the pool is the capacity limit, and both are logged.
6. **Two things the note cannot settle from this source**, both about the checks rather than the design:
   the gate tool's `--cases` (T5's fork check assumes `bench/share_gate.py`, the sibling task's tool, has
   it by then) and the test windows' token paths (`~/golden/long-tokens/tokens.npy` for the 28,489-token
   sequence and `~/golden/synth/tokens.npy` for the 270,000-token one, per §6; the tests' own defaults
   are `/data/dsgb10/golden/...`). *Recommendation*: as written — T5's fork check falls back to a full
   `share_gate.py run` on a freshly started engine compared with the full run if `--cases` is missing, and
   the orchestrator's window layout is what §6 names.
