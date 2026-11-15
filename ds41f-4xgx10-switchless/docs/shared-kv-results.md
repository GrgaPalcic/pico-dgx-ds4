# Shared KV: the measured results

Every number below was measured on 2026-11-14, in the test windows W0–W11 and one production run, all on the
four DGX Sparks (`spark-0` is rank 0). Each table names the window and the time. The design is
`docs/shared-kv.md`; the plan and the task board are the plan's. Nothing here is a projection unless it says so.

## 1. What it is

`--shared-kv` (the engine flag `dsgb10_rank`, off by default) gives every rank one KV pool and moves the live
slots' compressed caches into it: a page is the store's own page (4096 positions of all four kv sources at the
store's offsets), a live slot reads and writes the store's pages through a slot table row in device memory (one
`uint64_t` page pointer an entry, 2,056 B a row at 1M), and taking a slot, leaving it or moving a session
between a slot and the store is bookkeeping — no cache is copied, and a fork shares its base's full pages and
copies only the partial tail page it will write (copy-on-write). With the flag off every code path, kernel
launch, memory layout, log line and output hash is byte-for-byte as before; with it on, `--slots` keeps its
meaning (the live-session limit), every slot can hold `--max-ctx` positions, `--big-slots` is refused, and the
pool is the capacity limit. A request that outgrows the pool ends with an error
(`the shared KV pool is full (…)`), never a crash and never a short write.

## 2. Exactness

### 2.1 The golden reference hashes (official checkpoint)

`golden/refhash.sh -- --shared-kv` (flag on) and `golden/refhash.sh` (flag off), `EXPECT=official` — the lane
serves deepseek-ai's official checkpoint. The official table, in every run:

```
tf c7605e0b6ba8d9d2 · k1 = nol2pf = fp8 = bf16 7941647b162df1a8 · temp 02ef3e0858d7eb45
ced16 8790ee9388f4abd8 · ced28 de99bfce6ea96ba8
```

| window (time) | build (binary sha256) | flag on | flag off |
|---|---|---|---|
| W0, 10:36–11:07 | base `f211791` (`75cab73308246cce`) | — | 8 of 8 |
| W1, 13:10–13:39 | int `e6fb7b9` (`aff0120ab057900c`) | refused (T1's refusal) | 8 of 8 |
| W2, 14:11–14:19 | int `a784e7e` (`99b24ddb32689465`) | **0 of 8** (every hash differed) | 8 of 8 |
| W3, 14:27–14:33 | int (`c6ca66e286ba7c76`) | 8 of 8 | — |
| W4, 14:41–15:30 | int `f22353b` (`5e7e7c37db59184e`) | 8 of 8 | 8 of 8 (W4's second `refhash`) |
| W6, 15:57–16:31 | int `2e9539d` (`456c42c48910c268`) | 8 of 8 | 8 of 8 |
| W10, 20:00–20:32 | `dd700b1` (`69e3693637d1dc98`) | 8 of 8 | 8 of 8 |

W2's flag-on run is the page-offset bug (§5, bug 2): `tf 9e3e227a289f5c16`, `k1 f52bc3d18b0c8b65`,
`fp8`/`bf16` `1157b49a247569d9`, `temp a9cbef0fca6716bb`, `ced16 1fb8c8a4751dd5f3`,
`ced28 a72eebd26d68e36e`; the prefill's first token was right and the first decode token wrong.

### 2.2 The serve tests with the flag on

Every test against a `--shared-kv` engine (`run4.sh --mode serve`, the part's arguments in
`box/w0.parts` → `logs/<W>/parts.txt`). `--ced` is the strict comparison (a turn of at least 128 new
positions replays the decoder and must equal the same prompt prefilled cold).

| test | what it compares | result | window |
|---|---|---|---|
| `serve_test.py --ced` | 17 reuse/CED/fork/snapshot checks | 17 of 17 | W4, W6, W14 |
| `snap_test.py --ced` | 19 snapshot/fork/divergence checks | 19 of 19 | W4, W6, W14 |
| `slice_test.py` | slices of a stored session | 0 failed | W4, W14 |
| `batch_test.py --slots 4` | tokens **and** logprobs, batched vs alone | 0 failed | W4, W6, W14 (flag off: 0 failed, W4) |
| `disk_test.py` | store-only vs spilling, then `--compare` | 4 of 4 against cold; `--compare` 12 of 12 equal | W4, W6, W14 |
| `kept_test.py` | a stop and start | `--compare` 4 of 4 turns equal | W4, W6, W14 |
| `pool_test.py` | the full pool's defined outcome | 3 of 3 | W5, W14 (`--session-gib 0.012 --prompt 4000 --max-new 8000`) |
| `share_test.py` | shared pages, live restarts, slot moves | 55 of 55 | W8, W14 (flag on), W8b (flag off) |
| `--mode batch --batch-tf`, `DSGB10_WIDE=64`, 8 sessions | each session's samples hash, flag on vs flag off | 8 of 8 equal at 64-row and at 16-row steps | W14 (`4ae0212`) |
| `revisit.py` | a revisit from disk equals its cold comparison | exit 0 | W2 (flag on), W4 |

W4's `pool_test` was 2 of 3: the 20,300-token prompt did not fit `--session-gib 0.02` (the snapshots share the
pool), so the "tokens before the error" check had none; W5 reran it at `--session-gib 0.012` and it passed 3 of 3
(the error text, 87 tokens sent before it equal the reference's first 32, and the engine up afterwards).

W8c ran `share_test.py` against the build **before** `2e9539d` and got **51 of 55** — the two bugs that commit
fixed (§5, bug 4) — where the fixed build is 55 of 55.

### 2.3 The kernel-level page checks

The tests build their own pool and slot table on the device (no flag, no server), run the same kernels through
the pages and compare bit for bit with the contiguous run in the same process.

| test | result | window |
|---|---|---|
| `slot_tests` (contiguous) | 3013 checks, 0 differ | W1 |
| `slot_tests --pages` (TQ: two sources share a page) | 6853 checks, 0 differ | W8b |
| `slot_tests --pages-badlayout` (the control) | 823 of 6853 differ, exit 1 | W8b |
| `fused_tests` (contiguous) | 1359 checks, 0 differ | W1 |
| `fused_tests --pages` | 2469 checks, 0 differ | W8b |
| `fused_tests --pages-badlayout` (the control) | 198 of 2469 differ, exit 1 | W8c |
| `test_pattn '' index pages` / `attn pages` / `comp pages` | 5 of 5 / 5 of 5 / 6 of 6 matched | W1 |

The bad-layout controls are the regression check for the page-offset bug (bug 2): with the views laid out by
`layout_of(ratio)` instead of `page_off(k, a)` they must differ, and they do.

### 2.4 The served gate (strict-exact arguments)

`bench/share_gate.py run` on both halves with the same `--nonce`, then `compare` — the same prompts, so every
case id must be equal. The **strict-exact arguments** are the production arguments minus `DSGB10_OEA` and the
wide64 knobs (`DSGB10_WIDE`, `DSGB10_WIDE_COST`, `DSGB10_DRAFT1`): `--max-ctx 1048576 --slots 16` + BUDGET,
head4/markov8 drafts, HEAD4_C1, CF_MLP, CERT.

| window | run | result |
|---|---|---|
| W5, 15:30–15:56 | strict, flag off (`f22353b`) | **pass** (resumed, fork, concurrent 1/4/8/16 all equal); perf-1 137.4 tok/s, perf-16 188.1 tok/s |
| W5 | strict, flag on (`f22353b`) | concurrent-4/8 **FAIL**; `compare` with flag off: 4 of 11 case ids differ |
| W6, 15:57–16:31 | strict, flag on (`2e9539d`) | **pass**; `compare` with W5's flag off: **11 of 11 case ids equal** |
| W8c, 18:50–19:37 | strict, flag on (`52c0640`) | **pass**; `compare` with W5's flag off: **11 of 11 equal** |
| W0/W1 | production arguments, base | **fail** concurrent-8 (W0 passed only because its tool's concurrent requests were warm; the tool was fixed in W1) |
| W4 | production minus `OEA` (WIDE on), flag off / flag on (`f22353b`) | fail concurrent-8/16 / fail concurrent-4/8/16 (the flag-on extra failures were bug 10) |
| W13, 22:06–22:31 | production minus `OEA` (WIDE on), flag off / flag on (`fc41169`, bug 10 fixed), same nonce | both fail only concurrent-8, on the same prompt (WIDE's ulp); `compare`: **11 of 11 case ids equal** |
| W4, W8c, production | full production arguments (`OEA`, WIDE), `--cases resumed,fork,perf` | **pass** (W4 flag off and on; W8c flag off and on; production 20:40) |

16 sessions grown at once past 140,000 tokens (`--cases grow16`):

| window | build | result |
|---|---|---|
| W4 | production args (WIDE on), flag on | 16 of 16 finals differ from alone |
| W5 | strict, flag on (`f22353b`) | 12 of 16 differ |
| W6 | strict, flag on (`2e9539d`) | **pass**: every final reply equals the same messages alone |

Fork on a freshly started engine (the fork case alone against a just-started engine, compared with the full
run's): W4 flag on pass; W6 strict flag on pass; W0/W1 base pass (`compare` 0 differing).

### 2.5 Why concurrency equality needs the strict arguments

"Concurrent = alone" cannot hold under the production arguments with **or without** `--shared-kv`, for two
reasons of the fork's own:

- **`DSGB10_OEA=4,12,9`** is the fork's batch-aware expert re-routing, documented lossy
  (`engine/kernels/moe.cu:1452`, `common.cuh:317`): from 9 rows of a step, a row's experts depend on the other
  rows, so under production arguments a reply depends on what else is in flight, by design. The evidence is the
  base: W1's unchanged `f211791` fails concurrent-8 with the fixed tool (`distinct prompt 4` differs from
  alone), and its `--nospec` run fails the same way.
- **The wide64 bodies** (`DSGB10_WIDE=64`, steps above 16 rows) are ~1 bf16 ulp off the 16-row path by the
  fork's own docs (`docs/wide64.md`, "Exactness"), so with WIDE on, concurrent-4/8/16 fail (W4) and grow16 is 16
  of 16 (W4) — flag on and flag off alike.

The exactness gates therefore run with the strict arguments (no `OEA`, no WIDE/WIDE_COST/DRAFT1), where the
flag-off run passes (W5) and the flag-on run passes from `2e9539d` on (W6, W8c) and its case ids equal the
flag-off ones 11 of 11. The speed runs keep the full production arguments (§4).

## 3. Memory

### 3.1 At ready, production arguments

Rank 0 and the other ranks, at ready, as the startup lines and the `free` samples report them.

| run | KV memory | device free, rank 0 / others | host available, rank 0 / others |
|---|---|---|---|
| W4, flag off (store 12 GiB) | 12.00 GiB store | 12.90 / 18.13 GiB | 17.63 / 17.9 GiB |
| W4, flag on, default cap | 26.00 GiB pool | 3.80 / 8.99 GiB | 8.60 / 8.9 GiB |
| W8c, flag off | 12.00 GiB store | 12.73 GiB | 17.58 GiB |
| W8c, flag on, `--session-gib 17` | 17.00 GiB pool (4,973 pages, 20,369,408 positions) | 12.70 GiB | 17.55 GiB |
| production, 20:40, flag on, `--session-gib 17` | 17.00 GiB pool (4,973 pages of 3.5 MiB, 20,369,408 positions; 16 slot rows of 257 and one 3.5 MiB zero page) | 18.24 GiB | 17.88 GiB |
| production, 22:55, `4ae0212`, `--session-gib 17` | the same pool | 18.06 GiB | 17.71 GiB |

The flag's default cap at production flags (`--slots 16 --max-ctx 1048576`, no `--session-gib`) is
12 GiB + 16 × 895.4 MiB ≈ 26.0 GiB, which takes the whole offer: W4's rank 0 ready with 3.80 GiB device free.
The lane therefore passes `--session-gib 17`, which keeps today's total and free memory (W8c: 12.70 GiB free
with the pool against 12.73 GiB with the store). The production start's own line and the fork's production
start agree: 18.24 / 17.88 GiB against 18.24 / 17.78 GiB (the fork ran without the flag's 3.7 MiB of blocks).

### 3.2 One live slot, and 16 of them at 1M

| what a live slot reserves, at 1M | flag off | flag on |
|---|---|---|
| the compressed caches | 895.4 MiB (938,992,640 B, a full-context slot) | — (the pool holds them) |
| window rings, 40 layers | 5.16 MiB | 5.16 MiB |
| compressor carries, 4 sources | 0.25 MiB | 0.25 MiB |
| the drafter's windows, 4 blocks | 0.52 MiB | 0.52 MiB |
| the slot table's row | — | 2,056 B (257 × 8 at 1M) |
| **total** | **895.4 MiB** | **5.9 MiB** |

16 live sessions at 1M in one pool (the design's sizing arithmetic, `docs/shared-kv.md` §7, at the service's
flags): 4,096 pages = 14.0 GiB, leaving 839 pages = 3.4M positions for other sessions. The flag's own blocks
are the slot table (32.9 KB = 16 rows × 257 × 8 B at 1M) and one 3.5 MiB zero page. This case was **not run**;
the grow16 runs of §2.4 grew 16 sessions past 140,000 tokens, not to 1M.

## 4. Speed

### 4.1 Decode, ms per token

`run4.sh --mode decode`, 247 tokens measured, `~/golden/synth/tokens.npy` (270,000 tokens), repeated runs.
Spread is the three (or two) runs.

| window (build) | run | 1k | 131k |
|---|---|---|---|
| W5 (`f22353b`) | int flag off ×3 | 16.941 / 16.945 / 16.989 (mean 16.958) | 17.166 / 17.146 / 17.207 (17.173) |
| W5 (`f22353b`) | int flag on ×3 | 17.120 / 17.063 / 17.031 (17.071, **+0.67 %**) | 17.315 / 17.270 / 17.242 (17.276, **+0.60 %**) |
| W6 (`2e9539d`) | base ×2 | 17.001 / 17.019 (17.010) | 17.188 / 17.245 (17.217) |
| W6 (`2e9539d`) | int flag off ×2 | 17.093 / 17.062 (17.078, +0.40 % over base) | 17.279 / 17.261 (17.270, +0.31 %) |
| W8c (`52c0640`) | flag off ×3 | 17.123 / 17.097 / 17.101 (17.107) | 17.293 / 17.307 / 17.274 (17.291) |
| W8c (`52c0640`) | flag on ×3 | 17.092 / 17.073 / 16.913 (17.026) | 17.284 / 17.267 / 17.124 (17.225) |

Every decode run of a pair gives the same token hashes (1k `7941647b162df1a8`, 131k `cab591a245b4eebe`), so the
comparison is of the same computation. W5/W6's flag-on cost (~0.10 ms a token over flag off: +0.113 ms at 1k and
+0.103 ms at 131k) was the emit kernel's page loads; TP's fix removed it, and W8c's flag-on runs are within
noise of the flag-off ones (indeed slightly faster in this sample, which is the noise of the box).

### 4.2 Per-kernel times (nsys)

W7, 16:32–16:59, rank 0, `--mode bench --ks 1` (per launch, µs, flag off / flag on), before TP's fix:

| kernel | launches a step | 1k | 131k |
|---|---|---|---|
| `emit_kernel` | 4 | 33.1 / 52.9 | 36.5 / 47.7 |
| `index_dense` | 8 | 21.3 / 22.8 | 38.2 / 40.6 |
| `attn_kernel` | 40 | 17.7 / 18.1 | 18.6 / 19.7 |

Base and int flag off are the same kernels within noise (bench wall 17.079 ms against 17.032 ms), so the
integration build costs nothing at the kernel level by itself; the flag-on deltas above are the emit's four
page-entry loads (the largest) plus the indexer's and the attention's indirection.

The emit fix's effect, `bench_emit` (new in TP, a realistic 4 GiB pool, an L2 flush a launch), W9/W10
(`945fa63` / `dd700b1`): the paged emit equals the contiguous one (+0.00 … +0.13 µs; +2.0 µs once, one 2 µs
timer step), `index_dense` paged +0.0 … +3.8 µs (0–10 %) — the design's accepted +2–5 % indirection — and
`sparse_attn` paged is equal or faster. The fix hoists the entry's two page loads to the top of `emit_row` and
prefetches the store lines; its SASS moves the four tail table loads to two at the top with five prefetches and
restores the weight-load burst (12 against the contiguous 13, was 8), while the flag-off instantiation's SASS
stays byte-identical. W12 (after bug 9's fix): every case's paged outputs equal the contiguous ones; emit
paged = contiguous within one 2 µs timer step, `index_dense` +0 … +9 µs (0–17 %), `sparse_attn` equal.

### 4.3 Throughput under production arguments

`share_gate.py run --cases perf --perf 1,16` with the full production arguments (`DSGB10_OEA`, WIDE):

| window (build) | flag | perf-1, tok/s | perf-16, tok/s aggregate | median per stream |
|---|---|---|---|---|
| W4 (`f22353b`) | off | 139.1 | 263.9 | 20.7 tok/s |
| W4 (`f22353b`) | on | 138.4 | 266.7 | 19.5 tok/s |
| W8c (`52c0640`) | off | 107.8 | 227.0 | 16.3 tok/s |
| W8c (`52c0640`) | on | 107.7 | 240.9 | 17.3 tok/s |
| production, 20:40 (`dd700b1`) | on | 117.9 | 240.8 | 18.3 tok/s |
| production, 22:55 (`4ae0212`) | on | 133.7 | 241.5 | 20.4 tok/s |

(Perf-1 and perf-16 differ between W4 and W8c because the box and the ring's state differ; each pair was run
back to back.)

### 4.4 TTFT cold / warm / appended

The production lane, 20:40, `share_gate.py run --cases resumed,fork,perf` (its key, nonce
`6a99b0bc06c242ff`, report `results/PROD-shared-kv.json`), seconds:

| prompt | cold | warm (same request again) | appended (+1 sentence) |
|---|---|---|---|
| 4k | 0.54 | 0.02 | 0.27 |
| 131k | 13.57 | 0.26 | 0.28 |
| 600k | 65.13 | 0.04 | 0.27 |

A fork of a 600k prompt at 131k: 51.69 s (cached 131,072 positions). The flag-off baseline of the same tool on
the base engine (W0, 10:36–11:07, before the tool's cold-prompt fix) was 131k cold 13.12 s, warm 0.27 s;
600k cold 51.36 s, warm 0.04 s, +1 sentence 0.24 s — its "cold" prompts shared prefixes with earlier cases
(cached 3,348 / 4,114 / 131,152), which is why W1 reran the gate with the fixed tool.

### 4.5 A revisit from disk, and prefill

W4, flag on, `revisit.py` (`--session-gib 0.5`, disk tier): a 36k-token revisit 0.087 s against a cold prefill
3.99 s; a 180k-token revisit 0.096 s against a cold 19.42 s. The design's win is that no cache is copied on the
way back.

Prefill, flag on against flag off: W4's 131k decode runs prefill 130,815 positions in 31.97 s (**4,092 tok/s**)
both on and off; W5's runs 4,170–4,179 tok/s; W8c's 4,118–4,207 tok/s. The base's own serve prefill (W0,
`serve_test --speed`) was 1k 323.0 ms (3,171 tok/s), 4k 601.8 ms (6,807), 8k 961.3 ms (8,522),
16k 1,825.7 ms (8,974). So the flag changes prefill by no more than the runs' own spread.

### 4.6 OEA and wide64: throughput and quality

Without `DSGB10_OEA` and the wide64 knobs (`DSGB10_WIDE`, `DSGB10_WIDE_COST`), a reply equals the same request
sent alone at any concurrency (§2.5). `DSGB10_DRAFT1` (the one-pass drafter) keeps that: its drafts are verified,
so it cannot change a reply (W20: `share_gate.py compare` lossless vs lossless + `DSGB10_DRAFT1=6`, 0 of 7 case ids
differ, concurrent-1/4/8/16 equal alone). What the others cost to drop, flag on throughout:

`bench/multistream.py` (the fork's 4 prompts, 512 tokens, T = 0), aggregate tok/s, two repetitions each:

| config | 1 stream | 4 | 8 | 16 |
|---|---|---|---|---|
| lossless (neither) | 77.0 / 77.0 | 150.4 / 150.1 | 184.2 / 183.7 | 323.1 / 335.7 |
| OEA only | 78.6 / 77.0 | 158.9 / 158.9 | 197.9 / 198.3 | 382.4 / 344.2 |
| WIDE only | 79.0 / 79.0 | 149.8 / 150.0 | 197.7 / 197.7 | 294.6 / 284.5 |
| OEA + WIDE | 77.1 / 78.9 | 158.5 / 158.8 | 212.7 / 214.1 | 311.1 / 310.9 |
| lossless, W20 (4871f62) | 76.9 / 78.7 | 149.0 / 149.1 | 183.5 / 184.2 | 358.8 / 333.1 |
| lossless + DRAFT1, W20 | 78.9 / 78.9 | 149.8 / 149.5 | 190.0 / 189.5 | 348.0 / 360.6 |

DRAFT1 adds 0.4–3.2 % (means of the two W20 reps) and is on in production since 4871f62.

`share_gate.py --cases perf` on the served lane (4k-token prompts, prefill included): lossless at `2588aba`
perf-1 112.9 / 136.8, perf-16 178.3 / 169.7 tok/s; OEA + WIDE at `4ae0212` 133.7 and 241.5 (§4.3).

Quality: `--mode batch --batch-tf` with `DSGB10_TF_DUMP` (the golden next token's rank and logprob in the top 20),
8 sessions × 2,560 positions of a 28,489-token transcript, contexts 1,024 … 18,944, no position repeated (20,480
rows); reference = lossless at 16-row steps. McNemar counts b (reference right, run wrong) and c (the reverse);
Δlp is over the rows with both in the top 20.

| run | top-1 | in top 20 | b / c, z | Δlp, nats/token (95 % CI) | perplexity ratio | GPU ms/step |
|---|---|---|---|---|---|---|
| lossless, 16-row steps | 12,739 (62.20 %) | 17,450 | — | — | — | 68.1 |
| OEA, 16-row steps | 12,633 (61.68 %) | 17,442 | 1,085 / 979, −2.33 | −0.019 ± 0.020 | 1.019 [0.999, 1.039] | 59.4 |
| OEA + WIDE, 64-row steps | 12,767 (62.34 %) | 17,457 | 969 / 997, +0.63 | +0.000 ± 0.019 | 1.000 [0.981, 1.019] | 131.5 |

OEA at 16-row steps costs ~0.5 pp of top-1 (z −2.33) and ~2 % perplexity, the latter not significant at this
size; at 64-row steps with WIDE nothing measurable. The z treats rows as independent; rows of one session are
correlated, so it overstates the significance. One transcript, one run each. Argmax agreement with the reference is
74–75 % in both runs: any numeric change moves argmaxes that much (issue #9), so it is not a loss measure.

## 5. Bugs found on the way

Bug 1 is pre-existing in the fork; 6–9 are in the test harness and the new bench, not in the engine's serving
path; 10 was in the serving path and reached production; 11–12 are in the golden modes; the rest are in the new
work. Each row: symptom, cause, fix, the check that now covers it.

| # | where | symptom | cause | fix | the check that covers it |
|---|---|---|---|---|---|
| 1 | pre-existing in the fork: the 9..16-row norm (TF16) | `batch_test --slots 4` **40 failed** on the base (`f211791`): every batched request's logprobs differ from its run alone, and two requests' tokens differ at near-ties (C, D: first differences at 27 of 200 and 201 of 320); the same at `--max-ctx 131072` | from 9 rows of a step the norm-fused GEMV prologues are replaced by a standalone `dsk::rmsnorm` (`unfused16()`, on by default since `6f6392e`), whose block is `D/8` (640 at K = 5120) against the fused prologue's 256; `rmsnorm_scale` partitions the sum of squares by `blockDim.x`, so the two disagree in the last bits (~0.03 bf16 elements a row), which moves a logit by ~1e-4 — above the reported logprob's ulp, below a typical top-2 gap | pass the fused prologue's block size (`norm_block`, 256) at the three sites (`model.cu` attn_norm, q_norm, the router gate); `dsk::rmsnorm` gained the argument, 0 keeps `D/8` for its other 27 callers | `batch_test --slots 4` 0 failed flag off and on (W4/W6); the host emulation of the two reductions (300/300 16-row draws identical at K = 5120 and 1280); `refhash.sh` unchanged (a single session's step is never above 8 rows) |
| 2 | new work (`5f0d039`) | W2 flag on: **every refhash differed** (`tf 9e3e227a289f5c16`, `k1 f52bc3d18b0c8b65`, `fp8`/`bf16 1157b49a247569d9`, `temp a9cbef0fca6716bb`, `ced16 1fb8c8a4751dd5f3`, `ced28 a72eebd26d68e36e`); the prefill's first token right, the first decode token wrong | `page_view` took a page's per-source offsets from `layout_of(ratio)`, which places every ratio-2 source at the start of the page, so layers 2, 8 and 14 overwrote each other's entries (a prefill chunk read what it had just written, decode read another source's); the same commit's `KvSrc::at` also skipped the carries' slot offset under the flag | `page_off(k, a)` (every source has its own block of the page), and `KvSrc::at`'s offset | W3's `refhash.sh -- --shared-kv` 8 of 8; the kernel tests' `--pages-badlayout` control (823 of 6853 and 198 of 2469 differ), which reproduces the layout bug by construction |
| 3 | new work (`f22353b`) | W3 flag on: refhash 8 of 8, but every serve test hung (the step watchdog, 120 s, on the first request) | serve captures its step graphs before the pool is reserved, and the slot table was taken out of the pool, so the captured views had a null table and read null contiguous caches | the slot table is made with the model (not from the pool), and `reserve_pages` writes every row | W4's serve tests: `serve_test --ced` 17 of 17, `snap` 19 of 19, `slice` 0 failed, `batch` 0 failed, `disk` 12 of 12, `kept` 4 of 4 |
| 4 | new work (`2e9539d`) | W5's strict gate flag on: concurrent-4/8 FAIL, `compare` with flag off 4 of 11 differ; `share_test` on that build 51 of 55 (the fork F1/F2 turns after the base's rewrite differ from cold, and scenario 4's last turns) | `state_drop` cut a live session's pages at its last *save*, not at the restart point, so a live regenerate or divergence lost entries it had written while live; and a restart inside a page a fork shared wrote over the fork's entries | drop at the restart point, and copy the shared tail page on write (copy-on-write) | W6's strict gate pass with `compare` 11 of 11 equal to the flag-off run; `share_test` 55 of 55 on the fixed build against 51 of 55 on the pre-fix one (W8c) |
| 5 | new work (`52c0640`, TP) | flag on decoded ~0.10 ms a token slower than flag off (W5: +0.113 ms at 1k, +0.103 at 131k; W6: +0.06 ms of that was the integration build's own) | `emit_row<kPaged>`'s four stores recomputed their page entry: two cold slot-table loads per cache in the kernel's tail, serialized with the stores they gate, and a smaller burst of the weight-load loop (8 against the contiguous 13) on a DRAM-latency-bound 33 µs read | hoist the entry's two page loads to the top of `emit_row` and prefetch the store lines; the flag-off instantiation's SASS stays byte-identical | W8c's decode A/B: flag on 17.026 / 17.225 ms against flag off 17.107 / 17.291 at 1k / 131k (no regression left); `bench_emit`; `slot_tests --pages` / `fused_tests --pages` bit for bit |
| 6 | the soak harness, pre-existing (`945fa63`) | W8c's soak FAILed on both builds from the harness: ranks 1–3 "no memory samples", and the flag-on run reported **0 evictions** while the disk tier showed spills | `soak.py` sampled fixed worker addresses belonging to another fleet, and counted only the server's `evict` log lines, which the pool's own spills (no server evict line) do not produce | the workers come from the fleet file; the pool's spills count as evictions | W9: flag off PASS (0 bad, 224 evictions), flag on FAIL (rank 0 not flat, bug 7); W10: both PASS |
| 7 | the soak harness + the engine's mapping names (`dd700b1`) | W9's flag-on soak FAILed: rank 0 was not flat, its Engram row cache found by its mapping's size as 0 → 0 MiB, so the cache's 512 MiB fill counted as growth (ranks 1–3's 800–1400 MiB were found) | the Engram row cache's mapping was unnamed and had merged with a neighbour, so its size could not be read | the mappings are named (`[anon:engram-rows]`) and `soak.py` finds the row cache by name | W10's flag-on soak PASS, every rank flat, rank 0's row cache found (6 → 446 MiB) |
| 8 | the bench, new (`945fa63`) | `bench_emit` crashed with a misaligned address (W8b) | its page stride was not on the 64 KiB grain | the bench places its pages on the grain | W9/W10's `bench_emit` runs its cases (exit 1 only on the `sparse_attn` comparison, bug 9) |
| 9 | the bench, new (`8ae8490`, TB) | `bench_emit` reported `sparse_attn` paged `DIFFERENT` (W9, and W10 with NaN-free inputs) | W9: the inputs held e4m3 NaN (scales 120–127, `0x7F` is NaN); then: the reference was copied with `Dev::host()` on the legacy stream before the kernel on the bench's non-blocking stream had written `out` (only the attention writes its output in a second, combine kernel) | NaN-free fills (`dd700b1`); every snapshot after `cudaStreamSynchronize`, and two contiguous runs compared first (`NONDETERMINISTIC`) (`8ae8490`) | W12: `bench_emit` **all cases equal** |
| 10 | new work, **in production 20:40–21:10** (`fc41169`) | none seen in the gates' passing configurations; found reading T2's documented gap: "a paged step wider than 16 rows resolves the later groups' rows from the unoffset table; no production step is that wide" — but the service sets `DSGB10_WIDE=64`. W13 batch-tf, 8 sessions × 8 rows (64-row steps), the build without the fix: 6 of 8 sessions' hashes differ from flag off | a step of more than 16 rows runs the indexer and the compressor's emit in groups of 16; `model.cu` offset the contiguous `slot` pointer by the group's first row but not `KvPages::slot`, so rows 16.. scanned and wrote the pages of rows 0..15's sessions | offset `KvPages::slot` per group | W14 batch-tf at 64-row steps: flag on = flag off for every session; the build without the fix differs |
| 11 | the golden modes (`91c029d`) | `--mode batch` under `--shared-kv` died on every rank: "slot 7 has no session to hold its pages" | `apply_ops` lets a slot with no server session go (`state_slot_row(s, 0)`), and the golden modes' slots hold handles `-(s + 1)` | a negative (golden) handle is kept on `h = 0` | W13/W14 batch-tf runs |
| 12 | the golden modes (`d1f0f1b`) | W13 batch-tf at 16-row steps: flag on differed from flag off for the sessions at contexts 4096 and 8192 | `batch_tf_loop` launched without a room call, so a session crossing a page boundary read and wrote the zero page (the server's loops all make room) | a room call per entry per step | W14 batch-tf at 16-row steps: flag on = flag off |
| 13 | new work (`4ae0212`) | none seen; T7's review listed it as open: a fork's or a restart's tail-page copy with the pool full of live sessions would end the engine (`die`) | the copy is allocated inside a state operation, which has no error path, and the pool's live pages cannot be evicted | one spare page held back (`SharedKv::spare`): a copy that finds the pool full takes it; a session's growth takes one back first, or ends as a full pool (`kEndPool`, the request's error, not the engine's) | W14's serve suite with it in place; the spare path itself not exercised |

The homelab lane watchdog's interaction with the ring (the plan's backlog 5) is a homelab fix, not an engine
one, and is not part of this feature.

## 6. How it was measured

**The windows.** W0–W14, 2026-11-14 10:36–22:56, each a run of parts against a build; the production run at
20:40. The ring was borrowed from the service lane (`pico-ds4`); the lane's idle yield was lowered to 120 s for
the windows and restored to 600 s at the end. W8 held a user's request (`PREEMPT=0`) and was stopped at 18:31;
from W8b on the windows are preemptible (`PREEMPT=1`), so a request stops a part and the part runs again later
(`box/window.sh`). The production lane ran `dd700b1` from 20:40 and went back to the fork's build at 21:10, when
bug 10 was found; it was redeployed at the final commit after W14.

| window (2026-11-14) | build | what it ran |
|---|---|---|
| W0, 10:36–11:07 | base `f211791` (`~/ds4-kv-share/base`) | the baseline: refhash, the serve tests, the gate, the speed test |
| W1, 13:10–13:39 | int `e6fb7b9` (flag refused) | the integration build flag off, the page-mode kernel tests, the refusals, the base controls |
| W2, 14:11–14:19 | int `a784e7e` | flag off (equal), flag on (every refhash differed → bug 2) |
| W3, 14:27–14:33 | int `c6ca66e286ba7c76` | flag on: refhash equal, serve tests hung → bug 3 |
| W4, 14:41–15:30 | int `f22353b` | the flag on end to end: refhash, serve tests, gates, memory, decode, prefill |
| W5, 15:30–15:56 | int `f22353b` | the strict gates, the decode A/B, `pool_test` → bug 4 |
| W6, 15:57–16:31 | int `2e9539d` | the strict gates pass, grow16, fork-cold, serve tests, refhash, decode |
| W7, 16:32–16:59 | int | nsys profiles of rank 0 (`--mode bench --ks 1`) |
| W8, 18:25–18:31 | int `52c0640` | `share_test` flag on 55 of 55 (stopped: a held request) |
| W8b, 18:32– | int `52c0640` | `share_test` flag off, the kernel page modes and their bad-layout controls, `bench_emit` → bug 8 |
| W8c, 18:50–19:2x | int `52c0640` | `share_test` on the pre-`2e9539d` build (51 of 55), the gates, the decode A/B |
| W9, 19:37–19:59 | int `945fa63` | the soak (flag on FAIL from rank 0's row cache → bug 7; flag off PASS), `bench_emit` |
| W10, 20:00–20:32 | int `dd700b1` | the soak both ways PASS, refhash both ways, `bench_emit` |
| W11, 20:32–20:39 | int `dd700b1` | `keptx.sh`: kept sessions across a mode switch |
| W12, 21:26–22:00 | int `1b8c3f5` (bug 10 fixed, TB) | batch-tf flag off (the reference hashes; flag on died: bug 11), refhash flag on, `bench_emit` all equal |
| W13, 22:01–22:31 | int `91c029d` | batch-tf flag on and the build without bug 10's fix (bug 12 found), the served gate with WIDE on, flag off vs on |
| W14, 22:32–22:56 | int `4ae0212` | batch-tf flag on = off at 64 and 16 rows, the no-fix control, refhash, the serve suite flag on |
| production, 20:40 | `dd700b1` in `~/pico-dgx-ds4-skv` | the lane on `--shared-kv --session-gib 17`, the served gate |
| production, 22:55 | `4ae0212` in `~/pico-dgx-ds4-skv` | the lane redeployed; the served gate `--cases resumed,fork,perf`: pass (600k cold 64.69 s, warm 0.04 s, appended 0.27 s) |

**The test engines.** `run4.sh --mode serve --socket ~/test.sock --disk-dir '~/test-spill' --disk-gib 8
--disk-tag <tag>` plus the part's engine arguments; the exact part lines are `box/w0.parts`, copied to
`logs/<W>/parts.txt`. The served gate ran against a test dsfront on `127.0.0.1:18890` (never `:8888`) with a
test-only key (`~/ds4-kv-share/test-key`); the production run used the lane's own key and `:8888`. No test used
the production lane except the production run itself.

**The token files.** `~/golden/long-tokens` (28,489 tokens, `make_cedcheck.py --tokens-only`, tokenizer sha256
`c90dfa01249db1be`) for the refhash and serve tests' default; `~/golden/synth/tokens.npy` for the tests whose
default is the fork's `golden/synth270k`, which is not shipped (it mixes texts this repo lacks) — the stand-in
is the long transcript repeated to 270,000 tokens (`box/mk_synth.py`), used by kept, batch, disk, snap and
slice. `EXPECT=official`, because the lane serves deepseek-ai's official checkpoint: `refhash.sh`'s expected
values are the table in §2.1.

**The gate tool and its nonce.** `bench/share_gate.py` (TG): prompts are a nonce line plus numbered filler
lines that name the case, so no case's prompt is a prefix of another's; both halves of a comparison run with
the **same `--nonce`**, or the prompts differ and so does every hash. `run` writes a JSON report, `compare`
matches case ids and lists the ones whose hashes differ. The fork case is one request, recorded and compared
against the same case on a freshly started engine (`--cases fork`), because dsfront matches a second send of
the same prompt to its own session as a resume. The production run: `run --cases resumed,fork,perf --perf 1,16`
against `http://127.0.0.1:8888/v1`, report `results/PROD-shared-kv.json`, nonce `6a99b0bc06c242ff`, `pass: true`,
no failures. The reports hold numbers only — case, role, size, sha256, timings, usage counts — no prompt or
reply text, and the engine's logs carry no token ids either.

**nsys.** W7 profiled rank 0 with `--mode bench --ks 1`, per launch, flag off and flag on, at 1k and 131k, with
the base and the int flag-off build as controls. The per-kernel numbers are §4.2; the profiler adds no request
content.

**The builds.** The windows ran the integration tree (`int`), whose binary sha256 is in each window's refhash
line; the production binary is the same source as the tested `dd700b1` but not the same bytes (sha256
`6e8c973a` against `69e36936`, likely the build paths compiled into the error messages — not checked).

## 7. Not tested, and open

- **A soak on the production lane under the flag, longer than the test's.** The soaks were 9 minutes each on a
  test engine (`--slots 4 --session-gib 2 --disk`, W9/W10), flag on and off; the production lane served the
  gate and was left running, but no soak was run on it.
- **Kept sessions across a production restart.** W11 tested a flag deploy/rollback (`keptx.sh`: kept_test
  "before" on one mode, a planned stop, "after" on the other, off → on and on → off): the new start reuses no
  kept session ("none reused: of another engine binary, weights or options"; the run id hashes the options,
  `--shared-kv` among them) and every session comes back cold (cached 0), so `kept_test --compare` reports
  1 of 4 turns equal (it expects warm turns) and a deploy or a rollback of the flag is a clean start. An actual
  production restart with kept sessions was not run.
- **16 live sessions at 1M.** §3.2 is the design's sizing arithmetic; the runs grew 16 sessions past 140,000
  tokens (W6), not to 1M.
- **The full pool at production scale.** `pool_test` ran at `--session-gib 0.012`/`0.02` and `--slots 2`/`4`
  (W4/W5); the 17 GiB pool's full-pool path (the error text, dsfront's HTTP 500 / SSE error, the session's
  release) was not exercised at that size.
- **Steps wider than 16 rows under the flag in the served path.** The wide64 group offset (bug 10) is checked
  by batch-tf at 64-row steps (W14), not by a served gate: concurrency under `DSGB10_WIDE` is not exact by the
  fork's own design (next item), so the served gates cannot compare it.
- **The wide64 bodies' ~1 bf16 ulp** (`docs/wide64.md`): exactness under `DSGB10_WIDE` cannot hold by the
  fork's own docs, so it is not a target of the strict gates (§2.5).
- **Concurrent = alone under `DSGB10_OEA`**: the fork's lossy batch-aware routing breaks it with or without the
  flag (§2.5); the strict gates are the exactness claim, the production-argument runs are the speed ones.
- **A copy-on-write with the pool full of live sessions.** It died (`state_copy_snaps`/`state_drop` have no
  error path); `4ae0212` holds one spare page back for it (bug 13). The spare path itself is not exercised by a
  test: the suites ran with it in place (W14) and did not need it.
- **The soak's greedy regenerates** are identical 9 of 13 (flag off) and 25 of 26 (flag on); the plan records
  this as observed and not a soak criterion.
