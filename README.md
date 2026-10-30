# Week 9 — SeqLock: lock-free quote publication

**Files:** `seqlock.h` (implementation), `seqlock_bench.cpp` (correctness test + benchmark)

```bash
clang++ -std=c++20 -O2 -pthread seqlock_bench.cpp -o seqlock_bench
./seqlock_bench            # correctness, then benchmark
./seqlock_bench ablation   # one-change-at-a-time attribution
```

**Machine:** Apple Silicon MacBook, 8 hardware threads, macOS, Apple clang, `-O2`.
Apple Silicon has no user-mode cycle counter, so **ns/op is measured** (`steady_clock`)
and **cycles/op is estimated as ns/op x 3.2 GHz**. Each number is the **median of 51 runs
x 200k reads**, after a 200k-read warm-up. macOS cannot pin threads to cores, so the
scheduler places the writer and reader.

## 1. Design

`T = Quote { double bid, ask; uint32_t bq, aq; }` (24 B). The sequence counter and payload
share **one 64 B-aligned cache line**, so a reader pays one line transfer per update.

| Step | Writer (single) | Reader (lock-free) |
|---|---|---|
| 1 | `seq.store(s+1, relaxed)` (odd = busy) | `s1 = seq.load(acquire)`; odd -> retry |
| 2 | `atomic_thread_fence(release)` | copy payload |
| 3 | copy payload | `atomic_thread_fence(acquire)` |
| 4 | `seq.store(s+2, release)` (even = published) | `s2 = seq.load(relaxed)`; `s1 != s2` -> torn, retry |

- **Writer -> reader:** the release store of `s+2` pairs with the reader's acquire load of
  `s1`. A reader that sees an even `s1` therefore sees the complete payload of that write.
- **Detecting a concurrent write:** the writer's release fence orders the odd store before
  the payload stores. The reader's acquire fence keeps the payload loads before the `s2`
  load. If any byte the reader copied came from a newer write, `s2` must differ from `s1`,
  so torn copies are always rejected.
- The writer never waits for readers. `read()` retries with exponential backoff
  (1 -> 64 pauses) and, by default, until it succeeds.

## 2. Correctness

Each write `k` publishes a self-checking quote: `bid=k`, `ask=k+0.25`, `bq=k`, `aq=~k`. A
reader flags a **torn** read if the fields disagree and a **regress** if `k` goes backwards.
The test uses 1 writer x 20M writes.

| Implementation | Readers | Reads | Torn | Regress | Result |
|---|---|---|---|---|---|
| SeqLock | 1 | 48,064 | 0 | 0 | PASS |
| SeqLock | 3 | 1,809,123 | 0 | 0 | PASS |
| SeqLock | 7 | 112,908,080 | 0 | 0 | PASS |
| NaiveSeqLock (v0) | 3 | 259,661,189 | 0 | 0 | PASS |
| **NoCheck** (no seq check, control) | 3 | 127,014,093 | **105,551,101** | 0 | tears detected, as expected |

The control matters. Without the sequence check, 83% of reads are torn, which shows the
test really can catch a broken implementation.

## 3. Performance — reader cost under a live writer

| Scenario | `shared_mutex` | seqlock v0 (naive) | **seqlock v1** | v1 vs v0 | v1 vs `shared_mutex` |
|---|---|---|---|---|---|
| A. no writer | 50.9 cyc · 15.89 ns | 11.0 · 3.44 | **11.0 · 3.44** | 1.00x | **4.6x** |
| B. 1 update / 1000 cyc | 429.6 · 134.24 | 15.9 · 4.97 | **12.9 · 4.04** | 1.23x | **33x** |
| C. 1 update / 200 cyc | 352.5 · 110.15 | 27.2 · 8.49 | **20.7 · 6.48** | 1.31x | **17x** |
| D. back-to-back writer | 378.6 · 118.31 | 585.8 · 183.06 | **380.5 · 118.89** | 1.54x | 1.0x |

Throughput for v1 is 290.6 / 247.3 / 154.2 / 8.4 Mreads/s in scenarios A / B / C / D.

**v0 ("before")** is the first version one usually writes. It uses `seq_cst` for every
operation, the writer bumps the counter with `fetch_add` (an atomic RMW), there is no
cache-line alignment, and it retries without backoff.

**v1 ("after")** is the final design. The writer uses plain relaxed/release stores, the
reader uses acquire loads and fences, the data sits on its own aligned line, and retries
use exponential backoff.

How to read the scenarios:

- **A/B/C (realistic quote rates).** The seqlock is 4.6x cheaper than `shared_mutex` with
  no writer and 17–33x cheaper with a live writer. A seqlock read is only loads: it never
  writes shared memory. A `shared_lock`/`unlock` pair is two atomic RMWs on the lock word.
  With a live writer, that lock line also moves between cores on every acquisition, and
  the reader may have to wait while the writer holds the lock.
- **v0 vs v1 is a small effect on this machine.** The v0 number for A was 11.0 in this run
  and 13.9 in the ablation run (Section 4), and B moved from 15.9 to 13.2. So differences
  of 2–3 cycles are within run-to-run noise. The 1.2–1.3x "before/after" ratio in B/C
  should not be read as a reliable speedup.
- **D (writer never pauses) is dominated by the writer's rate, not the reader's code.**
  Every write invalidates the line the reader needs, so reader cost tracks how fast the
  writer runs. D is also very noisy on macOS because threads cannot be pinned (v0 measured
  585.8 cycles here but 69.5 in the ablation run). `shared_mutex` looks competitive in D
  only because it makes the **writer wait for readers**, which slows publication and leaves
  quotes stale. The seqlock never delays the writer, and that is the property market data
  needs.

## 4. The single change that helped most

Ablation (`./seqlock_bench ablation`, same machine, reader cycles/op, median of 51 runs).
Each variant is v0 with exactly **one** change:

| Variant | A. no writer | B. 1 update / 1000 cyc | D. back-to-back |
|---|---|---|---|
| v0 naive | 13.9 | 13.2 | 69.5 |
| + acquire fence (reader `seq_cst` fence -> acquire) | 11.6 | 13.0 | 70.2 |
| + release writer (`fetch_add` RMW -> relaxed/release stores) | 11.0 | 12.5 | 178.0 |
| + align / pause | 11.0 | 13.3 | 69.1 |
| all three | 11.0 | 12.9 | 189.4 |
| SeqLock v1 (final) | 11.0 | 12.6 | 216.5 |

**The change that helped most was the design itself: replacing the `shared_mutex` read path
with a seqlock read that only performs loads.** Under a writer publishing every ~1000
cycles, reader cost fell from **429.6 to 12.9 cycles/op (33x)**, and with no writer from
50.9 to 11.0 (4.6x). A reader-writer lock forces every reader to *write* the shared lock
word: two atomic RMWs per read, plus a cache-line transfer whenever the writer has touched
the lock. A seqlock reader never writes shared state, so in the common case it costs about
two loads of a line that is already in its cache, plus a 24-byte copy.

By contrast, the memory-ordering micro-optimizations barely matter on Apple Silicon. In
B, every variant lands within 12.5–13.3 cycles, inside the noise. ARM64 implements acquire
loads as `ldar`/`ldapr` and acquire fences as `dmb ishld`, and these are cheap on M-series
cores, so `seq_cst` costs little extra for the reader. (On x86, a `seq_cst` fence becomes a
full `MFENCE`, and the same change is expected to matter much more.)

The D column also teaches something. Making the **writer** cheaper (dropping the
`fetch_add` RMW) *raised* reader cost from 69.5 to 178.0 cycles. The faster writer
publishes more updates, so it invalidates the reader's line more often. Under a saturated
writer, reader cycles/op measures the writer's speed, not the reader's code. That is why
the realistic scenarios A–C are the ones to judge.

## 5. Why a seqlock trips ThreadSanitizer, and why it is still correct

```bash
clang++ -std=c++20 -O1 -g -fsanitize=thread -pthread seqlock_bench.cpp -o tsan_plain && ./tsan_plain test
clang++ -std=c++20 -O1 -g -fsanitize=thread -pthread -DSEQLOCK_ATOMIC_PAYLOAD=1 seqlock_bench.cpp -o tsan_atomic && ./tsan_atomic test
```

**Why TSan fires.** In the classic seqlock the reader `memcpy`s the payload *while the
writer may be writing it*. Under the C++ memory model, two non-atomic accesses to the same
location from different threads, at least one a write, with no happens-before between them,
are a data race. That is formally undefined behavior, and it is exactly what TSan detects.

TSan cannot know that the result of a racy copy is **discarded** whenever `s1 != s2`. In
addition, TSan ignores standalone `atomic_thread_fence` (GCC even warns with `-Wtsan`), so
it cannot model the fence-based ordering that makes the protocol work. Detection is
timing-dependent, so the warning appears in some runs and not in others.

**How we reason about correctness anyway.**

1. **Protocol argument (Section 1).** Any read that returns `true` saw the same even `s1`
   before and after the copy. The fences guarantee that no part of a newer write could have
   been copied without changing `seq`, so every accepted snapshot is a complete single
   write.
2. **The racy copy can never be observed.** `T` must be trivially copyable
   (`static_assert`), so a half-written copy is just bytes in a local buffer. No
   constructor, pointer, or invariant runs on it before the check rejects it.
3. **Empirical stress.** Section 2 shows zero torn or out-of-order reads over hundreds of
   millions of reads, plus a negative control that does catch tears.
4. **A race-free build.** With `-DSEQLOCK_ATOMIC_PAYLOAD=1`, the payload is stored as
   `std::atomic<uint64_t>` words accessed with `memory_order_relaxed`. This is legal C++
   with no data race, and TSan reports nothing. On x86-64 and ARM64, relaxed 8-byte
   atomic loads and stores compile to the same plain `ldr`/`str` or `mov` instructions,
   so the formally correct version costs nothing. (P1478 proposes `atomic_load_per_byte_memcpy`
   for exactly this use case.)
