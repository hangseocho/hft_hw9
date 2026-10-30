// seqlock_bench.cpp — correctness test + benchmark for SeqLock<Quote>.
//
// Build:  g++ -std=c++20 -O2 -pthread seqlock_bench.cpp -o seqlock_bench   (Linux or macOS)
// Run:    ./seqlock_bench            (correctness test, then benchmarks)
//         ./seqlock_bench test       (correctness only — use this under TSan)
#include "seqlock.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <pthread.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

// ---------------------------------------------------------------- platform layer
// x86-64      : cycles from RDTSC (constant-rate TSC).
// ARM / macOS : no user-mode cycle counter, so cycles = steady_clock ns x CPU_GHZ
//               (env var, default 3.2 = Apple M1 P-core). Printed as "(est.)".
static double g_ghz = 3.2;
#if defined(__x86_64__) || defined(__i386__)
static constexpr bool kTrueCycles = true;
static inline std::uint64_t cycles_now() { return __rdtsc(); }
static inline void cpu_relax() { _mm_pause(); }
#else
static constexpr bool kTrueCycles = false;
static inline std::uint64_t cycles_now() {
    const auto ns = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return std::uint64_t(ns * g_ghz);
}
static inline void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}
#endif

// Pin threads on Linux. macOS has no affinity API; threads are left to the scheduler.
static void pin_to_cpu([[maybe_unused]] int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu % std::thread::hardware_concurrency(), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

template <class V> static double median(V v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Writer value k encodes a self-checking quote: every field derives from k,
// so a reader can tell whether the four fields come from the same write.
static Quote make_quote(std::uint64_t k) {
    return Quote{double(k), double(k) + 0.25, std::uint32_t(k), ~std::uint32_t(k)};
}
static bool consistent(const Quote& q) {
    const auto k = std::uint64_t(q.bid);
    return q.ask == q.bid + 0.25 && q.bq == std::uint32_t(k) && q.aq == ~std::uint32_t(k);
}

static void spin_cycles(std::uint64_t c) {
    if (!c) return;
    const std::uint64_t t0 = cycles_now();
    while (cycles_now() - t0 < c) cpu_relax();
}

// ---------------------------------------------------------------- baselines
// Baseline: the same API protected by std::shared_mutex.
template <typename T> class SharedMutexBox {
public:
    void write(const T& v) { std::unique_lock lk(m_); data_ = v; }
    bool read(T& out) const { std::shared_lock lk(m_); out = data_; return true; }
private:
    mutable std::shared_mutex m_;
    T data_{};
};

// "Before" version: the first seqlock one typically writes.
//  - every atomic op is seq_cst (writer's stores become XCHG on x86)
//  - no cache-line alignment: seq/payload sit on whatever line the
//    allocator gives them, next to unrelated hot data
//  - retry loop spins without PAUSE
template <typename T> class NaiveSeqLock {
public:
    void write(const T& v) {
        seq_.fetch_add(1);              // seq_cst RMW  (LOCK XADD)
        std::memcpy(&data_, &v, sizeof(T));
        seq_.fetch_add(1);
    }
    bool read(T& out) const {
        for (;;) {
            unsigned s1 = seq_.load();
            if (s1 & 1) continue;
            std::memcpy(&out, &data_, sizeof(T));
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (seq_.load() == s1) return true;
        }
    }
    std::atomic<unsigned> seq_{0};
    T data_{};
};

// Ablation: v0 with exactly one aspect changed at a time, to attribute the speedup.
//   kAcqFence : reader uses acquire fence instead of seq_cst fence (MFENCE)
//   kRelStore : writer uses relaxed/release stores instead of seq_cst fetch_add
//   kAlignPause: alignas(64) + PAUSE in the retry loop
template <typename T, bool kAcqFence, bool kRelStore, bool kAlignPause>
class AblationSeqLock {
public:
    void write(const T& v) {
        if constexpr (kRelStore) {
            unsigned s = seq_.load(std::memory_order_relaxed);
            seq_.store(s + 1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            std::memcpy(&data_, &v, sizeof(T));
            seq_.store(s + 2, std::memory_order_release);
        } else {
            seq_.fetch_add(1);
            std::memcpy(&data_, &v, sizeof(T));
            seq_.fetch_add(1);
        }
    }
    bool read(T& out) const {
        for (;;) {
            unsigned s1 = seq_.load(kAcqFence ? std::memory_order_acquire
                                              : std::memory_order_seq_cst);
            if (!(s1 & 1)) {
                std::memcpy(&out, &data_, sizeof(T));
                std::atomic_thread_fence(kAcqFence ? std::memory_order_acquire
                                                   : std::memory_order_seq_cst);
                if (seq_.load(kAcqFence ? std::memory_order_relaxed
                                        : std::memory_order_seq_cst) == s1) return true;
            }
            if constexpr (kAlignPause) cpu_relax();
        }
    }
private:
    alignas(kAlignPause ? 64 : alignof(std::atomic<unsigned>)) std::atomic<unsigned> seq_{0};
    T data_{};
};

// Deliberately broken: no sequence check. Used to prove the test can see tears.
template <typename T> class NoCheckBox {
public:
    void write(const T& v) {
        // byte-wise volatile copy so the compiler cannot make it a single wide store
        auto* d = reinterpret_cast<volatile unsigned char*>(&data_);
        auto* s = reinterpret_cast<const unsigned char*>(&v);
        for (std::size_t i = 0; i < sizeof(T); ++i) d[i] = s[i];
    }
    bool read(T& out) const {
        auto* s = reinterpret_cast<const volatile unsigned char*>(&data_);
        auto* d = reinterpret_cast<unsigned char*>(&out);
        for (std::size_t i = 0; i < sizeof(T); ++i) d[i] = s[i];
        return true;
    }
private:
    alignas(64) T data_{};
};

// ---------------------------------------------------------------- correctness
struct TestResult { std::uint64_t reads = 0, torn = 0, regress = 0, failed = 0; };

template <class Box>
static TestResult correctness(int readers, std::uint64_t writes) {
    Box box;
    box.write(make_quote(0));
    std::atomic<bool> done{false};
    std::vector<TestResult> res(readers);
    std::vector<std::thread> th;

    for (int r = 0; r < readers; ++r)
        th.emplace_back([&, r] {
            TestResult t;
            std::uint64_t last = 0;
            Quote q;
            while (!done.load(std::memory_order_relaxed)) {
                if (!box.read(q)) { ++t.failed; continue; }
                ++t.reads;
                if (!consistent(q)) { ++t.torn; continue; }
                const auto k = std::uint64_t(q.bid);
                if (k < last) ++t.regress;   // must never go back in time
                last = k;
            }
            res[r] = t;
        });

    for (std::uint64_t k = 1; k <= writes; ++k) box.write(make_quote(k));
    done = true;
    for (auto& t : th) t.join();

    TestResult sum;
    for (auto& t : res) {
        sum.reads += t.reads; sum.torn += t.torn;
        sum.regress += t.regress; sum.failed += t.failed;
    }
    return sum;
}

template <class Box>
static bool report_test(const char* name, int readers, std::uint64_t writes, bool expect_clean) {
    auto r = correctness<Box>(readers, writes);
    // Correctness = no torn value and no time travel. (failed = read() gave up; with the
    // default unbounded read() this is always 0.)
    const bool clean = r.torn == 0 && r.regress == 0;
    std::printf("  %-22s readers=%d writes=%llu reads=%llu torn=%llu regress=%llu failed=%llu -> %s\n",
                name, readers, (unsigned long long)writes, (unsigned long long)r.reads,
                (unsigned long long)r.torn, (unsigned long long)r.regress,
                (unsigned long long)r.failed, clean == expect_clean ? "PASS" : "FAIL");
    return clean == expect_clean;
}

// ---------------------------------------------------------------- benchmark
struct BenchResult { double cyc_per_op, ns_per_op, mops; };

// One writer (CPU 0) updating every `gap` cycles, one timed reader (CPU 1).
// Each run: `ops` reads timed with rdtsc + steady_clock. Median over `runs`.
template <class Box>
static BenchResult bench(std::uint64_t gap, int runs, std::uint64_t ops, bool live_writer) {
    Box box;
    box.write(make_quote(1));
    std::atomic<bool> stop{false}, go{false};

    std::thread writer;
    if (live_writer)
        writer = std::thread([&] {
            pin_to_cpu(0);
            std::uint64_t k = 1;
            while (!go.load(std::memory_order_acquire)) cpu_relax();
            while (!stop.load(std::memory_order_relaxed)) {
                box.write(make_quote(++k));
                spin_cycles(gap);
            }
        });

    std::vector<double> cyc, ns;
    double sink = 0;
    std::thread reader([&] {
        pin_to_cpu(1);
        go.store(true, std::memory_order_release);
        Quote q;
        for (std::uint64_t i = 0; i < ops; ++i) { box.read(q); sink += q.bid; }   // warm-up
        for (int r = 0; r < runs; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t c0 = cycles_now();
            for (std::uint64_t i = 0; i < ops; ++i) {
                box.read(q);
                sink += q.bid;          // consume so the read can't be elided
            }
            const std::uint64_t c1 = cycles_now();
            const auto t1 = std::chrono::steady_clock::now();
            cyc.push_back(double(c1 - c0) / ops);
            ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / ops);
        }
    });
    reader.join();
    stop = true;
    if (writer.joinable()) writer.join();
    if (sink == 42.0) std::puts("");    // keep sink alive

    const double n = median(ns);
    return {median(cyc), n, 1e3 / n};
}

template <class Box>
static void report_bench(const char* name, std::uint64_t gap, bool live) {
    auto b = bench<Box>(gap, 51, 200'000, live);
    std::printf("  %-22s %10.1f %10.2f %12.1f\n", name, b.cyc_per_op, b.ns_per_op, b.mops);
}

static void bench_suite(const char* title, std::uint64_t gap, bool live) {
    std::printf("\n%s\n  %-22s %10s %10s %12s\n", title, "impl", "cycles/op", "ns/op", "Mreads/s");
    report_bench<SharedMutexBox<Quote>>("shared_mutex", gap, live);
    report_bench<NaiveSeqLock<Quote>>("seqlock v0 (naive)", gap, live);
    report_bench<SeqLock<Quote>>("seqlock v1 (final)", gap, live);
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    const bool test_only = argc > 1 && std::string(argv[1]) == "test";
    if (const char* g = std::getenv("CPU_GHZ")) g_ghz = std::atof(g);
    if (!kTrueCycles)
        std::printf("note: no cycle counter on this CPU -> cycles/op = ns/op x %.2f GHz (est.)"
                    "  [set CPU_GHZ to change]\n", g_ghz);
    if (argc > 1 && std::string(argv[1]) == "ablation") {
        for (auto [title, gap, live] : {std::tuple{"[A] no writer", 0ull, false},
                                        std::tuple{"[B] writer gap 1000 cyc", 1000ull, true},
                                        std::tuple{"[D] writer back-to-back", 0ull, true}}) {
            std::printf("\n%s\n  %-22s %10s %10s %12s\n", title, "variant", "cycles/op", "ns/op", "Mreads/s");
            report_bench<AblationSeqLock<Quote, false, false, false>>("v0 naive", gap, live);
            report_bench<AblationSeqLock<Quote, true,  false, false>>("v0 + acquire fence", gap, live);
            report_bench<AblationSeqLock<Quote, false, true,  false>>("v0 + release writer", gap, live);
            report_bench<AblationSeqLock<Quote, false, false, true >>("v0 + align/pause", gap, live);
            report_bench<AblationSeqLock<Quote, true,  true,  true >>("all three", gap, live);
            report_bench<SeqLock<Quote>>("SeqLock v1 (final)", gap, live);
        }
        return 0;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("hardware threads: %u   SEQLOCK_ATOMIC_PAYLOAD=%d\n", hw, SEQLOCK_ATOMIC_PAYLOAD);

    std::puts("\n== correctness: 1 writer + N readers, self-checking payload ==");
    bool ok = true;
    const std::uint64_t W = test_only ? 2'000'000 : 20'000'000;
    for (int readers : {1, 3, 7}) ok &= report_test<SeqLock<Quote>>("SeqLock", readers, W, true);
    ok &= report_test<NaiveSeqLock<Quote>>("NaiveSeqLock", 3, W, true);
    if (!test_only)   // sanity: the test must be able to catch a broken implementation
        ok &= report_test<NoCheckBox<Quote>>("NoCheck (expect tears)", 3, W, false);
    std::printf("correctness: %s\n", ok ? "ALL PASS" : "FAILURES");
    if (test_only || !ok) return ok ? 0 : 1;

    std::puts("\n== benchmark: 1 reader (CPU1), median of 51 runs x 200k reads ==");
    bench_suite("[A] no writer (uncontended read)", 0, false);
    bench_suite("[B] live writer, ~1 update / 1000 cycles (~2.8M quotes/s)", 1000, true);
    bench_suite("[C] live writer, ~1 update / 200 cycles", 200, true);
    bench_suite("[D] live writer, back-to-back (worst case)", 0, true);
    return 0;
}