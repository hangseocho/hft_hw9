// seqlock.h — single-writer / multi-reader sequence lock.
//
//   SeqLock<T>::write(const T&)  : single writer, wait-free
//   SeqLock<T>::read(T&) const   : readers never block the writer and never
//                                   write shared memory; retry on a torn read.
//
// Protocol (even/odd sequence counter):
//   writer:  seq = s+1 (odd)  -> copy payload -> seq = s+2 (even, release)
//   reader:  s1 = seq (acquire); if odd -> retry
//            copy payload
//            fence(acquire); s2 = seq; if s1 != s2 -> torn, retry
//
// Build switch:
//   SEQLOCK_ATOMIC_PAYLOAD=1  store the payload as relaxed std::atomic<uint64_t>
//                             words. Formally race-free and ThreadSanitizer-clean
//                             (same machine code on x86-64: plain MOVs).
//   default (0)               plain memcpy of T. This is the classic seqlock and
//                             is a formal data race -> TSan reports it (see README).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SEQLOCK_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define SEQLOCK_PAUSE() __asm__ __volatile__("yield")
#else
#define SEQLOCK_PAUSE() ((void)0)
#endif

#ifndef SEQLOCK_ATOMIC_PAYLOAD
#define SEQLOCK_ATOMIC_PAYLOAD 0
#endif

inline constexpr std::size_t kCacheLine = 64;

template <typename T>
class SeqLock {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SeqLock payload must be trivially copyable (it is copied byte-wise "
                  "and may be observed half-written)");

public:
    SeqLock() noexcept { T zero{}; store_payload(zero); }
    explicit SeqLock(const T& init) noexcept { store_payload(init); }

    SeqLock(const SeqLock&) = delete;
    SeqLock& operator=(const SeqLock&) = delete;

    // Single writer only. Never blocks.
    void write(const T& v) noexcept {
        // Only this thread modifies seq_, so a relaxed load of our own value is fine.
        const std::uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);         // odd: write in progress
        // Release fence: the odd store above is ordered before the payload stores
        // below (as seen by any reader that later performs an acquire on seq_).
        std::atomic_thread_fence(std::memory_order_release);
        store_payload(v);
        seq_.store(s + 2, std::memory_order_release);         // even: publish
    }

    // Lock-free read. Retries torn reads until it gets a consistent snapshot.
    // max_retries == 0 (default) means "retry until success" -> always true.
    // A finite max_retries returns false if every attempt was torn.
    // Exponential backoff (1,2,4..64 pauses) on retry gives the writer room to
    // finish, so a reader is not starved by a back-to-back writer.
    bool read(T& out, std::uint32_t max_retries = 0) const noexcept {
        std::uint32_t backoff = 1;
        for (std::uint32_t i = 0; max_retries == 0 || i < max_retries; ++i) {
            if (try_read(out)) return true;
            for (std::uint32_t p = 0; p < backoff; ++p) SEQLOCK_PAUSE();
            if (backoff < 64) backoff <<= 1;
        }
        return false;
    }

    // One attempt; false means "torn or writer active, try again".
    bool try_read(T& out) const noexcept {
        const std::uint32_t s1 = seq_.load(std::memory_order_acquire);
        if (s1 & 1u) return false;                            // writer active
        load_payload(out);
        // Acquire fence: payload loads above cannot be reordered after the
        // seq_ re-load below.
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::uint32_t s2 = seq_.load(std::memory_order_relaxed);
        return s1 == s2;
    }

    std::uint32_t sequence() const noexcept {
        return seq_.load(std::memory_order_acquire);
    }

private:
#if SEQLOCK_ATOMIC_PAYLOAD
    static constexpr std::size_t kWords = (sizeof(T) + 7) / 8;

    void store_payload(const T& v) noexcept {
        std::uint64_t buf[kWords] = {};
        std::memcpy(buf, &v, sizeof(T));
        for (std::size_t i = 0; i < kWords; ++i)
            words_[i].store(buf[i], std::memory_order_relaxed);
    }
    void load_payload(T& out) const noexcept {
        std::uint64_t buf[kWords];
        for (std::size_t i = 0; i < kWords; ++i)
            buf[i] = words_[i].load(std::memory_order_relaxed);
        std::memcpy(&out, buf, sizeof(T));
    }
#else
    void store_payload(const T& v) noexcept { std::memcpy(&data_, &v, sizeof(T)); }
    void load_payload(T& out) const noexcept {
        // Racy by design: may read a half-written value; the seq re-check rejects it.
        std::memcpy(&out, &data_, sizeof(T));
    }
#endif

    // seq_ and payload share one cache line (for small T): a reader pays one
    // line transfer per writer update instead of two. The whole object is
    // cache-line aligned and padded so nothing else lands on that line.
    alignas(kCacheLine) std::atomic<std::uint32_t> seq_{0};
#if SEQLOCK_ATOMIC_PAYLOAD
    std::atomic<std::uint64_t> words_[kWords];
#else
    T data_;
#endif
    // tail padding to a full line comes from alignas on the first member
};

// Example payload from the assignment.
struct Quote {
    double bid, ask;
    std::uint32_t bq, aq;
};
static_assert(sizeof(Quote) == 24);