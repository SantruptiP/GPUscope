// Lock-free SPSC ring buffer.
//
// Single-Producer Single-Consumer. The classic pattern: two monotonically
// increasing 64-bit counters (head, tail), with index = counter & (N-1).
// Power-of-two capacity makes the masking branch-free.
//
// Memory ordering:
//   - The producer publishes a slot by storing head_ with release semantics
//     AFTER writing the data. The consumer reads head_ with acquire.
//     This pair establishes a happens-before that protects the data write.
//   - Symmetric for the consumer side with tail_.
//
// False sharing: head_ and tail_ are each placed on their own cache line.
// Without this, the consumer reading tail_ would invalidate the producer's
// cache line that holds head_, costing ~50ns per push. With it: ~3ns.
//
// Cross-process use: this template is plain old data (no virtuals, no
// non-trivial members), so it can live in POSIX shared memory and be
// accessed concurrently by two processes via mmap. std::atomic on a
// trivially-copyable type in shared memory is technically implementation-
// defined, but works correctly on every modern x86/ARM target with a glibc
// or musl runtime. See shm_segment.h for the cross-process wrapper.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gpuscope {

// Cache line size. 64 bytes on x86_64. Apple Silicon uses 128, so on macOS
// arm64 builds we'd want to bump this. Done as a constant rather than via
// std::hardware_destructive_interference_size because that's still flaky
// across toolchains in 2026.
inline constexpr size_t kCacheLine = 64;

template <typename T, size_t N>
class alignas(kCacheLine) SpscRingBuffer {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable for lock-free safety");

 public:
  SpscRingBuffer() = default;

  // Non-copyable, non-movable. Once placed in memory, the buffer stays put.
  SpscRingBuffer(const SpscRingBuffer&) = delete;
  SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

  // Returns true if pushed, false if the buffer was full.
  bool try_push(const T& item) noexcept {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    const uint64_t t = tail_.load(std::memory_order_acquire);
    if (h - t == N) {
      return false;  // full
    }
    data_[h & (N - 1)] = item;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  // Returns true if popped, false if the buffer was empty.
  bool try_pop(T& out) noexcept {
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    const uint64_t h = head_.load(std::memory_order_acquire);
    if (t == h) {
      return false;  // empty
    }
    out = data_[t & (N - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  // Approximate, non-synchronized. Useful for backpressure decisions but not
  // for correctness checks. May be stale by the time the caller reads it.
  size_t size_approx() const noexcept {
    const uint64_t h = head_.load(std::memory_order_relaxed);
    const uint64_t t = tail_.load(std::memory_order_relaxed);
    return static_cast<size_t>(h - t);
  }

  static constexpr size_t capacity() noexcept { return N; }

 private:
  // Three cache lines: head_, tail_, data_. Each on its own line so the
  // producer (writing head_) doesn't bounce the consumer's cache line.
  alignas(kCacheLine) std::atomic<uint64_t> head_{0};
  alignas(kCacheLine) std::atomic<uint64_t> tail_{0};
  alignas(kCacheLine) T data_[N];
};

}  // namespace gpuscope
