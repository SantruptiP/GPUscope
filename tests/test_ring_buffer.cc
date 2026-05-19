// Tests for SpscRingBuffer.
//
// Single-threaded: basic push/pop, full/empty handling, wraparound.
// Multi-threaded: producer thread pushes N items, consumer thread pops them,
//                 every item is delivered in order, no duplicates, no losses.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "gpuscope/ring_buffer.h"

using gpuscope::SpscRingBuffer;

#define EXPECT(cond) do { \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::abort(); \
    } \
  } while (0)

void test_push_pop_basic() {
  SpscRingBuffer<int, 8> r;
  EXPECT(r.size_approx() == 0);

  int out = 0;
  EXPECT(!r.try_pop(out));  // empty
  EXPECT(r.try_push(42));
  EXPECT(r.size_approx() == 1);
  EXPECT(r.try_pop(out));
  EXPECT(out == 42);
  EXPECT(r.size_approx() == 0);
  std::printf("  push_pop_basic OK\n");
}

void test_full_returns_false() {
  SpscRingBuffer<int, 4> r;
  EXPECT(r.try_push(1));
  EXPECT(r.try_push(2));
  EXPECT(r.try_push(3));
  EXPECT(r.try_push(4));
  EXPECT(!r.try_push(5));  // full
  EXPECT(r.size_approx() == 4);
  std::printf("  full_returns_false OK\n");
}

void test_wraparound() {
  // Push and pop more items than the buffer holds; verify the index
  // masking works correctly across wraparound.
  SpscRingBuffer<int, 4> r;
  for (int i = 0; i < 100; ++i) {
    EXPECT(r.try_push(i));
    int out = -1;
    EXPECT(r.try_pop(out));
    EXPECT(out == i);
  }
  std::printf("  wraparound OK\n");
}

void test_mt_soak() {
  // Producer pushes 0..N-1; consumer pops; expect exactly N items in order.
  constexpr int N = 200'000;
  SpscRingBuffer<int, 1024> r;
  std::atomic<bool> done{false};

  std::thread prod([&] {
    for (int i = 0; i < N; ++i) {
      while (!r.try_push(i)) {
        std::this_thread::yield();
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::vector<int> received;
  received.reserve(N);
  int out = 0;
  while (true) {
    if (r.try_pop(out)) {
      received.push_back(out);
      if (static_cast<int>(received.size()) == N) break;
    } else if (done.load(std::memory_order_acquire) && r.size_approx() == 0) {
      break;
    } else {
      std::this_thread::yield();
    }
  }
  prod.join();

  EXPECT(static_cast<int>(received.size()) == N);
  for (int i = 0; i < N; ++i) EXPECT(received[i] == i);
  std::printf("  mt_soak OK (%d items)\n", N);
}

int main() {
  std::printf("ring_buffer tests:\n");
  test_push_pop_basic();
  test_full_returns_false();
  test_wraparound();
  test_mt_soak();
  std::printf("all ring_buffer tests passed.\n");
  return 0;
}
