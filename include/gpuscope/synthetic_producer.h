// Synthetic event generator.
//
// Produces a realistic stream of events mimicking a perception pipeline:
//   - Frame arrives on the CPU side (CallbackEnter)
//   - Pre-processing on CPU
//   - Kernel launch -> kernel start -> kernel end on stream 0
//   - Maybe another kernel on stream 1 (head split for detection + seg)
//   - D2H copy
//   - Post-processing callback
//
// The interesting bit is the contention injection modes. With --mode=clean,
// the pipeline runs normally. With --mode=cpu_bottleneck, the post-processing
// callback intentionally takes 20ms (way too long), so the next frame's
// kernel can't launch on time -- a textbook CPU-bottlenecks-GPU bug that
// the GpuIdleCpuBusy detector should catch.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>

#include "gpuscope/event.h"
#include "gpuscope/ring_buffer.h"

namespace gpuscope {

enum class ContentionMode {
  kClean,            // no injected contention
  kCpuBottleneck,    // CPU callback takes 20ms occasionally
  kStreamSerialize,  // (future) two streams compete
};

inline uint64_t monotonic_ns() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clk::now().time_since_epoch())
      .count();
}

// One frame worth of events. Pushes into `ring`; if full, drops the event
// (real producer would have a metrics counter for drops).
template <typename Ring>
inline void emit_frame(Ring& ring, ContentionMode mode, uint32_t frame_id,
                       uint32_t tid, std::mt19937& rng) {
  std::uniform_int_distribution<int> jitter(0, 200);  // microseconds
  const uint64_t t0 = monotonic_ns();

  auto try_push = [&](Event e) {
    // Spin-retry briefly; in real producer we'd just drop.
    for (int i = 0; i < 4 && !ring.try_push(e); ++i) {
      std::this_thread::yield();
    }
  };

  // CPU callback: pre-processing (1ms + jitter).
  try_push(Event::make(EventType::kCpuCallbackEnter, t0, 0, 0, tid,
                       frame_id, "preprocess"));
  const uint64_t preproc_dur = 1'000'000 + jitter(rng) * 1000;
  std::this_thread::sleep_for(std::chrono::nanoseconds(preproc_dur));
  uint64_t t1 = monotonic_ns();
  try_push(Event::make(EventType::kCpuCallbackExit, t1, 0, 0, tid,
                       frame_id, "preprocess"));

  // CUDA: H2D copy.
  try_push(Event::make(EventType::kCudaMemcpyHtoD, t1, 800'000, 0, 0,
                       frame_id, "input_h2d"));
  std::this_thread::sleep_for(std::chrono::microseconds(800));
  const uint64_t t2 = monotonic_ns();

  // CUDA: kernel launch + start + end on stream 0.
  try_push(Event::make(EventType::kCudaKernelLaunch, t2, 0, 1, tid,
                       frame_id, "perception_backbone"));
  // Simulate ~30us of launch latency.
  std::this_thread::sleep_for(std::chrono::microseconds(30));
  const uint64_t t3 = monotonic_ns();
  try_push(Event::make(EventType::kCudaKernelStart, t3, 0, 1, 0,
                       frame_id, "perception_backbone"));
  // Kernel runtime ~3ms.
  std::this_thread::sleep_for(std::chrono::microseconds(3000));
  const uint64_t t4 = monotonic_ns();
  try_push(Event::make(EventType::kCudaKernelEnd, t4, 3'000'000, 1, 0,
                       frame_id, "perception_backbone"));

  // D2H.
  try_push(Event::make(EventType::kCudaMemcpyDtoH, t4, 400'000, 1, 0,
                       frame_id, "output_d2h"));
  std::this_thread::sleep_for(std::chrono::microseconds(400));
  uint64_t t5 = monotonic_ns();

  // Post-processing callback. In bottleneck mode, every 3rd frame takes
  // a long time on purpose -- the kind of bug the detector should catch.
  try_push(Event::make(EventType::kCpuCallbackEnter, t5, 0, 0, tid,
                       frame_id, "postprocess"));
  uint64_t postproc_us = 800 + jitter(rng);
  if (mode == ContentionMode::kCpuBottleneck && (frame_id % 3 == 0)) {
    postproc_us = 20'000;  // 20ms -- way too long
  }
  std::this_thread::sleep_for(std::chrono::microseconds(postproc_us));
  const uint64_t t6 = monotonic_ns();
  try_push(Event::make(EventType::kCpuCallbackExit, t6, 0, 0, tid,
                       frame_id, "postprocess"));
}

}  // namespace gpuscope
