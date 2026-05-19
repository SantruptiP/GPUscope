// Tests for GpuIdleCpuBusyDetector.
//
// Strategy: hand-craft event sequences with known timing patterns. Don't
// rely on the synthetic generator -- we want deterministic timestamps.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpuscope/detector.h"
#include "gpuscope/detectors/gpu_idle_cpu_busy.h"
#include "gpuscope/event.h"

using gpuscope::Event;
using gpuscope::EventType;
using gpuscope::Finding;
using gpuscope::GpuIdleCpuBusyDetector;

#define EXPECT(cond) do { \
    if (!(cond)) { \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::abort(); \
    } \
  } while (0)

constexpr uint64_t kMs = 1'000'000;  // ns per millisecond
constexpr uint32_t kTid = 1234;

// Helper: feed events to the detector, return findings.
std::vector<Finding> run(GpuIdleCpuBusyDetector& d,
                         const std::vector<Event>& events) {
  std::vector<Finding> out;
  for (const auto& e : events) d.on_event(e, out);
  d.on_flush(out);
  return out;
}

void test_clean_pipeline_no_findings() {
  // GPU does work every ~5ms, CPU callbacks are short (~1ms). Nothing fires.
  GpuIdleCpuBusyDetector d(/*threshold_ns=*/5 * kMs);
  std::vector<Event> evs;
  uint64_t t = 1'000'000'000;  // some arbitrary start
  for (int frame = 0; frame < 10; ++frame) {
    evs.push_back(Event::make(EventType::kCpuCallbackEnter, t, 0, 0, kTid,
                              frame, "cb"));
    evs.push_back(Event::make(EventType::kCudaKernelEnd, t + 2 * kMs,
                              3 * kMs, 1, 0, frame, "kernel"));
    evs.push_back(Event::make(EventType::kCpuCallbackExit,
                              t + 1 * kMs, 0, 0, kTid, frame, "cb"));
    t += 8 * kMs;
  }
  auto findings = run(d, evs);
  EXPECT(findings.empty());
  std::printf("  clean_pipeline_no_findings OK\n");
}

void test_cpu_bottleneck_fires() {
  // CPU thread enters callback, GPU does nothing for 20ms, then callback
  // exits. Detector should produce exactly one finding.
  GpuIdleCpuBusyDetector d(/*threshold_ns=*/5 * kMs);
  std::vector<Event> evs;
  uint64_t t = 1'000'000'000;
  // Initial GPU activity to set last_gpu_activity_ns_.
  evs.push_back(Event::make(EventType::kCudaKernelEnd, t, 1 * kMs,
                            1, 0, 0, "warmup"));
  t += 2 * kMs;
  // CPU callback enters, runs 20ms, exits. No GPU activity in between.
  evs.push_back(Event::make(EventType::kCpuCallbackEnter, t, 0, 0, kTid,
                            1, "bottleneck_cb"));
  t += 20 * kMs;
  evs.push_back(Event::make(EventType::kCpuCallbackExit, t, 0, 0, kTid,
                            1, "bottleneck_cb"));

  auto findings = run(d, evs);
  EXPECT(findings.size() == 1);
  EXPECT(findings[0].detector_name == "gpu_idle_cpu_busy");
  EXPECT(findings[0].description.find("bottleneck_cb") != std::string::npos);
  std::printf("  cpu_bottleneck_fires OK (%s)\n",
              findings[0].description.c_str());
}

void test_gpu_active_during_callback_no_finding() {
  // CPU callback runs 20ms but GPU was busy in the middle. No finding.
  GpuIdleCpuBusyDetector d(5 * kMs);
  std::vector<Event> evs;
  uint64_t t = 1'000'000'000;
  evs.push_back(Event::make(EventType::kCpuCallbackEnter, t, 0, 0, kTid,
                            1, "cb"));
  // GPU kernel completes in the middle of the callback.
  evs.push_back(Event::make(EventType::kCudaKernelEnd, t + 10 * kMs,
                            1 * kMs, 1, 0, 1, "kernel"));
  evs.push_back(Event::make(EventType::kCpuCallbackExit, t + 20 * kMs,
                            0, 0, kTid, 1, "cb"));
  auto findings = run(d, evs);
  EXPECT(findings.empty());
  std::printf("  gpu_active_during_callback_no_finding OK\n");
}

void test_short_callback_below_threshold() {
  // Callback runs 2ms which is below the 5ms threshold. No finding even
  // though GPU was idle.
  GpuIdleCpuBusyDetector d(5 * kMs);
  std::vector<Event> evs;
  uint64_t t = 1'000'000'000;
  evs.push_back(Event::make(EventType::kCudaKernelEnd, t, 0, 1, 0, 0, "k"));
  t += 1 * kMs;
  evs.push_back(Event::make(EventType::kCpuCallbackEnter, t, 0, 0, kTid,
                            1, "shortcb"));
  t += 2 * kMs;
  evs.push_back(Event::make(EventType::kCpuCallbackExit, t, 0, 0, kTid,
                            1, "shortcb"));
  auto findings = run(d, evs);
  EXPECT(findings.empty());
  std::printf("  short_callback_below_threshold OK\n");
}

int main() {
  std::printf("detector tests:\n");
  test_clean_pipeline_no_findings();
  test_cpu_bottleneck_fires();
  test_gpu_active_during_callback_no_finding();
  test_short_callback_below_threshold();
  std::printf("all detector tests passed.\n");
  return 0;
}
