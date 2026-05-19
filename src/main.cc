// gpuscope main binary.
//
// Spawns a producer thread that runs the synthetic generator and a consumer
// thread that pulls events off the ring buffer and feeds them to detectors.
// After --duration_ms, joins everything and prints findings.
//
// This is the dev mode that works on a Mac without CUDA. The real CUPTI
// producer (not yet wired in this commit) plugs into the same ring buffer.
//
// Usage:
//   gpuscope --mode=clean        --duration_ms=2000
//   gpuscope --mode=cpu_bottleneck --duration_ms=5000

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gpuscope/detector.h"
#include "gpuscope/detectors/gpu_idle_cpu_busy.h"
#include "gpuscope/event.h"
#include "gpuscope/ring_buffer.h"
#include "gpuscope/synthetic_producer.h"

namespace {

constexpr size_t kRingSize = 4096;  // 4K events ~= 256KB. Plenty for 1kHz.
using Ring = gpuscope::SpscRingBuffer<gpuscope::Event, kRingSize>;

struct Args {
  std::string mode = "clean";
  int duration_ms = 2000;
  uint64_t threshold_us = 5000;  // 5ms default
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto eq = s.find('=');
    if (eq == std::string::npos) continue;
    std::string k = s.substr(0, eq);
    std::string v = s.substr(eq + 1);
    if (k == "--mode") a.mode = v;
    else if (k == "--duration_ms") a.duration_ms = std::stoi(v);
    else if (k == "--threshold_us") a.threshold_us = std::stoul(v);
  }
  return a;
}

gpuscope::ContentionMode mode_from_str(const std::string& s) {
  if (s == "cpu_bottleneck") return gpuscope::ContentionMode::kCpuBottleneck;
  return gpuscope::ContentionMode::kClean;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse(argc, argv);

  // Ring lives on the heap because it's 256KB+ with cache alignment --
  // bigger than typical stack budget. In the cross-process variant this
  // would be the mmap'd shared memory region.
  auto ring = std::make_unique<Ring>();

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> events_produced{0};
  std::atomic<uint64_t> events_consumed{0};
  std::atomic<uint64_t> drops{0};

  // Producer thread: synthetic generator running until stop is signaled.
  std::thread producer([&] {
    std::mt19937 rng(42);
    uint32_t frame_id = 0;
    const auto mode = mode_from_str(args.mode);
    const uint32_t tid =
        static_cast<uint32_t>(std::hash<std::thread::id>{}(
            std::this_thread::get_id()));
    while (!stop.load(std::memory_order_relaxed)) {
      const size_t before = ring->size_approx();
      gpuscope::emit_frame(*ring, mode, frame_id++, tid, rng);
      const size_t after = ring->size_approx();
      events_produced.fetch_add(after >= before ? after - before : 0,
                                std::memory_order_relaxed);
      // Frame interval: aim for ~30Hz nominal. The 20ms bottleneck eats
      // into this in cpu_bottleneck mode -- that's the point.
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
  });

  // Consumer thread: drain ring, feed detectors.
  std::vector<std::unique_ptr<gpuscope::Detector>> detectors;
  detectors.push_back(std::make_unique<gpuscope::GpuIdleCpuBusyDetector>(
      args.threshold_us * 1000));

  std::vector<gpuscope::Finding> findings;
  std::thread consumer([&] {
    gpuscope::Event ev;
    while (!stop.load(std::memory_order_relaxed)) {
      if (ring->try_pop(ev)) {
        events_consumed.fetch_add(1, std::memory_order_relaxed);
        for (auto& d : detectors) d->on_event(ev, findings);
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    // Drain remaining events on shutdown.
    while (ring->try_pop(ev)) {
      events_consumed.fetch_add(1, std::memory_order_relaxed);
      for (auto& d : detectors) d->on_event(ev, findings);
    }
    for (auto& d : detectors) d->on_flush(findings);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(args.duration_ms));
  stop.store(true, std::memory_order_relaxed);
  producer.join();
  consumer.join();

  std::cout << "events_consumed=" << events_consumed.load()
            << " drops=" << drops.load()
            << " findings=" << findings.size() << "\n";

  if (findings.empty()) {
    std::cout << "no contention detected.\n";
    return 0;
  }

  std::cout << "\n--- findings ---\n";
  for (const auto& f : findings) {
    const char* sev = "info";
    if (f.severity == gpuscope::Severity::kWarn) sev = "warn";
    if (f.severity == gpuscope::Severity::kCritical) sev = "CRITICAL";
    std::cout << "[" << sev << "] " << f.detector_name << ": "
              << f.description << "\n"
              << "    hint: " << f.hint << "\n\n";
  }
  return findings.empty() ? 0 : 1;
}
