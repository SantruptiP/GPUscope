// GpuIdleCpuBusy detector.
//
// Pattern: a CPU thread is inside a user callback (CallbackEnter without a
// matching CallbackExit) for longer than kThresholdMs, AND the GPU has not
// completed any work in the same window.
//
// Why this matters: in AV/robotics pipelines, the CPU thread that owns the
// frame's lifecycle (decode -> launch perception -> wait -> launch planning)
// frequently does too much CPU work between kernel launches. The GPU sits
// idle. The "natural" fix is to move the CPU work off the critical path or
// to start the next kernel earlier — but the bug is invisible without
// telling someone "your GPU was idle for 8ms while thread 42 was running."

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpuscope/detector.h"
#include "gpuscope/event.h"

namespace gpuscope {

class GpuIdleCpuBusyDetector : public Detector {
 public:
  // Threshold for "callback ran too long" and "GPU idle too long". Both
  // measured against the same wall clock.
  explicit GpuIdleCpuBusyDetector(uint64_t threshold_ns = 5'000'000)
      : threshold_ns_(threshold_ns) {}

  const char* name() const override { return "gpu_idle_cpu_busy"; }

  void on_event(const Event& ev, std::vector<Finding>& out) override {
    switch (ev.type) {
      case EventType::kCudaKernelEnd:
      case EventType::kCudaMemcpyHtoD:
      case EventType::kCudaMemcpyDtoH:
        // GPU did something. Update the "last GPU activity" marker.
        last_gpu_activity_ns_ = ev.timestamp_ns + ev.duration_ns;
        break;

      case EventType::kCpuCallbackEnter:
        // Record that this thread is in a callback. Note: this is single-
        // level only — nested callbacks on the same thread overwrite. Real
        // detector would track a stack; that's a known limitation.
        callback_enter_[ev.tid] = {ev.timestamp_ns, ev.name};
        break;

      case EventType::kCpuCallbackExit: {
        auto it = callback_enter_.find(ev.tid);
        if (it == callback_enter_.end()) break;
        const uint64_t cb_start = it->second.first;
        const std::string& cb_name = it->second.second;
        const uint64_t cb_end = ev.timestamp_ns;
        const uint64_t cb_dur = cb_end - cb_start;
        callback_enter_.erase(it);

        if (cb_dur < threshold_ns_) break;
        // Callback ran long enough to be interesting. Did the GPU do
        // anything in that window?
        if (last_gpu_activity_ns_ >= cb_start) break;  // GPU was active

        const uint64_t idle_dur = cb_end - last_gpu_activity_ns_;
        if (idle_dur < threshold_ns_) break;

        Finding f;
        f.timestamp_ns = cb_end;
        f.window_start_ns = cb_start;
        f.window_end_ns = cb_end;
        f.severity = idle_dur > 10 * threshold_ns_ ? Severity::kCritical
                                                   : Severity::kWarn;
        f.detector_name = name();
        f.description = "GPU idle for " + std::to_string(idle_dur / 1000)
                      + "us while CPU thread " + std::to_string(ev.tid)
                      + " ran callback '" + cb_name + "' for "
                      + std::to_string(cb_dur / 1000) + "us";
        f.hint = "Move CPU work off the GPU-launch critical path, or "
                 "issue the next kernel before the callback returns. Check "
                 "Nsight Systems for the gap between kernel end and the "
                 "next launch.";
        out.push_back(std::move(f));
        break;
      }

      default:
        break;
    }
  }

  void on_flush(std::vector<Finding>& /*out*/) override {
    // Could emit "stuck callback" findings here. Skipped for v1.
  }

 private:
  uint64_t threshold_ns_;
  uint64_t last_gpu_activity_ns_ = 0;
  // tid -> (entry timestamp, callback name)
  std::unordered_map<uint32_t, std::pair<uint64_t, std::string>>
      callback_enter_;
};

}  // namespace gpuscope
