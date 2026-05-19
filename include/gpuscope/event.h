// Event schema.
//
// One fixed-size record covers every kind of event we care about: CUDA
// kernel launches, memcpy operations, stream syncs, CPU callback enters/
// exits, context switches. Fixed-size matters because:
//   1. The ring buffer slot is sizeof(Event), so the producer can copy by
//      value without indirection.
//   2. Two processes sharing the buffer don't need a shared allocator.
//   3. The 64-byte size means each event lives in exactly one cache line.
//
// Design note: we explicitly do NOT use std::string or std::variant or
// anything with heap allocation. The producer is on the hot path of a
// CUDA workload; every malloc/free here would invalidate the project.
//
// Source attribution: the source_ field distinguishes synthetic events
// from real CUPTI/perf events so downstream consumers (UI, alerting) can
// treat them differently.

#pragma once

#include <cstdint>
#include <cstring>

namespace gpuscope {

enum class EventType : uint16_t {
  // CUDA (produced by CUPTI Activity API in real builds, or synthetic
  // generator in dev builds)
  kCudaKernelLaunch = 1,   // kernel queued on stream
  kCudaKernelStart  = 2,   // kernel began executing on device
  kCudaKernelEnd    = 3,   // kernel completed on device
  kCudaMemcpyHtoD   = 4,
  kCudaMemcpyDtoH   = 5,
  kCudaStreamSync   = 6,   // cudaStreamSynchronize returned

  // CPU (produced by perf_event_open in real builds)
  kCpuCallbackEnter = 100, // user-instrumented callback entry
  kCpuCallbackExit  = 101,
  kCpuContextSwitch = 102, // sched_switch tracepoint
};

enum class EventSource : uint8_t {
  kSynthetic = 0,
  kCupti     = 1,
  kPerf      = 2,
  kManual    = 3,  // user code calling gpuscope::emit() directly
};

// 64 bytes exactly. Verify with static_assert below.
struct Event {
  uint64_t   timestamp_ns;    // CLOCK_MONOTONIC at event time          (8)
  uint64_t   duration_ns;     // 0 for instantaneous events             (16)
  uint64_t   correlation_id;  // links launch -> start -> end           (24)
  uint32_t   tid;             // CPU thread id, or 0 for pure-GPU       (28)
  uint32_t   stream_id;       // CUDA stream id, or 0 for CPU events    (32)
  EventType  type;            //                                        (34)
  uint16_t   flags;           // reserved                               (36)
  EventSource source;         //                                        (37)
  uint8_t    _pad[3];         // pad name to 8-byte boundary            (40)
  char       name[24];        // truncated kernel / callback name       (64)

  // Convenience constructor used by both synthetic and real producers.
  static Event make(EventType t, uint64_t ts_ns, uint64_t dur_ns,
                    uint32_t stream, uint32_t thread,
                    uint64_t corr, const char* nm,
                    EventSource src = EventSource::kSynthetic) {
    Event e{};
    e.timestamp_ns = ts_ns;
    e.duration_ns = dur_ns;
    e.correlation_id = corr;
    e.type = t;
    e.source = src;
    e.stream_id = stream;
    e.tid = thread;
    if (nm) {
      std::strncpy(e.name, nm, sizeof(e.name) - 1);
      e.name[sizeof(e.name) - 1] = '\0';
    }
    return e;
  }
};

static_assert(sizeof(Event) == 64,
              "Event must be exactly one cache line. If you changed the "
              "schema, adjust the padding fields above.");
static_assert(alignof(Event) <= 8, "Event should not over-align.");

// Helper to stringify EventType for logs.
inline const char* event_type_name(EventType t) {
  switch (t) {
    case EventType::kCudaKernelLaunch:  return "cuda.kernel_launch";
    case EventType::kCudaKernelStart:   return "cuda.kernel_start";
    case EventType::kCudaKernelEnd:     return "cuda.kernel_end";
    case EventType::kCudaMemcpyHtoD:    return "cuda.memcpy_h2d";
    case EventType::kCudaMemcpyDtoH:    return "cuda.memcpy_d2h";
    case EventType::kCudaStreamSync:    return "cuda.stream_sync";
    case EventType::kCpuCallbackEnter:  return "cpu.callback_enter";
    case EventType::kCpuCallbackExit:   return "cpu.callback_exit";
    case EventType::kCpuContextSwitch:  return "cpu.context_switch";
  }
  return "unknown";
}

}  // namespace gpuscope
