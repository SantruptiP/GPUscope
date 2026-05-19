// Detector interface.
//
// A detector consumes events in chronological order and emits Findings when
// it spots a pattern. Detectors are stateful — they accumulate context as
// events stream in.
//
// Why a streaming interface and not a batched one: we want detectors to be
// able to run in the aggregator's hot loop with ~no allocation. Batched
// detectors (collect N events, then analyze) have a latency floor of N
// events, which makes them useless for sub-100ms detection.
//
// Composability: detectors don't talk to each other. The aggregator fans
// each event out to every registered detector. If two detectors need the
// same precomputed state (e.g., both need "is the GPU currently busy"),
// they each compute it. Cheap; keeps detectors testable in isolation.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpuscope/event.h"

namespace gpuscope {

enum class Severity : uint8_t {
  kInfo = 0,
  kWarn = 1,
  kCritical = 2,
};

struct Finding {
  uint64_t timestamp_ns;     // when the pattern was detected
  uint64_t window_start_ns;  // the time window the pattern spans
  uint64_t window_end_ns;
  Severity severity;
  std::string detector_name;
  std::string description;   // human-readable summary
  std::string hint;          // "where to look next"
};

class Detector {
 public:
  virtual ~Detector() = default;

  // Process one event. Append any Findings produced to `out`.
  // The aggregator owns `out` and drains it between event batches.
  virtual void on_event(const Event& ev, std::vector<Finding>& out) = 0;

  // Called when the aggregator is shutting down. Lets the detector emit
  // any "still-pending" Findings (e.g., a callback that entered but never
  // exited).
  virtual void on_flush(std::vector<Finding>& out) = 0;

  virtual const char* name() const = 0;
};

}  // namespace gpuscope
