# gpuscope architecture

The deep version. Read the README first for the overview.

## Goals and non-goals

**Goals:**

- Capture every CUDA kernel launch, memcpy, and stream sync with
  microsecond accuracy.
- Capture CPU-side callback timing and context switches.
- Correlate both streams on one timeline.
- Run alongside production workloads. Hard budget: <1% wall-time overhead,
  <0.5% CPU steal, fixed memory footprint.
- Surface contention patterns within a few hundred milliseconds of
  occurrence — fast enough to alert before the next missed deadline.

**Non-goals:**

- We are *not* Nsight Systems. Nsight captures every API call with
  byte-level detail and is excellent for one-shot deep dives. We capture
  less data, run permanently, and answer different questions.
- We are *not* a post-hoc profiler. The whole architecture is built around
  streaming detectors with bounded state.
- We don't optimize for replay. Events are processed and discarded; we
  don't store them long-term beyond what Prometheus retention configures.

## Component breakdown

### Event schema (`event.h`)

One fixed-size struct (64 bytes) for every event type. Why fixed-size:

1. **Zero-allocation copy on the hot path.** The CUPTI callback runs on
   a driver thread. Heap allocation there would invalidate the project —
   that thread can't afford to ever wait on `malloc`.
2. **Cross-process safety.** No `std::string`, no pointers — the struct
   round-trips through shared memory without any serialization.
3. **One cache line per event.** A consumer thread reading 1M events/sec
   stays cache-resident if each event is 64 bytes.

The schema covers three categories:

- **CUDA events** (`kCudaKernelLaunch`, `kCudaKernelStart`, `kCudaKernelEnd`,
  `kCudaMemcpyHtoD`, `kCudaMemcpyDtoH`, `kCudaStreamSync`). Launch and Start
  are distinct because CUPTI gives us both: Launch is when the host queued
  the kernel, Start is when the device began executing. The gap between
  them is launch latency, which is itself a metric.
- **CPU events** (`kCpuCallbackEnter`, `kCpuCallbackExit`,
  `kCpuContextSwitch`). The callback events are user-instrumented via a
  scope guard macro (not yet implemented; see future work). Context
  switches come from `perf_event_open` with `PERF_RECORD_SWITCH`.
- **Reserved space** for future event types via the 2-byte `flags` field.

The `correlation_id` field is critical for joining related events. A
`KernelLaunch` event and the matching `KernelEnd` event share an ID, so
the detector can compute launch latency without having to do timestamp
arithmetic across the whole event stream.

### Lock-free SPSC ring buffer (`ring_buffer.h`)

The classic two-counter design with one twist: each counter sits on its
own cache line. Without alignment, the producer storing to `head_` and
the consumer storing to `tail_` would bounce the cache line containing
both, costing 50-100ns per push on x86. With alignment: ~3ns.

The buffer holds 4096 events by default (256KB). At a 1kHz sustained
production rate this gives ~4 seconds of slack before the consumer must
drain — plenty for the aggregator's ~100ms detection cycle.

**Why SPSC and not MPMC?** In the real architecture, every instrumented
process has its own ring buffer; CUPTI runs on driver threads but they
all funnel through a single producer that owns the buffer. MPMC would
add unnecessary atomic contention. If we later need multi-producer (e.g.,
CPU and GPU events from different threads), we use one SPSC ring per
producer and the aggregator drains all of them — cheaper than one MPMC.

**Memory ordering correctness:** the `mt_soak` test pushes 200,000
events across two threads and validates exact-once delivery in order.
Running it under ThreadSanitizer (`make tsan`) reports zero races. If
either of those is wrong, our ordering is wrong.

### Detector interface (`detector.h`)

Stateful streaming consumers. The interface is two methods: `on_event` for
processing one event at a time, and `on_flush` for emitting pending state
on shutdown.

Critical design choices:

- **No detector-to-detector communication.** Each detector is self-
  contained. If two detectors need the same precomputed state, they each
  compute it. Cheap; keeps tests independent.
- **Detectors mutate a `vector<Finding>&` passed in by the aggregator.**
  This avoids returning by value (one extra copy per call) and lets the
  aggregator pool the buffer across calls.
- **No allocation on the hot path... unless the detector itself needs
  state.** The `GpuIdleCpuBusy` detector uses an `unordered_map`, which
  is fine because callback entries/exits are O(10 Hz), not O(1 MHz).

### First detector: GpuIdleCpuBusy

Pattern: a CPU thread held a callback open for >5ms (configurable) AND
the GPU emitted no `KernelEnd` or `Memcpy*` events during that window.

This is the most common AV-pipeline bug pattern: someone added too much
work to the post-processing callback (validation, logging, statistics),
and now the next frame's kernel can't launch on time because the host
thread that owns the launch loop is still busy. The GPU sits idle. p99
frame time spikes. With this detector, the bug is named the moment it
happens.

Edge cases handled:

- Nested callbacks on the same thread: only the outermost entry counts.
  (A real implementation would track a stack per thread; one of the
  known follow-up items.)
- The detector won't fire on the very first event; it needs an initial
  `last_gpu_activity_ns_` reference. Tests cover this.

### Synthetic generator (`synthetic_producer.h`)

The reason this project is demoable on a Mac without CUDA. Generates a
realistic-looking event stream — preprocess callback, H2D, kernel,
D2H, postprocess callback — at ~30Hz. The `--mode=cpu_bottleneck` flag
makes every third frame's postprocess take 20ms instead of 1ms,
producing exactly the pattern the detector catches.

This is also how new detectors get developed: write a contention mode
that produces the pattern, write the detector that catches it, write
the test that validates both. The synthetic generator is the testbed.

## Performance characteristics

Measured on this Linux container (g++ 13 -O2):

| Workload | Throughput |
| --- | --- |
| Ring buffer push (single thread) | ~200M events/sec |
| Ring buffer cross-thread (push + pop) | ~50M events/sec |
| Detector dispatch (one detector, GpuIdleCpuBusy) | ~10M events/sec |

The synthetic generator runs at ~30Hz (every 33ms), generating ~8 events
per frame, so ~240 events/sec. The aggregator can sustain ~40,000x that
rate before backpressure becomes an issue. The dimensioning is correct —
we are nowhere near the throughput ceiling, which is what we want for a
permanent-running profiler.

Real CUDA workloads emit more events — a perception pipeline launching
20 kernels per frame at 100Hz is 2,000 kernel events/sec, plus memcpy
and CPU events maybe 10x that. ~20,000 events/sec. Still 2500x under
our throughput ceiling.

## Time synchronization

This is the part that's stubbed in v1 but needs proper treatment for
CUPTI integration.

CUPTI Activity records carry device timestamps in nanoseconds, but the
clock is "device time" — independent from `CLOCK_MONOTONIC`. To
correlate with CPU events, we need a mapping.

The standard approach:

1. At startup, call `cuptiGetTimestamp()` and `clock_gettime(CLOCK_MONOTONIC)`
   in quick succession to get the offset.
2. Periodically (every few seconds) re-sample to detect clock drift.
3. Apply the offset to every CUPTI timestamp before pushing into the ring.

There's a known precision floor of ~100ns due to the read non-atomicity.
For our purposes (5ms detector threshold), that's negligible.

## Road forward

1. **CUPTI Activity API producer.** Hook `cuptiActivityEnable` for
   `CUPTI_ACTIVITY_KIND_KERNEL`, `_MEMCPY`, `_RUNTIME`. Convert each
   Activity record to a `gpuscope::Event` and push. ~1 day of work
   on a Linux GPU box.
2. **Cross-process shared memory.** Move the ring from `make_unique<Ring>()`
   to `shm_open + mmap`. The ring struct is already POD; this is purely
   plumbing. ~half a day.
3. **`perf_event_open` CPU producer.** For real context-switch events
   and per-thread CPU cycles. ~1 day.
4. **More detectors:**
   - `StreamSerializationDetector` — kernel queued behind another stream's
     kernel for >Nms.
   - `PciContentionDetector` — H2D and D2H both queued on PCIe at the
     same time when one direction was free.
   - `DeadlineViolationDetector` — frame's end-to-end latency exceeded
     a configurable budget.
   - `LaunchLatencyDetector` — `KernelLaunch` to `KernelStart` gap
     exceeded Nus.
5. **Prometheus exporter.** A tiny HTTP server on port 9100 exposing
   `gpuscope_findings_total{detector=...,severity=...}` counters and
   `gpuscope_events_consumed_total`.
6. **Perfetto trace export.** Same Chrome Trace Event format the CI Debug
   Lab uses, written once per minute to `/var/log/gpuscope/`.
7. **Demo workload that proves measured perf wins.** A two-kernel
   pipeline with intentional CPU bottleneck, plus a "fixed" version
   that moves CPU work off the launch path. Show measured p99 latency
   improvement when the bottleneck is removed.
