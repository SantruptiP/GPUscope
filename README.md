# gpuscope

Real-time GPU/CPU contention profiler for CUDA workloads.

Captures fine-grained events from CUDA (via CUPTI) and the CPU (via
`perf_event_open`), correlates them on a unified monotonic timeline, and
detects contention patterns live — telling you not just *what's slow* but
*what's fighting whom*.

Designed to run alongside production AV/robotics workloads with sub-1%
overhead. Lock-free shared-memory IPC between the instrumented process and
the aggregator. Composable detector rules. Built for the "why is my 30Hz
perception loop missing deadline 2% of the time" question that standard
profilers (Nsight Systems, Nsight Compute) can't answer because they're
post-hoc and heavyweight.

## Current status

This commit ships the **spine** — the parts that don't depend on a GPU:

- Lock-free SPSC ring buffer (`include/gpuscope/ring_buffer.h`)
- 64-byte unified event schema (`include/gpuscope/event.h`)
- Detector interface (`include/gpuscope/detector.h`)
- First detector: GPU-idle-while-CPU-busy
- Synthetic event generator that mimics a perception pipeline, with
  pluggable contention-injection modes
- End-to-end demo binary (`gpuscope --mode=cpu_bottleneck`)
- 8 unit tests, all passing under both -O2 and ThreadSanitizer

What's intentionally not here yet (the next milestone):

- CUPTI Activity/Callback API integration (the real CUDA producer)
- `perf_event_open` integration (the real CPU producer)
- Cross-process POSIX shared memory wrapper (right now the ring lives
  in-process; the API is the same when we move it to `shm_open`)
- More detectors: cross-stream serialization, PCIe direction contention,
  deadline-violation, kernel-launch-latency
- Prometheus / Perfetto export
- Live dashboard

## Build and run

Requires g++ 10+ or clang++ 12+. No external dependencies.

```bash
make            # build everything
make test       # 8 tests, all green
make tsan       # ring buffer under ThreadSanitizer (proves lock-free contract)
```

Run the demo:

```bash
./build/gpuscope --mode=clean --duration_ms=2000
# events_consumed=234 drops=0 findings=0
# no contention detected.

./build/gpuscope --mode=cpu_bottleneck --duration_ms=2000
# events_consumed=234 drops=0 findings=18
# [warn] gpu_idle_cpu_busy: GPU idle for 20140us while CPU thread 12345
#        ran callback 'postprocess' for 20072us
#   hint: Move CPU work off the GPU-launch critical path, or issue the
#         next kernel before the callback returns. Check Nsight Systems
#         for the gap between kernel end and the next launch.
```

Flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--mode=` | `clean` | `clean` or `cpu_bottleneck`; what contention to inject |
| `--duration_ms=` | `2000` | how long to run the synthetic generator |
| `--threshold_us=` | `5000` | detector threshold for "GPU idle too long" |

## Architecture (quick version)

```
[Instrumented app]                       [Aggregator process]
       │                                          │
       ▼                                          ▼
[CUPTI Activity callback]            [Detector pipeline]
[perf_event_open ringbuf]                          │
       │                                  ┌────────┴────────┐
       ▼                                  ▼                 ▼
[gpuscope::Event]               [GpuIdleCpuBusy]    [StreamSerialize]
       │                                  │                 │
       ▼                                  └────────┬────────┘
[SpscRingBuffer in /dev/shm] ─────► [drain] ◄──────┘
                                            │
                                            ▼
                                  Findings → Prometheus,
                                             Perfetto export,
                                             stdout report
```

The full design rationale (why SPSC vs MPMC, why 64-byte events, how
device timestamps map to CLOCK_MONOTONIC, etc.) lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Where this fits

This is a "real-time observability for GPU pipelines" project. The GPU CI
Debug Lab (separate repo) is the offline counterpart — it consumes
`gpuscope` output and gates PRs on it. Together they cover the
"detect contention live in production" and "prevent contention from
regressing across PRs" stories.
