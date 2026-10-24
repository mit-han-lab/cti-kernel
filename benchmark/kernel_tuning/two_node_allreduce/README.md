<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors -->

# B1 paired collective benchmark

Run `bench.py` simultaneously on both hosts with `--rank 0` / `--rank 1`
and the same `--store` rank-0 TCPStore address and port. Use a new store per
invocation. Requires two integrated sm_12x GPUs, CuTe DSL 4.8.0, rdma-core
headers/runtime, pinned-memory allowance, and the pynccl extension.

```sh
PYTHONPATH=python python benchmark/kernel_tuning/two_node_allreduce/bench.py --help
```

Options: `--op all_reduce|all_gather` defaults to `all_reduce`. `--sizes` takes
per-rank byte counts (all-reduce defaults: 2, 8, 16, 24, 32, 64, 128, 256 KiB;
all-gather defaults: 129280, 258560, 387840, 517120, DSV4 Flash TP2 BF16 logits
at B=1..4). `--pairs` defaults to 60 and cannot be smaller; `--out` writes
rank-0 JSON including every pair. `--ctas` and `--threads` sweep the grid;
use the same options on both hosts.

The process pins to available big cores excluding the proxy core. NCCL uses
`max_size_bytes=0` (in-place all-reduce, as PyNccl dispatches on this pair),
`NCCL_MAX_NCHANNELS=2`, and inherits that affinity. B1 uses its default adaptive
proxy, armed during throughput samples. Each eager launch sequence or graph
contains 43 all-reduce calls or one all-gather call. All-gather compares
`provider.all_gather(x, out)` with pynccl `all_gather(output, input)`, still
initialized with `max_size_bytes=0`. Provider capacity covers the largest size.
CUDA events time K sequences; calibration sets K per
implementation near 3 ms. Each sample has a store barrier and retains the
slower rank's time. Order alternates NCCL/B1 and B1/NCCL per pair. The ratio
is NCCL/B1 within each pair; a seeded 10,000-resample paired bootstrap gives
the percentile 95% CI of its median. Pass means CI_low > 1.

Random BF16 correctness checks compare bits to the rank-order FP32 reference
and NCCL, and compare maximum/RMS error to an FP64 oracle. Timed buffers are
zero so repeated in-place sums cannot overflow. JSON also contains idle proxy
CPU fraction over two seconds and 15 single-call wake samples at each of
1/20/100 ms, with and without arm. Wake measurements include inter-host skew
after the store barrier; inspect them as end-to-end observations.
All-gather checks bitwise bytes against NCCL and the expected rank-major
concatenation. It uses the same eager/graph pairing and CI, and skips idle CPU
and wake measurements (`idle_cpu: null`, `wake_penalty: []` in JSON).

No environment variables are
printed by this script. Teardown requires both ranks to finish.
