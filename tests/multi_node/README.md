<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors -->

# Two-node B1 qualification

Run the same pytest selection on both hosts, one process/GPU per host. This
directory is outside the default testpaths. Install the CuTe extra (DSL 4.8.0),
rdma-core development headers and runtime; allow pinned memory registration.
Set `CTI_TWO_NODE_RANK` to 0 or 1 and `CTI_TWO_NODE_STORE` to the rank-0
TCPStore address and port, then run:

```sh
PYTHONPATH=python python -m pytest -q -s tests/multi_node/test_two_node_allreduce.py
```

Both processes must start together with the same selection. Each invocation
creates a fresh store, a single provider and test-specific key prefixes.
The suite exchanges correctness flags and hashes so a mismatch fails both
ranks. It covers all three dtypes, eager sizes through 256 KiB, graph replay
with mixed sizes and eager interleaving, in-place output, reference fallback, the both-ranks
NCCL fallback when one host lacks libibverbs, asymmetric local layouts
(rank 0 misaligned, strided or with a staged output, rank 1 aligned; eager and
graph replay) that must stay on the same transport,
adaptive idle CPU use and extension symbols. Optional NCCL comparison uses
a store-backed gloo group and reports a skip reason if unavailable. If a
host name resolves to a loopback address, set `GLOO_SOCKET_IFNAME` and
`NCCL_SOCKET_IFNAME` to the interface that reaches the other host. The
proxy builds against `infiniband/verbs.h` from the system include directory
(or `CTI_VERBS_INCLUDE`), searched after the compiler's own sysroot.

Calls have collective ordering requirements. A missing peer or link failure
can leave the other GPU waiting; the local abort flag handles local proxy
errors only. Teardown follows a store barrier, device synchronization, host
unregistration, proxy join/drain and memory release. Do not interrupt one
rank while the other is issuing collectives.

## GDN decode under TP2

`test_two_node_gdn_decode.py` runs the Qwen3.5-9B GDN layer's decode chain at TP2, with the same
`CTI_TWO_NODE_*` setup: column-parallel in_proj, the decode on each rank's (8, 16) head shard,
row-parallel out_proj and the two-node all-reduce. It checks three things:
- the shard decode is bitwise the full (16, 32) decode's slice (batch-invariant mode);
- the all-reduced chain is identical on both ranks and within BF16 rounding of the single-GPU chain;
- a CUDA-graph replay of the chain equals the eager chain bitwise.
