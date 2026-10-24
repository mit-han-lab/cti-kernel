# hd256 VERIFY at GQA 16 and 64: CuTe vs FlashInfer (RTX 5090)

Scope of `HD256_GQA8_CUTE_ROWS` (dispatch.py). Paired per-layer graph chains of
DraftAttention with the provider forced (CuTe vs the FlashInfer device-schedule
provider), causal full mask, D=256, Q=16, capacity 32768, 10 layers of rotating K/V
(>= 2x L2). 20 interleaved candidate/reference pairs after 5 warm-ups; epsilon is the
p95 of 20 FlashInfer self-pairs. One process on one RTX 5090 (kernels/rc-0.1.13 tip
aeba9e39). maxdiff is the CuTe vs FlashInfer max abs output difference.

| heads (qo/kv) | B | ctx | CuTe us | FlashInfer us | median CuTe/FI | epsilon | maxdiff |
|---|---|---|---|---|---|---|---|
| 64/4 | 1 | 100 | 8.04 | 9.40 | 0.857 | 0.032 | 0.0078 |
| 64/4 | 1 | 1000 | 17.49 | 16.22 | 1.079 | 0.066 | 0.0020 |
| 64/4 | 3 | 100 | 8.68 | 14.45 | 0.602 | 0.049 | 0.0078 |
| 64/4 | 3 | 1000 | 40.45 | 28.43 | 1.426 | 0.080 | 0.0020 |
| 64/1 | 1 | 100 | 8.10 | 7.61 | 1.064 | 0.011 | 0.0039 |
| 64/1 | 1 | 1000 | 17.49 | 15.97 | 1.095 | 0.023 | 0.0020 |
| 64/1 | 3 | 100 | 8.60 | 14.54 | 0.593 | 0.012 | 0.0039 |
| 64/1 | 3 | 1000 | 40.56 | 28.09 | 1.439 | 0.031 | 0.0020 |

Rule (DF4): per (device, kv heads, B, Q) the provider with the smaller worst ratio
over contexts. 64/4 B=1: CuTe worst 1.079, FlashInfer worst 1.168 -> CuTe. 64/4 and
64/1 B=3: CuTe (1.43-1.44 at ctx 1000, FlashInfer 1.66-1.69 at ctx 100). 64/1 B=1:
FlashInfer is faster at both contexts (CuTe 1.06-1.10x); HD256_CUTE_ROWS still picks
CuTe there, a known provider-choice miss between two of our providers.
The GQA-8 table therefore applies to group 8 only (its measured shapes, 16/2 and 8/1).
