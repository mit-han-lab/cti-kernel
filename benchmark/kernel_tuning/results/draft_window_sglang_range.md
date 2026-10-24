# DraftAttention support range against SGLang (2026-10-16)

Rule: SGLang's range is the minimum. Every shape, length, head dim,
dtype, batch size and layout that SGLang's DFlash draft/verify path accepts, including the FlashInfer
kernel it calls, must be accepted by ours without a raise or a torch fallback.

SGLang path studied: `sglang/srt/layers/attention/flashinfer_backend.py` together with the DFlash model,
utils and info modules. Draft and target verify both call FlashInfer
`BatchPrefillWithPagedKVCacheWrapper` (FA2 backend). The installed FlashInfer is 0.6.11.post3; 0.7.0.post1
has the same AOT head-dim set. This is a static source audit. The pinned geometries are covered by GPU
tests (`tests/gpu/test_dflash_draft_window.py`, `tests/gpu/test_dflash_long_context.py`).

| Dimension | SGLang + FlashInfer FA2 | Ours (Draft CuTe / Draft FA2 / FixedQuery verify) | Gap | Dense Qwen3.5/3.8 hits it? | Plan |
|---|---|---|---|---|---|
| Context C | `context_len`, no fixed cap; actual CSR entries are int32 | B*C <= 2^31-1 on all; FA2 Q<=32 adds conservative schedule bounds (B=1, P=1 full rejects C > 2^30); Q>32 uses int64 intermediates | FA2 bounds on allocated capacity rather than live CSR | No (262,144 and 131,072 tested, B=1-4) | Bound FA2 intermediates in int64 / derive from live CSR; next round |
| Batch B | Request-pool capacity, no fixed cap | CuTe B <= 65535 (grid); FA2/verify no fixed cap | CuTe B > 65535; auto now checks configuration and routes to FA2 | No at the pinned B=1-4 | Construction-time fallback implemented with A4 |
| Query Q (block) | Any Q >= 1; verify also ragged Q | Draft and FixedQuery any fixed Q >= 1, C >= Q (provider grid/int32 bounds) | A4 closed; ragged/variable Q remains | No (Q=8/16) | Fixed-Q range lifted; `tests/gpu/test_draft_any_query.py` passes on the RTX 5090 and GB10 (PRO deferred: device not available) |
| Head dim | AOT (64,64),(128,128),(256,256); JIT other and Dqk != Dvo | 64/128/256, Dqk = Dvo | JIT-only dims and Dqk != Dvo | No | Add only when a target model needs it |
| GQA / MQA | Hq % Hkv == 0, any ratio | Same; MQA admitted on CuTe and FA2 | None (grid limits only) | No | None |
| Heads | grid-z = Hkv <= 65535 | CuTe Hkv*ceil(Q*G/rows) <= 65535 | CuTe extreme heads x GQA | No | Same construction-time FA2 route as B |
| Page size P | SGLang passes P=1; FA2 any positive P | Any positive P | None | No | None |
| q / KV dtype | q FP16/BF16; KV FP16/BF16/FP8 (e4m3, e5m2), q and KV types independent | One dtype for q/K/V/out, FP16/BF16 | FP8 KV cache, mixed q/KV dtypes | Only if the engine enables an FP8 KV cache (pinned pairs use BF16) | FP8-KV FA2 path first (FlashInfer supports it), then CuTe; ETA after the dense work |
| Window / causal | window_left -1 or >= 0; no right bound; sliding layers forced causal in the SGLang DFlash model | Any W >= 1 or None, causal or not; W >= C is full; CuTe and FA2 custom-mask noncausal masks are symmetric; plain FA2 rejects noncausal W < Q | A6 closed in implementation by `flashinfer-mask`; auto uses CuTe when supported and custom-mask FA2 otherwise, including off SM12x | No | SM120/SM121 tests use explicit custom-mask selection (`tests/gpu/test_draft_custom_mask.py`, passing on the RTX 5090 and GB10; PRO deferred); the noncausal sliding semantics differ from this SGLang checkout (see note) |
| Layout | NHD/HND paged KV, q row/head strides, strided KV | q contiguous or declared row stride (fused QKV); K/V NHD tuple, contiguous | HND, strided KV, undeclared q stride | No (NHD, declared stride) | HND/strided KV when the engine needs it |
| int32 indices | int32 CSR in SGLang; FA2 page offsets widen to size_t | Draft B*C, B*Q <= 2^31-1 (tested); FA2 extra intermediate bounds only for Q<=32 | Conservative small-Q FA2 bounds | No | With the C row |
| CUDA graphs | Draft and verify with fixed capacities; variable Q inside a max | Draft CuTe/FA2 and FixedQuery capture-safe; eager == graph tested | Variable-Q graphs | No | With the Q row |

Note on the noncausal sliding mask: this SGLang checkout forces sliding DFlash layers to causal, while
the pinned Qwen3.8-27B DFlash2 pair is noncausal W2048 (symmetric). Admission is unaffected, and our
FP32 reference defines the noncausal window as symmetric. The engine owns the model-level mask choice.

A6 custom-mask ABI was source-checked on FlashInfer 0.6.11.post3, 0.7.0.post1,
and 0.7.1. The provider uses a page-trimmed request window union, packed
little-endian mask bits and fixed byte strides in borrowed integer workspace;
native `window_left=-1` leaves the complete predicate to CUSTOM. See
the ABI audit and provider list and
`tests/gpu/test_draft_custom_mask.py`. Implementation closure does not claim
GPU qualification: SM120/SM121 explicit selection, off-SM12x runtime execution,
and the default GPU suite runtime still need hardware validation.
