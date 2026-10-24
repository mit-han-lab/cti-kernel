# Linear numerics: CuTe vs cuBLAS against FP64

Question from the engine (2026-10-15): with CuTe dispatch, the decode graph-vs-eager max log-prob difference is 0.108, against 0.051 with forced cuBLAS, and the mean difference doubles. Does CuTe round with a systematic bias?

Answer: no. The study found no bias, no rounding shortcut and no extra amplification of perturbations in the CuTe GEMMs. Evidence:

1. **Rounding mode.** Every CuTe variant converts FP32 to BF16 with round-to-nearest-even. The PTX has `cvt.rn.bf16.f32` / `cvt.rn.bf16x2.f32` and the SASS has `F2FP.BF16.F32.PACK_AB` with no `.RZ`. There are no `.rz`, `.ftz` or `.approx` instructions. This was checked for the smem, reg and split-K reduction kernels, for sm_120a and sm_121a.
2. **Split-K and loads.** Partials are FP32 and summed in a fixed order with plain FADD (round-to-nearest). Every product goes through `mma.sync ... f32.bf16.bf16.f32` (HMMA.16816.F32.BF16), which is the same tensor-core accumulation that cuBLAS uses for M ≥ 2. Nothing is stored as a BF16 partial.
3. **Signed error vs FP64** (table below). Inputs are realistic decode activations: RMSNorm-like rows with 1% outlier channels, shifted rows, and SwiGLU products for down. There are 4 batches per M, and the same 8 rows are reused for every M. The column "mean err along sign(ref)" averages (y - ref)/ulp(ref) · sign(ref), where a negative value means shrinkage toward zero.
   - CuTe's mean is within ±0.04 ulp of zero, inside 3 standard errors (≤ 0.043 ulp), and matches cuBLAS's on every row.
   - CuTe's RMS error is ≤ cuBLAS's on most rows. There are two exceptions. The first is the RTX 5090 4096² rows with plain RMSNorm inputs (1.83 vs 1.55 ulp). The second is lm_head at M=1, where cuBLAS uses a GEMV with FFMA accumulation. In both, the RMS is dominated by outputs near zero, and the mean is still unbiased.
   - Both tensor-core paths move 60-66% of their misrounded outputs toward zero (the HMMA alignment truncation). That share is the same for CuTe and cuBLAS, and the misrounded fraction is ≤ 0.6%.
4. **M invariance.** CuTe rows at M=1..7 are bitwise equal to the same rows at M=8 under one bucket configuration. cuBLAS rows equal their M=8 rows for M=2..7 on every shape except down at M=2 on the RTX 5090. Most shapes differ at M=1 (GEMV), so a padded cuBLAS batch is not bitwise either.
5. **Propagation.** The same input perturbation was pushed through an 8-layer residual stack (norm, o, gate_up, SiLU·mul, down; M=8; 16 seeds) and then a 32k-vocab head. The perturbation was a 1-ulp nudge on 2% of the inputs, standing in for a padded-batch difference in a non-GEMM kernel. Only the GEMM backend changes between columns. The CuTe log-prob differences match cuBLAS and an exactly rounded FP32-accumulate GEMM:

| device | CuTe mean / max | cuBLAS mean / max | exact-rounding GEMM mean / max |
|---|---|---|---|
| RTX 5090 | 0.00747 / 0.0511 | 0.00749 / 0.0525 | 0.00749 / 0.0624 |
| RTX PRO 6000 | 0.00751 / 0.0505 | 0.00751 / 0.0548 | 0.00754 / 0.0473 |
| GB10 | 0.00752 / 0.0508 | 0.00751 / 0.0507 | 0.00751 / 0.0625 |

The engine's doubling is therefore not caused by the CuTe GEMMs' rounding. A single max at a margin-0.125 token is one sample. Once a token flips, everything after it diverges, and the per-position mean doubles with it. The kernel side suggests comparing the two configurations over more prompts, or comparing per-position differences only up to the first divergent token.

Reproduce: `python benchmark/kernel_tuning/linear_smallm_numerics.py` on each GPU.

## Signed error per shape (M=1-8, lm_head M=1; ulp = BF16 ulp of the FP64 reference)

| device | shape (N, K) | activations | M | CuTe mean err along sign(ref) (ulp) | cuBLAS | CuTe RMS (ulp) | cuBLAS | CuTe correctly rounded | cuBLAS |
|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | gdn in_proj (12352, 4096) | rmsnorm | 1-8 | -0.0010 | -0.0015 | 0.339 | 0.394 | 0.9986 | 0.9978 |
| RTX 5090 | gdn in_proj (12352, 4096) | rmsnorm_shifted | 1-8 | +0.0005 | -0.0035 | 0.385 | 1.339 | 0.9987 | 0.9979 |
| RTX 5090 | out_proj/o (4096, 4096) | rmsnorm | 1-8 | +0.0009 | +0.0034 | 1.830 | 1.550 | 0.9979 | 0.9979 |
| RTX 5090 | out_proj/o (4096, 4096) | rmsnorm_shifted | 1-8 | -0.0014 | -0.0014 | 0.308 | 0.309 | 0.9975 | 0.9975 |
| RTX 5090 | qkv+gate (10240, 4096) | rmsnorm | 1-8 | +0.0001 | +0.0002 | 0.290 | 0.317 | 0.9994 | 0.9979 |
| RTX 5090 | qkv+gate (10240, 4096) | rmsnorm_shifted | 1-8 | +0.0004 | +0.0001 | 0.291 | 0.344 | 0.9994 | 0.9977 |
| RTX 5090 | gate_up (24576, 4096) | rmsnorm | 1-8 | -0.0003 | -0.0005 | 0.293 | 0.307 | 0.9988 | 0.9980 |
| RTX 5090 | gate_up (24576, 4096) | rmsnorm_shifted | 1-8 | -0.0007 | -0.0007 | 0.383 | 0.461 | 0.9987 | 0.9978 |
| RTX 5090 | down (4096, 12288) | swiglu | 1-8 | +0.0005 | -0.0006 | 0.328 | 0.563 | 0.9976 | 0.9944 |
| RTX 5090 | in_proj_ba (64, 4096) | rmsnorm | 1-8 | +0.0003 | -0.0097 | 0.285 | 0.584 | 0.9994 | 0.9988 |
| RTX 5090 | in_proj_ba (64, 4096) | rmsnorm_shifted | 1-8 | -0.0026 | -0.0046 | 0.285 | 0.290 | 1.0000 | 0.9987 |
| RTX 5090 | lm_head (248320, 4096) | rmsnorm | 1 | -0.0003 | +0.0001 | 1.840 | 0.294 | 0.9976 | 0.9998 |
| RTX 5090 | lm_head (248320, 4096) | rmsnorm_shifted | 1 | -0.0009 | -0.0001 | 0.499 | 0.291 | 0.9975 | 0.9998 |
| RTX PRO 6000 | gdn in_proj (12352, 4096) | rmsnorm | 1-8 | -0.0008 | -0.0009 | 0.294 | 0.341 | 0.9986 | 0.9974 |
| RTX PRO 6000 | gdn in_proj (12352, 4096) | rmsnorm_shifted | 1-8 | -0.0001 | -0.0002 | 0.306 | 0.348 | 0.9988 | 0.9978 |
| RTX PRO 6000 | out_proj/o (4096, 4096) | rmsnorm | 1-8 | -0.0004 | -0.0005 | 0.292 | 0.296 | 0.9995 | 0.9994 |
| RTX PRO 6000 | out_proj/o (4096, 4096) | rmsnorm_shifted | 1-8 | -0.0024 | -0.0025 | 0.289 | 0.289 | 0.9995 | 0.9994 |
| RTX PRO 6000 | qkv+gate (10240, 4096) | rmsnorm | 1-8 | -0.0000 | -0.0003 | 0.293 | 0.335 | 0.9991 | 0.9977 |
| RTX PRO 6000 | qkv+gate (10240, 4096) | rmsnorm_shifted | 1-8 | +0.0005 | -0.0000 | 0.377 | 0.469 | 0.9991 | 0.9978 |
| RTX PRO 6000 | gate_up (24576, 4096) | rmsnorm | 1-8 | -0.0006 | -0.0012 | 0.298 | 0.331 | 0.9987 | 0.9978 |
| RTX PRO 6000 | gate_up (24576, 4096) | rmsnorm_shifted | 1-8 | +0.0009 | +0.0009 | 0.561 | 0.699 | 0.9986 | 0.9978 |
| RTX PRO 6000 | down (4096, 12288) | swiglu | 1-8 | +0.0008 | +0.0003 | 0.642 | 0.635 | 0.9989 | 0.9984 |
| RTX PRO 6000 | in_proj_ba (64, 4096) | rmsnorm | 1-8 | +0.0009 | +0.0009 | 0.291 | 0.291 | 1.0000 | 0.9990 |
| RTX PRO 6000 | in_proj_ba (64, 4096) | rmsnorm_shifted | 1-8 | -0.0095 | -0.0086 | 0.290 | 0.296 | 0.9999 | 0.9984 |
| RTX PRO 6000 | lm_head (248320, 4096) | rmsnorm | 1 | -0.0013 | -0.0005 | 0.324 | 0.289 | 0.9977 | 0.9998 |
| RTX PRO 6000 | lm_head (248320, 4096) | rmsnorm_shifted | 1 | -0.0024 | -0.0002 | 1.129 | 0.303 | 0.9976 | 0.9998 |
| GB10 | gdn in_proj (12352, 4096) | rmsnorm | 1-8 | -0.0018 | -0.0018 | 0.438 | 0.436 | 0.9975 | 0.9978 |
| GB10 | gdn in_proj (12352, 4096) | rmsnorm_shifted | 1-8 | +0.0029 | +0.0030 | 1.433 | 1.432 | 0.9974 | 0.9978 |
| GB10 | out_proj/o (4096, 4096) | rmsnorm | 1-8 | -0.0009 | -0.0011 | 0.296 | 0.295 | 0.9988 | 0.9977 |
| GB10 | out_proj/o (4096, 4096) | rmsnorm_shifted | 1-8 | -0.0011 | -0.0015 | 0.303 | 0.346 | 0.9987 | 0.9977 |
| GB10 | qkv+gate (10240, 4096) | rmsnorm | 1-8 | +0.0005 | +0.0005 | 0.326 | 0.326 | 0.9976 | 0.9977 |
| GB10 | qkv+gate (10240, 4096) | rmsnorm_shifted | 1-8 | -0.0162 | -0.0156 | 6.014 | 5.916 | 0.9976 | 0.9977 |
| GB10 | gate_up (24576, 4096) | rmsnorm | 1-8 | -0.0011 | -0.0010 | 0.582 | 0.582 | 0.9976 | 0.9979 |
| GB10 | gate_up (24576, 4096) | rmsnorm_shifted | 1-8 | +0.0133 | +0.0082 | 7.457 | 5.789 | 0.9976 | 0.9978 |
| GB10 | down (4096, 12288) | swiglu | 1-8 | -0.0022 | -0.0032 | 0.341 | 0.482 | 0.9962 | 0.9939 |
| GB10 | in_proj_ba (64, 4096) | rmsnorm | 1-8 | +0.0127 | +0.0109 | 0.292 | 0.294 | 1.0000 | 0.9977 |
| GB10 | in_proj_ba (64, 4096) | rmsnorm_shifted | 1-8 | -0.0052 | -0.0063 | 0.284 | 0.286 | 0.9996 | 0.9967 |
| GB10 | lm_head (248320, 4096) | rmsnorm | 1 | -0.0010 | +0.0000 | 0.478 | 0.291 | 0.9976 | 0.9998 |
| GB10 | lm_head (248320, 4096) | rmsnorm_shifted | 1 | -0.0001 | +0.0005 | 0.434 | 0.289 | 0.9975 | 0.9998 |
