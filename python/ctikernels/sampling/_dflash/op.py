# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Request-local DFlash sampling over engine-assembled global distributions.

Construct outside capture; warm each variant once before graph capture.
CPU uses the reference provider; CUDA uses CuTe DSL 4.8.0. Every method writes
borrowed outputs and returns None. CUDA run does not allocate device storage or
synchronize. Engine owns policy, FP32 selector scoring, serial orchestration,
collectives, immutable admission seeds and lifecycle. Only T=0 is greedy.
Probability construction supports deterministic temperature, top-k, top-p and min-p
filtering, with or without seeded downstream sampling.

Seeds are a slot pool uint32[S,2]; cursors are int64[B]. The low/high 32-bit seed
words form the full 64-bit Philox key. Active slot IDs, IDs, finite temperatures >=0 and
nonnegative normalized probability rows are device admission preconditions.
Padding (-1) branches before any row inputs or live seed state are read.
No inputs may alias writable outputs/workspace. Logits may be row-strided;
all other tensors are contiguous.
"""

import math

import torch

from .reference import (
    ACCEPT,
    BONUS,
    CORRECTION,
    K1_RNG_ABI_VERSION,
    ORDINARY_NAMESPACE,
    ORDINARY_NAMESPACE_END,
    PROPOSAL,
)

__all__ = [
    "RequestLocalChainSampling",
    "MAX_VOCAB",
    "validate_filters",
    "K1_RNG_ABI_VERSION",
    "PROPOSAL",
    "ACCEPT",
    "CORRECTION",
    "BONUS",
    "ORDINARY_NAMESPACE",
    "ORDINARY_NAMESPACE_END",
]


def validate_filters(top_k: int | None, top_p: float | None, vocab: int):
    """Host admission policy, before greedy dispatch; return canonical (k, p).

    None/nonpositive k disables top-k; k>=V is identity. None/nonpositive p
    disables top-p. Positive p>1 and nonfinite p are invalid, never clamped.
    Normalize identity filters first, then T=0 is greedy and T>0 follows the
    random contract, seeded or not, truncated or not (including top-k=1).
    """
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise ValueError("vocab must be a positive integer")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool)):
        raise TypeError("top_k must be an integer or None")
    k = vocab if top_k is None or top_k <= 0 or top_k >= vocab else top_k
    if top_p is not None and (not math.isfinite(top_p) or top_p > 1):
        raise ValueError("top_p must be finite and <=1")
    p = 1.0 if top_p is None or top_p <= 0 else float(top_p)
    return k, p


# CUDA kernels stage per-4096-token-chunk records in shared memory (up to five
# 32-bit words per chunk); 2^21 keeps that at 10 KiB per CTA, far above real
# vocabularies. Both providers enforce it so the contract is device-independent.
MAX_VOCAB = 1 << 21


class RequestLocalChainSampling:
    """Borrowed buffers; probabilities uses up to eleven launches (+2 for min-p).

    ``workspace_size`` returns bytes (256-byte aligned, no allocator overhead).
    Supply contiguous uint8 storage of at least that size, aligned as reported
    by ``workspace_layout`` (4 bytes for sampling, 8 for probabilities).
    Chunk records occupy B*max(2,Q)*ceil(V/4096)*8 bytes, shared by mutually exclusive
    proposal, stochastic verify and greedy verify calls. Two words store sum/last
    or max/argmax; stochastic verify uses positions 0/1 for residual/p fallback.
    Q >= 1; Q and B*Q must be <= 2147483647 for int32 positions/counts
    and CUDA grid indices. Tensor and scratch offsets use int64.
    Q=1 emits only the bonus from target row 0, with accepted count zero.
    Scratch contents are private, not outputs.
    Probabilities uses per-position integer histograms and 4096-token chunk
    records, aligned to 8 bytes. Scratch may be uninitialized, including first
    use: a dedicated launch clears histograms on every filtered call.
    ``launch_counts`` describes API kernel launches, excluding setup compilation.
    """

    launch_counts = {
        "probabilities": 11,
        "prepare_cursors": 1,
        "sample_candidates": 1,
        "sample_dense_proposals": 2,
        "verify_dense": 2,
        "verify_point_mass": 2,
        "verify_sparse": 2,
    }

    MAX_VOCAB = MAX_VOCAB

    probability_launch_counts = {"none": 3, "top_k": 7, "top_p": 8, "both": 11, "auto": 11}
    probability_min_p_launches = 2

    def __init__(self, device, block_size: int, vocab_size: int):
        if type(block_size) is not int or not 1 <= block_size < 2**31:
            raise ValueError("block_size must be >= 1 and <= 2147483647 (int32 positions/counts)")
        if not isinstance(vocab_size, int) or not 1 <= vocab_size <= MAX_VOCAB:
            raise ValueError(f"vocab_size must be in [1, {MAX_VOCAB}]")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            from .cute import Provider
        elif self.device.type == "cpu":
            from .reference import Provider
        else:
            raise ValueError("DFlash sampling supports CPU and CUDA")
        self.block_size, self.vocab_size = block_size, vocab_size
        if block_size == 1:
            self.launch_counts = {
                **self.launch_counts,
                "sample_candidates": 0,
                "sample_dense_proposals": 0,
            }
        self._provider = Provider()

    @property
    def provider(self):
        return "cute" if self.device.type == "cuda" else "reference"

    def workspace_size(self, batch_capacity, representation="dense", *, length=None):
        return self.workspace_layout(batch_capacity, representation, length=length)["total_bytes"]

    def workspace_layout(self, batch_capacity, representation="dense", *, length=None):
        """Itemized borrowed scratch; length defaults to block_size for probabilities.

        Histogram regions each pack 11/11/10-bit levels. Probability chunk
        records hold max/argmax/Z_gt/top-k ties/top-p ties (five 32-bit words).
        Compact lists pack 1024 int32 IDs, 1024 FP32 values and an int32
        append counter per row; CAP=1024 is independent of the host hint.
        Row states hold six levels (three counts, three masses), each with
        six uint64 words: arrival counter, key, remaining, above, equal, full.
        Two FP32 [B*L,ceil(V/4096)] regions follow row states: min-p chunk
        maxima and partial sums, each rounded to 256 bytes, always reserved.
        P0 resets counters each filtered call; producers overwrite outputs.
        Allocator backing is accounted by the engine; no first-use zeroing.
        """
        if representation not in ("dense", "sparse", "candidates", "probabilities"):
            raise ValueError("unknown representation")
        if not isinstance(batch_capacity, int) or not 0 <= batch_capacity < 2**31:
            raise ValueError("invalid batch capacity")
        if batch_capacity * self.block_size >= 2**31:
            raise ValueError("B*Q must be <= 2147483647 (int32 CUDA grid indices)")
        if length is None:
            length = self.block_size
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or not 1 <= length <= self.block_size
        ):
            raise ValueError("length must be in [1, block_size]")
        nchunks = (self.vocab_size + 4095) // 4096
        if representation == "probabilities":
            rows = batch_capacity * length
            offset, regions = 0, []
            # Each histogram packs 11/11/10-bit levels (2048+2048+1024 bins).
            for name, payload, alignment in (
                ("top_k_histograms", rows * 5120 * 4, 4),
                ("top_p_histograms", rows * 5120 * 8, 8),
                ("probability_chunk_records", rows * nchunks * 20, 4),
                ("compact_lists", rows * (1024 * 4 + 1024 * 4 + 4), 4),
                ("row_states", rows * 6 * 6 * 8, 8),
                ("min_p_chunk_maxima", rows * nchunks * 4, 4),
                ("min_p_chunk_partials", rows * nchunks * 4, 4),
            ):
                extent = (payload + 255) // 256 * 256
                if payload:
                    regions.append(
                        {
                            "name": name,
                            "payload_bytes": payload,
                            "alignment": alignment,
                            "offset": offset,
                            "aligned_extent": extent,
                        }
                    )
                offset += extent
            return {"base_alignment": 8, "total_bytes": offset, "regions": tuple(regions)}
        payload = (
            batch_capacity * max(2, self.block_size) * nchunks * 8
            if representation in ("dense", "sparse")
            else 0
        )
        extent = (payload + 255) // 256 * 256
        return {
            "base_alignment": 4,
            "total_bytes": extent,
            "regions": (
                {
                    "name": "chunk_partials",
                    "payload_bytes": payload,
                    "alignment": 4,
                    "offset": 0,
                    "aligned_extent": extent,
                },
            )
            if payload
            else (),
        }

    def _check(self, specs, outputs, workspace, batch, representation, *, length=None, layout=None):
        for name, tensor, dtype, shape in specs:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.requires_grad:
                raise ValueError(f"{name} requires grad")
            if tensor.dtype != dtype:
                raise TypeError(f"{name} must have dtype {dtype}")
            layout_ok = tensor.is_contiguous()
            if name == "logits":
                layout_ok = (
                    tensor.stride(-1) == 1
                    and tensor.stride(-2) >= self.vocab_size
                    and (
                        tensor.ndim == 2
                        or tensor.shape[0] <= 1
                        or tensor.stride(0) >= tensor.shape[1] * tensor.stride(1)
                    )
                )
            if tensor.device != self.device or tuple(tensor.shape) != shape or not layout_ok:
                raise ValueError(f"{name} has wrong device, shape or layout")
        if (
            workspace.dtype != torch.uint8
            or workspace.device != self.device
            or workspace.ndim != 1
            or not workspace.is_contiguous()
        ):
            raise ValueError("workspace must be contiguous uint8 on the operation device")
        if layout is None:
            layout = self.workspace_layout(batch, representation, length=length)
        if (
            workspace.numel() < layout["total_bytes"]
            or workspace.data_ptr() % layout["base_alignment"]
        ):
            raise ValueError("workspace is too small or misaligned")
        writable = [*outputs, workspace]
        all_tensors = [s[1] for s in specs] + [workspace]
        for dst in writable:
            if not dst.numel():
                continue
            lo, hi = dst.data_ptr(), dst.data_ptr() + dst.numel() * dst.element_size()
            for src in all_tensors:
                if src is dst or not src.numel():
                    continue
                span = 1 + sum(
                    (size - 1) * stride
                    for size, stride in zip(src.shape, src.stride(), strict=True)
                )
                slo, shi = src.data_ptr(), src.data_ptr() + span * src.element_size()
                if lo < shi and slo < hi:
                    raise ValueError("outputs and workspace must not alias any other tensor")
        # Identity aliases must also be rejected (including identical output objects).
        for dst in outputs:
            if sum(t is dst for t in all_tensors) != 1:
                raise ValueError("output aliases another argument")

    def _common(self, temperatures, request_indices, seeds, cursors):
        if request_indices.ndim != 1 or seeds.ndim != 2 or seeds.shape[1] != 2:
            raise ValueError("request_indices must be a vector and seeds must have shape [S,2]")
        b = request_indices.numel()
        return b, [
            ("temperatures", temperatures, torch.float32, (b,)),
            ("request_indices", request_indices, torch.int32, (b,)),
            ("seeds", seeds, torch.uint32, (seeds.shape[0], 2)),
            ("cursors", cursors, torch.int64, (b,)),
        ]

    def probabilities(
        self,
        logits,
        temperatures,
        top_k,
        top_p,
        *,
        min_p=None,
        request_indices,
        workspace,
        out_probs,
        filters="auto",
    ):
        """Construct FP32 [B,L,V] probabilities, 1<=L<=block_size; no RNG.

        Logits may have strides (s0,s1,1), s1>=V and s0>=L*s1 for B>1.
        All other tensors, including verify_* probabilities, remain contiguous.
        Admission supplies canonical k in [1,V], p in (0,1], finite T>=0,
        and logits with finite support (finite values or -inf). Padding writes
        zeros without reading row inputs. T=0 ignores filters and picks the
        smallest argmax ID. T>0 uses FP32 scaling/softmax and exact 2^-62
        fixed-point top-p masses; identity top-p does not renormalize.
        ``filters`` is a host-static hint: none/top_k/top_p/both/auto. Omitted
        filters must be identity on active T>0 rows (admission precondition).
        Launch counts are 3/7/8/11/11 respectively, independent of device values.
        Optional contiguous FP32[B] min_p is finite and in [0,1] by admission.
        Engine order (10-17) is T -> top-k -> top-p -> min-p, as in S1 sample;
        vLLM V2 (0.31.0) instead applies min-p before top-k/top-p. For T>0 and
        min_p>0, keep p>=FP32(min_p*max(p)), then divide by probability_sum of
        kept p with FP32 div.rn. T=0, padding and min_p<=0 preserve output bits.
        Presence is host static: a tensor adds probability_min_p_launches (2)
        launches; None adds none. Verification consumes the resulting p as-is.
        Size scratch with length=L (default block_size). No first-use zeroing
        is required; each filtered call initializes its histograms on device.
        """
        if filters not in self.probability_launch_counts:
            raise ValueError("filters must be none, top_k, top_p, both or auto")
        if logits.ndim != 3 or not 1 <= logits.shape[1] <= self.block_size:
            raise ValueError("logits must have shape [B,L,V] with 1<=L<=block_size")
        if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("logits must have dtype bf16, fp16 or fp32")
        if request_indices.ndim != 1:
            raise ValueError("request_indices must be a vector")
        b, length = request_indices.numel(), logits.shape[1]
        specs = [
            ("logits", logits, logits.dtype, (b, length, self.vocab_size)),
            ("temperatures", temperatures, torch.float32, (b,)),
            ("top_k", top_k, torch.int32, (b,)),
            ("top_p", top_p, torch.float32, (b,)),
            ("request_indices", request_indices, torch.int32, (b,)),
            ("out_probs", out_probs, torch.float32, (b, length, self.vocab_size)),
        ]
        if min_p is not None:
            specs.append(("min_p", min_p, torch.float32, (b,)))
        self._check(specs, [out_probs], workspace, b, "probabilities", length=length)
        if b:
            self._provider.run(
                "probabilities" if min_p is None else "probabilities_min_p",
                (logits, temperatures, top_k, top_p, request_indices, out_probs, workspace),
                filters,
            )
            if min_p is not None:
                self._provider.run(
                    "min_p_renormalize",
                    (out_probs, temperatures, min_p, request_indices, workspace),
                )

    def warm_up(
        self,
        *,
        batch_capacity,
        dtypes=(torch.bfloat16, torch.float16, torch.float32),
        filters=("none", "top_k", "top_p", "both", "auto"),
        min_p=(False, True),
        modes=("point", "dense", "sparse", "sample", "candidates", "cursors"),
        strided_logits=True,
    ):
        """Compile variants once for every B<=capacity, L<=Q, S and logits stride.

        Dense logits rows (row stride V) and row-strided logits are separate
        variants; strided_logits=False warms only the dense one (calls with
        strided logits then need their own warm-up before capture).

        Returns the covered cache keys (empty for CPU or capacity zero).
        ``len(self._provider.compiled)`` deltas count new CuTe compilations.
        Fixed V/Q, dtype, hint, min-p presence, mode and CTA group specialize;
        tensor dimensions, strides and ordinary loop counts do not.
        """
        from . import _warmup as warm

        capacity = warm.capacity(batch_capacity)
        self.workspace_size(capacity)
        dtypes = warm.choices(dtypes, warm.DTYPES, "dtypes")
        filters = warm.choices(filters, warm.FILTERS, "filters")
        min_p = warm.choices(min_p, (False, True), "min_p")
        modes = warm.choices(modes, warm.MODES, "modes")
        if not isinstance(strided_logits, bool):
            raise TypeError("strided_logits must be a bool")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm this DFlash operation variant before capture")
        if not capacity or not hasattr(self._provider, "warm_variants"):
            return ()

        def variants(sms):
            yield from warm.probability_variants(
                self, capacity, dtypes, filters, min_p, strided_logits
            )
            yield from warm.chain_variants(self, modes)

        return self._provider.warm_variants(variants, self.device)

    def prepare_cursors(
        self, anchor_positions, prompt_lengths, *, request_indices, workspace, out_cursors
    ):
        """Device int64 anchor_position - prompt_length[slot] + 1; padding gets 0.

        Engine maintains nonnegative operands and a representable nonnegative
        difference. Recompute from committed positions each round; no RNG state
        is advanced by proposal or verification. No workspace payload is needed.
        """
        if request_indices.ndim != 1 or prompt_lengths.ndim != 1:
            raise ValueError("request_indices and prompt_lengths must be vectors")
        b = request_indices.numel()
        specs = [
            ("anchor_positions", anchor_positions, torch.int64, (b,)),
            ("prompt_lengths", prompt_lengths, torch.int64, (prompt_lengths.numel(),)),
            ("request_indices", request_indices, torch.int32, (b,)),
            ("out_cursors", out_cursors, torch.int64, (b,)),
        ]
        self._check(specs, [out_cursors], workspace, b, "candidates")
        if b:
            self._provider.run(
                "cursors", (anchor_positions, prompt_lengths, request_indices, out_cursors)
            )

    def sample_candidates(
        self,
        scores,
        ids,
        temperatures,
        position_i,
        *,
        request_indices,
        seeds,
        cursors,
        workspace,
        out_ids,
        out_q,
    ):
        """Sample 16 distinct ascending global IDs; out_q preserves input order.

        Greedy rows save zero q; padding writes out_ids=-1 and q=0.
        Candidate scores/temperature and softmax use FP32; ties choose smaller ID.
        Device position bounds are an admission precondition, without host readback.
        Q=1 is a no-op: position 0 is allowed and outputs are untouched.
        """
        if self.block_size > 1 and self.vocab_size < 16:
            raise ValueError("candidate sampling requires V>=16")
        b, specs = self._common(temperatures, request_indices, seeds, cursors)
        device_position = isinstance(position_i, torch.Tensor)
        if device_position:
            specs += [("position_i", position_i, torch.int32, ())]
        elif (
            not isinstance(position_i, int)
            or isinstance(position_i, bool)
            or not 0 <= position_i < max(1, self.block_size - 1)
        ):
            raise ValueError("position_i must be in [0, block_size-1), or 0 for Q=1")
        specs += [
            ("scores", scores, torch.float32, (b, 16)),
            ("ids", ids, torch.int32, (b, 16)),
            ("out_ids", out_ids, torch.int32, (b,)),
            ("out_q", out_q, torch.float32, (b, 16)),
        ]
        self._check(specs, [out_ids, out_q], workspace, b, "candidates")
        if b and self.block_size > 1:
            self._provider.run(
                "candidates",
                (
                    scores,
                    ids,
                    temperatures,
                    request_indices,
                    seeds,
                    cursors,
                    workspace,
                    out_ids,
                    out_q,
                    position_i if device_position else request_indices,
                ),
                None if device_position else position_i,
            )

    def sample_dense_proposals(
        self, q, temperatures, *, request_indices, seeds, cursors, workspace, out_ids
    ):
        b, specs = self._common(temperatures, request_indices, seeds, cursors)
        specs += [
            ("q", q, torch.float32, (b, self.block_size - 1, self.vocab_size)),
            ("out_ids", out_ids, torch.int32, (b, self.block_size - 1)),
        ]
        self._check(specs, [out_ids], workspace, b, "dense")
        if b and self.block_size > 1:
            self._provider.run(
                "sample", (q, temperatures, request_indices, seeds, cursors, workspace, out_ids)
            )

    def verify_dense(
        self,
        p,
        drafts,
        q,
        temperatures,
        *,
        request_indices,
        seeds,
        cursors,
        workspace,
        out_tokens,
        accepted_counts,
    ):
        self._verify(
            p,
            drafts,
            q,
            None,
            temperatures,
            request_indices,
            seeds,
            cursors,
            workspace,
            out_tokens,
            accepted_counts,
        )

    def verify_point_mass(
        self,
        p,
        drafts,
        temperatures,
        *,
        request_indices,
        seeds,
        cursors,
        workspace,
        out_tokens,
        accepted_counts,
    ):
        """Verify argmax drafts with implicit one-hot q; no q storage or reads.

        Bitwise identical to verify_dense with q[b,i,drafts[b,i]]=1 and
        all other q entries zero: FP32 U*1<p(d), residual max(p-onehot,0),
        zero-mass fallback to p, and the same CDF ABI r2.1 and RNG domains.
        T=0 uses target argmax with smaller-ID ties. Padding and borrowed
        dense workspace are unchanged; CUDA uses two launches.
        """
        self._verify(
            p,
            drafts,
            None,
            None,
            temperatures,
            request_indices,
            seeds,
            cursors,
            workspace,
            out_tokens,
            accepted_counts,
            point=True,
        )

    def verify_sparse(
        self,
        p,
        drafts,
        cand_ids,
        cand_q,
        temperatures,
        *,
        request_indices,
        seeds,
        cursors,
        workspace,
        out_tokens,
        accepted_counts,
    ):
        """Duplicate candidate IDs are aggregated in original column order."""
        self._verify(
            p,
            drafts,
            cand_q,
            cand_ids,
            temperatures,
            request_indices,
            seeds,
            cursors,
            workspace,
            out_tokens,
            accepted_counts,
        )

    def _verify(
        self,
        p,
        drafts,
        q,
        ids,
        temperatures,
        slots,
        seeds,
        cursors,
        workspace,
        out,
        counts,
        *,
        point=False,
    ):
        b, specs = self._common(temperatures, slots, seeds, cursors)
        g, v = self.block_size - 1, self.vocab_size
        specs += [
            ("p", p, torch.float32, (b, g + 1, v)),
            ("drafts", drafts, torch.int32, (b, g)),
            ("out_tokens", out, torch.int32, (b, g + 1)),
            ("accepted_counts", counts, torch.int32, (b,)),
        ]
        if not point:
            specs += [("q", q, torch.float32, (b, g, v if ids is None else 16))]
        if ids is not None:
            specs += [("cand_ids", ids, torch.int32, (b, g, 16))]
        self._check(specs, [out, counts], workspace, b, "dense")
        if b:
            inputs = (p, drafts) if point else (p, drafts, q) + (() if ids is None else (ids,))
            self._provider.run(
                "point" if point else "dense" if ids is None else "sparse",
                (*inputs, temperatures, slots, seeds, cursors, workspace, out, counts),
            )
