# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Seeded ordinary sampling with borrowed probability and CDF scratch."""

import torch

from .op import MAX_VOCAB, RequestLocalChainSampling
from .reference import ORDINARY_NAMESPACE

ORDINARY = ORDINARY_NAMESPACE


class RequestLocalSampling:
    """One request-local draw per row; construct and warm outside CUDA capture.

    Admission supplies finite T>=0, canonical k in [1,V], p in (0,1], optional
    min_p in [0,1], logits with finite support, valid seed slots and nonnegative
    int64 positions.
    Omitted filters must be identity on active T>0 rows. Padding slots (-1)
    read no row inputs and write -1. Only T=0 bypasses RNG.
    """

    MAX_VOCAB = MAX_VOCAB
    # "none" draws straight from the logits (no FP32 probabilities); filtered hints
    # leave top-p renormalization (P9) to the draw.
    launch_counts = {
        hint: count + (1 if hint == "none" else 2) - (hint in ("top_p", "both", "auto"))
        for hint, count in RequestLocalChainSampling.probability_launch_counts.items()
    }
    # sample(..., all_greedy=True): P1 chunk argmax records, then the row finalize.
    all_greedy_launch_count = 2

    def __init__(self, device, vocab_size: int):
        if isinstance(vocab_size, bool):
            raise ValueError(f"vocab_size must be in [1, {MAX_VOCAB}]")
        self._chain = RequestLocalChainSampling(device, 2, vocab_size)
        self.device, self.vocab_size = self._chain.device, vocab_size
        self._provider = self._chain._provider

    @property
    def provider(self):
        return self._chain.provider

    def workspace_size(self, batch_capacity):
        return self.workspace_layout(batch_capacity)["total_bytes"]

    def workspace_layout(self, batch_capacity):
        """Chain scratch (row states, min-p maxima/partials), then probabilities/CDF.

        256-byte region extents; 8-byte base alignment; no caller initialization.
        """
        if isinstance(batch_capacity, bool):
            raise ValueError("invalid batch capacity")
        layout = self._chain.workspace_layout(batch_capacity, "probabilities", length=1)
        offset, regions = layout["total_bytes"], list(layout["regions"])
        for name, payload in (
            ("probabilities", batch_capacity * self.vocab_size * 4),
            ("chunk_partials", batch_capacity * ((self.vocab_size + 4095) // 4096) * 8),
        ):
            extent = (payload + 255) // 256 * 256
            if payload:
                regions.append(
                    {
                        "name": name,
                        "payload_bytes": payload,
                        "alignment": 4,
                        "offset": offset,
                        "aligned_extent": extent,
                    }
                )
            offset += extent
        return {"base_alignment": 8, "total_bytes": offset, "regions": tuple(regions)}

    def sample(
        self,
        logits,
        temperatures,
        top_k,
        top_p,
        *,
        min_p=None,
        request_indices,
        seeds,
        positions,
        workspace,
        out_tokens,
        filters="auto",
        all_greedy=False,
    ):
        """Write int32[B] tokens using K1.10 ordinary RNG and CDF ABI r2.1.

        logits is bf16/fp16/fp32[B,V] with strides (s0,1), s0>=V;
        all other inputs and outputs are contiguous
        tensors on the operation device. CUDA performs no device allocation or
        readback. Workspace and output must not overlap any other argument.
        Optional FP32[B] min_p masks the K1 probabilities after top-p; equality
        at FP32(min_p * max(p)) is retained. None/zero is identity; T=0 ignores it.

        all_greedy=True is the caller's promise that every active row has T=0: each
        active row then takes the T=0 token (first argmax over the row, as the
        default path) without reading temperatures, top_k, top_p, min_p, seeds or
        positions, in one logits pass plus a record finalize (no sampling kernels).
        """
        if not isinstance(filters, str) or filters not in self.launch_counts:
            raise ValueError("filters must be none, top_k, top_p, both or auto")
        if not isinstance(all_greedy, bool):
            raise TypeError("all_greedy must be a bool")
        for name, value in (
            ("logits", logits),
            ("request_indices", request_indices),
            ("seeds", seeds),
            ("workspace", workspace),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
        if logits.ndim != 2:
            raise ValueError("logits must have shape [B,V]")
        if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("logits must have dtype bf16, fp16 or fp32")
        if request_indices.ndim != 1 or seeds.ndim != 2 or seeds.shape[1] != 2:
            raise ValueError("request_indices must be a vector and seeds must have shape [S,2]")
        b = request_indices.numel()
        specs = [
            ("logits", logits, logits.dtype, (b, self.vocab_size)),
            ("temperatures", temperatures, torch.float32, (b,)),
            ("top_k", top_k, torch.int32, (b,)),
            ("top_p", top_p, torch.float32, (b,)),
            ("request_indices", request_indices, torch.int32, (b,)),
            ("seeds", seeds, torch.uint32, (seeds.shape[0], 2)),
            ("positions", positions, torch.int64, (b,)),
            ("out_tokens", out_tokens, torch.int32, (b,)),
        ]
        if min_p is not None:
            specs.append(("min_p", min_p, torch.float32, (b,)))
        layout = self.workspace_layout(b)
        self._chain._check(specs, [out_tokens], workspace, b, "probabilities", layout=layout)
        if not b:
            return
        regions = {region["name"]: region for region in layout["regions"]}
        p_region = regions["probabilities"]
        start = p_region["offset"]
        partials = regions["chunk_partials"]
        work = workspace[partials["offset"] : partials["offset"] + partials["aligned_extent"]]
        fused = getattr(self._provider, "fused_ordinary", False)
        if all_greedy:
            if fused:
                self._provider.run(
                    "ordinary_identity",
                    (
                        logits,
                        temperatures,
                        None,
                        request_indices,
                        seeds,
                        positions,
                        workspace[:start],
                        work,
                        out_tokens,
                    ),
                    "greedy",
                )
                return
            # Reference providers: the same T=0 result from identity row inputs.
            temperatures = torch.zeros_like(temperatures)
            top_k = torch.full_like(top_k, self.vocab_size)
            top_p, min_p, filters = torch.ones_like(top_p), None, "none"
        if fused and filters == "none":
            self._provider.run(
                "ordinary_identity",
                (
                    logits,
                    temperatures,
                    min_p,
                    request_indices,
                    seeds,
                    positions,
                    workspace[:start],
                    work,
                    out_tokens,
                ),
            )
            return
        probs = workspace[start : start + p_region["payload_bytes"]].view(torch.float32)
        if fused:
            self._provider.run(
                "ordinary_filtered",
                (
                    logits,
                    temperatures,
                    top_k,
                    top_p,
                    min_p,
                    request_indices,
                    seeds,
                    positions,
                    probs.view(b, self.vocab_size),
                    workspace[:start],
                    work,
                    out_tokens,
                ),
                filters,
            )
            return
        probs = probs.view(b, 1, self.vocab_size)
        self._provider.run(
            "probabilities" if min_p is None else "probabilities_min_p",
            (
                logits.view(b, 1, self.vocab_size),
                temperatures,
                top_k,
                top_p,
                request_indices,
                probs,
                workspace[:start],
            ),
            filters,
        )
        tensors = (
            probs,
            temperatures,
            request_indices,
            seeds,
            positions,
            work,
            out_tokens.view(b, 1),
        )
        if min_p is not None:
            region = regions["min_p_chunk_maxima"]
            maxima = workspace[region["offset"] : region["offset"] + region["payload_bytes"]]
            self._provider.run("ordinary_min_p", (*tensors, min_p, maxima.view(torch.float32)))
        else:
            self._provider.run("ordinary", tensors)

    def warm_up(
        self,
        batch_capacity,
        dtypes=(torch.bfloat16, torch.float16, torch.float32),
        filters=("none", "top_k", "top_p", "both", "auto"),
        all_greedy=True,
        strided_logits=True,
    ):
        """Compile once per dtype/hint/min-p/group, covering B<=capacity and any S/stride.

        Dense rows (stride V) and row-strided logits are separate variants;
        strided_logits=False warms only the dense one.

        An iterable of batch sizes is also accepted; its maximum is the capacity.
        Both min-p presences and, if requested, the all-greedy path are warmed.
        Returns covered cache keys; len(self._provider.compiled) counts compiles.
        CPU and capacity zero return (). No capacity-sized buffers are allocated.
        """
        from . import _warmup as warm

        capacity = warm.capacity(batch_capacity)
        self.workspace_size(capacity)
        dtypes = warm.choices(dtypes, warm.DTYPES, "dtypes")
        filters = warm.choices(filters, warm.FILTERS, "filters")
        if not isinstance(all_greedy, bool):
            raise TypeError("all_greedy must be a bool")
        if not isinstance(strided_logits, bool):
            raise TypeError("strided_logits must be a bool")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm this DFlash operation variant before capture")
        if not capacity or not hasattr(self._provider, "warm_variants"):
            return ()
        return self._provider.warm_variants(
            lambda sms: warm.ordinary_variants(
                self, capacity, dtypes, filters, all_greedy, sms, strided_logits
            ),
            self.device,
        )
