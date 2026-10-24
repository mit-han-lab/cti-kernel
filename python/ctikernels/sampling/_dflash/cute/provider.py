# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Lazy, pinned CuTe provider. Compile/warm outside capture, then borrow all storage."""

import importlib.metadata
import threading

import torch

from ..reference.rng import ORDINARY_NAMESPACE

_COMPILING = threading.Lock()
_COMPILED = {}


def _nonempty_pool(pool, backing):
    # S=0 implies every row is padding. Kernels branch on slot<0 before
    # dereferencing this argument, so borrow one int64 row's bytes without
    # reading/copying it or allocating storage. CuTe requires positive extents.
    if pool.shape[0] == 0:
        return backing[:1].view(pool.dtype).reshape(1, *pool.shape[1:])
    return pool


class Provider:
    # RequestLocalSampling draws through ordinary_identity / ordinary_filtered.
    fused_ordinary = True

    def __init__(self):
        try:
            version = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("DFlash sampling requires nvidia-cutlass-dsl==4.8.0") from error
        if version != "4.8.0":
            raise RuntimeError("DFlash sampling requires nvidia-cutlass-dsl==4.8.0")
        from . import kernels

        self.kernels = kernels
        self.compiled = _COMPILED

    def run(self, mode, tensors, position=0):
        if tensors[0].shape[0] == 0:
            return
        # Compile for and launch on the tensor device, even if another device
        # is current in the caller. The engine's stream on this device is used.
        with torch.cuda.device(tensors[0].device):
            self._run(mode, tensors, position)

    def _prepare(self, mode, tensors, position, sms, smem_per_sm=None):
        """Normalize public arguments without allocating device storage."""
        reserve = compact_reserve(smem_per_sm)
        scalars = ()
        if mode in ("dense", "sparse", "sample", "point", "ordinary", "ordinary_min_p"):
            if mode in ("sample", "ordinary", "ordinary_min_p"):
                q, temperatures, slots, seeds, cursors, work, out = tensors[:7]
                # Borrow unused formals as aliases; min-p uses x/ids for extra inputs.
                x, ids = out, out
                if mode == "ordinary_min_p":
                    x, ids = tensors[7:]  # Unused sample formals carry min_p and chunk maxima.
                tensors = (q, x, q, ids, temperatures, slots, seeds, cursors, work, out, out)
                gamma, vocab = q.shape[1:]
                if gamma == 0:
                    return
            else:
                if mode == "point":
                    p, x, *tail = tensors
                    # q/ids are unused aliases: the point specialization never reads q.
                    tensors = (p, x, p, x, *tail)
                if mode == "dense":
                    p, x, q, *tail = tensors
                    tensors = (p, x, q, x, *tail)
                gamma, vocab = tensors[0].shape[1] - 1, tensors[0].shape[2]
            # Zero drafts have no readable proposal storage; CuTe needs positive extents.
            if gamma == 0:
                p, out = tensors[0], tensors[-2]
                tensors = (p, out, p, out, *tensors[4:])
            seeds = _nonempty_pool(tensors[6], tensors[7])
            tensors = (*tensors[:6], seeds, *tensors[7:])
            if mode in ("ordinary", "ordinary_min_p"):
                kernel = self.kernels.Chain(
                    gamma,
                    vocab,
                    "sample",
                    domain=ORDINARY_NAMESPACE,
                    scratch_positions=gamma,
                    **({"min_p": True} if mode == "ordinary_min_p" else {}),
                )
            else:
                kernel = self.kernels.Chain(gamma, vocab, mode)
        elif mode == "candidates":
            scores, ids, temperatures, slots, seeds, cursors, work, out, oq, pos = tensors
            # Candidates never use scratch. Do not pass the valid zero-byte
            # public workspace through CuTe's tensor type construction.
            tensors = (
                scores,
                ids,
                temperatures,
                slots,
                _nonempty_pool(seeds, cursors),
                cursors,
                out,
                oq,
                pos,
            )
            kernel = self.kernels.Candidates(position)
        elif mode in ("probabilities", "probabilities_min_p"):
            b, length, vocab = tensors[0].shape
            group = self.kernels.identity_group(b * length, vocab, sms) if position == "none" else 1
            kernel = self.kernels.Probabilities(
                length,
                vocab,
                position,
                b,
                group,
                # grid.y holds L up to 65535; longer L flattens rows onto grid.x.
                flat=length > 65535,
                # One compact CTA per SM only while the rows fit one wave; only
                # top-k hints launch the compact kernel.
                compact_reserve=reserve if b * length <= sms and position in TOP_K else 0,
                strided=strided_rows(tensors[0]),
                **({"output_maxima": True} if mode == "probabilities_min_p" else {}),
            )
        elif mode == "min_p_renormalize":
            b, length, vocab = tensors[0].shape
            kernel = self.kernels.MinPRenormalize(length, vocab, b, flat=length > 65535)
        elif mode == "ordinary_identity":
            logits, temperatures, min_p, slots, seeds, cursors, scratch, work, out = tensors
            b, vocab = logits.shape
            greedy_only = position == "greedy"
            use_min_p = min_p is not None and not greedy_only
            tensors = (
                logits,
                temperatures,
                min_p if use_min_p else temperatures,  # Unused alias without min_p.
                slots,
                _nonempty_pool(seeds, cursors),
                cursors,
                scratch,
                work,
                out,
            )
            group = self.kernels.identity_group(b, vocab, sms)
            groups = ((vocab + 4095) // 4096 + group - 1) // group
            # About eight CTAs per SM for P4/D1; more share one row prologue. A
            # count of one compiles the loop away (every B=1-4 row).
            loop = max(1, min(b * groups // (8 * sms), groups))
            kernel = self.kernels.OrdinaryIdentity(
                vocab,
                b,
                group,
                ORDINARY_NAMESPACE,
                min_p=use_min_p,
                greedy_only=greedy_only and self.kernels.greedy_group(b, vocab, sms),
                looped=loop > 1 and not greedy_only,
                strided=strided_rows(logits),
            )
            scalars = (loop,)
            position = "greedy" if greedy_only else use_min_p
        elif mode == "ordinary_filtered":
            logits, temperatures, top_k, top_p, min_p, slots, seeds, cursors = tensors[:8]
            probs, scratch, work, out = tensors[8:]
            b, vocab = logits.shape
            use_min_p = min_p is not None
            tensors = (
                logits,
                temperatures,
                top_k,
                top_p,
                min_p if use_min_p else temperatures,  # Unused alias without min_p.
                slots,
                _nonempty_pool(seeds, cursors),
                cursors,
                probs,
                scratch,
                work,
                out,
            )
            kernel = self.kernels.OrdinaryFiltered(
                vocab,
                b,
                position,
                ORDINARY_NAMESPACE,
                min_p=use_min_p,
                compact_reserve=reserve if b <= sms and position in TOP_K else 0,
                strided=strided_rows(logits),
            )
            position = (position, use_min_p)
        elif mode == "rng":
            kernel = self.kernels.Rng()
        elif mode == "cursors":
            anchors, prompts, slots, out = tensors
            tensors = (anchors, _nonempty_pool(prompts, anchors), slots, out)
            kernel = self.kernels.Cursors()
        else:
            raise ValueError("unknown CuTe operation")
        return kernel, tensors, position, scalars

    def _run(self, mode, tensors, position):
        properties = torch.cuda.get_device_properties(tensors[0].device)
        prepared = self._prepare(
            mode,
            tensors,
            position,
            properties.multi_processor_count,
            getattr(properties, "shared_memory_per_multiprocessor", None),
        )
        if prepared is None:
            return
        kernel, tensors, position, scalars = prepared
        key = compile_key(mode, position, kernel, tensors, tensors[0].device)
        compiled = self._get(key, kernel, tensors)
        compiled(*tensors, *scalars)

    def _get(self, key, kernel, tensors):
        compiled = self.compiled.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm this DFlash operation variant before capture")
            with _COMPILING:
                compiled = self.compiled.get(key)
                if compiled is None:
                    compiled = self._compile(kernel, tensors)
                    self.compiled[key] = compiled
        return compiled

    def warm_variants(self, variants, device):
        """Compile metadata-only variants; return unique covered cache keys.

        len(provider.compiled) before/after measures actual new compilations.
        No kernel launches or capacity-sized device allocations are needed.
        """
        keys = []
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm this DFlash operation variant before capture")
            properties = torch.cuda.get_device_properties(device)
            sms = properties.multi_processor_count
            smem = getattr(properties, "shared_memory_per_multiprocessor", None)
            for mode, inputs, hint in variants(sms):
                prepared = self._prepare(mode, inputs, hint, sms, smem)
                if prepared is None:
                    continue
                kernel, tensors, position, _ = prepared
                key = compile_key(mode, position, kernel, tensors, device)
                if key not in keys:
                    self._get(key, kernel, tensors)
                    keys.append(key)
        return tuple(keys)

    @staticmethod
    def _compile(kernel, tensors):
        import cutlass.cute as cute
        from cutlass.runtime import make_fake_stream

        return cute.compile(
            kernel,
            *build_fake_args(kernel, tensors),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


TOP_K = ("top_k", "both", "auto")  # filter hints whose kernels launch the compact pass


def strided_rows(logits):
    """Whether logits rows are not dense (row stride V); size-1 dims never count.

    Dense rows keep the constant-V row addressing; strided rows read the runtime
    strides (a separate variant, warmed by default).
    """
    dense = 1
    shape, strides = logits.shape[:-1], logits.stride()[:-1]
    for size, stride in zip(reversed(shape), reversed(strides), strict=True):
        if size > 1 and stride != dense * logits.shape[-1]:
            return True
        dense *= size
    return False


def compact_reserve(smem_per_sm):
    """Compact-CTA dynamic shared memory: just over half an SM (one CTA per SM).

    Used while the compact rows fit one wave (rows <= SMs); with more rows the
    CTAs must share SMs anyway, and a reservation would add waves.

    A device property, not a device table; without it, assume 100 KiB per SM.
    """
    return (smem_per_sm or 100 * 1024) // 2 + 16


def compile_key(mode, position, kernel, tensors, device):
    """Only semantic/CTA configuration and tensor types specialize compilation."""
    return (
        mode,
        position,
        kernel.config_key(),
        tuple((t.dtype, t.ndim) for t in tensors),
        torch.device(device),
    )


def build_fake_args(kernel, tensors):
    """Shared production/offline symbolic ABI; all dimensions and outer strides vary.

    Only the innermost stride is fixed at one. Rank, dtype and minimum natural
    alignment are invariant, including for singleton/empty and misaligned rows.
    OrdinaryIdentity's final scalar is the runtime P4/D1 loop count.
    """
    import cutlass
    import cutlass.cute as cute

    from .kernels import OrdinaryIdentity

    types = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.uint32: cutlass.Uint32,
        torch.uint8: cutlass.Uint8,
    }
    args = [
        cute.runtime.make_fake_tensor(
            types[t.dtype],
            tuple(cute.sym_int64() for _ in range(t.ndim)),
            tuple(1 if i == t.ndim - 1 else cute.sym_int64() for i in range(t.ndim)),
            assumed_align=t.element_size(),
        )
        for t in tensors
    ]
    if isinstance(kernel, OrdinaryIdentity):
        args.append(cutlass.Int32(1))
    return args
