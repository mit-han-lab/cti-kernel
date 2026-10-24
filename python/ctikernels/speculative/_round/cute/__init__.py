# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Cached dynamic-layout TVM-FFI launchers; no per-call DLPack conversion."""

from importlib.metadata import version
from threading import RLock

import torch

from ..op import BOOL, C_NAMES, I64, P_NAMES, R_NAMES, TABLES

# fixed_query_addresses is the Triton kernel (../triton).
NAMES = {"prepare": P_NAMES, "round": R_NAMES, "commit": C_NAMES}
KINDS = {"prepare_block": "prepare", "prepare_round": "round", "commit_block": "commit"}
PUBLIC = {kind: name for name, kind in KINDS.items()}

# B<=SMALL_BATCH takes each variant's unrolled kernel, larger B its any-B kernel.
SMALL_BATCH = 4
_COMPILED = {}
_LOCK = RLock()


def compile_count():
    return len(_COMPILED)


def compile_variant(kind, q, flags, arch):
    """Offline-capable compilation, with no real tensors or CUDA allocations.

    Returns the (B<=4, any-B) kernel pair of one variant; the host picks by B.
    """
    return tuple(_compile(kind, q, flags, arch, generic) for generic in (False, True))


def _compile(kind, q, flags, arch, generic):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from .kernels import Commit, Prepare

    names = NAMES[kind]
    args = []
    for name in names:
        dtype = cutlass.Int64 if name in I64 else cutlass.Boolean if name in BOOL else cutlass.Int32
        shape = (cute.sym_int64(), cute.sym_int64()) if name in TABLES else (cute.sym_int64(),)
        strides = (cute.sym_int64(), 1) if name in TABLES else (1,)
        args.append(
            cute.runtime.make_fake_tensor(
                dtype, shape, strides, assumed_align=1 if name in BOOL else 8 if name in I64 else 4
            )
        )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    if kind != "commit":
        part = {"prepare": "full", "round": "round"}[kind]
        kernel = Prepare(q, *flags, part=part, generic=generic)
        scalars = (
            cutlass.Int64(1),
            cutlass.Int64(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            cutlass.Int32(1),
        )
    else:
        kernel = Commit(q, *flags, generic=generic)
        scalars = (cutlass.Int64(1), cutlass.Int32(1))
    return cute.compile(
        kernel,
        tuple(args),
        *scalars,
        stream,
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch} --opt-level=3",
    )


class Provider:
    def __init__(self, device):
        if version("nvidia-cutlass-dsl") != "4.8.0":
            raise RuntimeError("DFlash round bookkeeping requires nvidia-cutlass-dsl==4.8.0")
        self.device = device
        self.arch = "".join(map(str, torch.cuda.get_device_capability(device)))
        # P's fixed grid: a host static of the device, never of F, M or B. Two
        # CTAs per SM keep the kv_indices copy's loads in flight at long contexts.
        # It also caps the prepare_round and commit grids, which grow with B.
        self.blocks = max(4, 2 * torch.cuda.get_device_properties(device).multi_processor_count)
        # Private test-only mutation; excluded from public API and warm_up.
        self._wait = True
        # Keys declared through warm_up; capture admits only these.
        self._declared = set()

    @staticmethod
    def _key_flags(variant, s, pdl):
        kind = KINDS[variant.kind]
        if kind == "commit":
            return kind, (s, variant.mode, variant.k2, pdl, True)
        return kind, (variant.rope, variant.k2, variant.cursors, pdl, True)

    @staticmethod
    def _describe(kind, q, flags):
        if kind == "commit":
            s, mode, k2, pdl, _ = flags
            what = f"mode={mode!r}, k2={k2}"
            extra = f", num_stop_tokens={s}"
        else:
            rope, k2, cursors, pdl, _ = flags
            what = f"rope={rope}, k2={k2}, cursors={cursors}"
            extra = ""
        return f"RoundVariant({PUBLIC[kind]!r}, {what}) at block_size={q}{extra}, use_pdl={pdl}"

    def _get(self, kind, q, flags):
        key = (self.device, self.arch, kind, q, *flags)
        compiled = _COMPILED.get(key)
        if torch.cuda.is_current_stream_capturing():
            # flags[-1] False is the private no-wait test mutant, never declared.
            if flags[-1] and key not in self._declared:
                raise RuntimeError(
                    f"speculative: {self._describe(kind, q, flags)} was not declared "
                    "to speculative.warm_up before capture"
                )
            if compiled is None:
                raise RuntimeError("call speculative.warm_up before capture")
        if compiled is None:
            with _LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    compiled = compile_variant(kind, q, flags, self.arch)
                    _COMPILED[key] = compiled
        return compiled

    def warm_up(self, q, s, pdl, variants):
        with torch.cuda.device(self.device):
            for variant in variants:
                kind, flags = self._key_flags(variant, s, pdl)
                self._get(kind, q, flags)
                self._declared.add((self.device, self.arch, kind, q, *flags))

    @staticmethod
    def _args(t, names):
        # CuTe 4.8 rejects zero extents. Borrow one element with the expected
        # dtype/rank from existing inputs. Static flags ensure it is never read.
        dummy = {}
        for name in names:
            value = t[name]
            if value is not None and value.numel():
                dummy.setdefault(value.dtype, value)
        args = []
        for name in names:
            value = t[name]
            if value is None or value.numel() == 0:
                dtype = torch.int64 if name in I64 else torch.bool if name in BOOL else torch.int32
                value = dummy[dtype][:1]
                if name in TABLES:
                    value = value.view(1, 1)
            args.append(value)
        return tuple(args)

    def prepare(self, t):
        flags = (
            t["rope_offsets"] is not None,
            t["k2_slots"] is not None,
            t["cursors"] is not None,
            t["use_pdl"],
            self._wait,
        )
        with torch.cuda.device(self.device):
            fn = self._get("prepare", t["block_size"], flags)[
                t["request_indices"].numel() > SMALL_BATCH
            ]
            fn(
                self._args(t, P_NAMES),
                t["max_seq_len"],
                t["guard_start"],
                t["mask_token_id"],
                t["kv_indices"].numel(),
                self.blocks,
            )

    def prepare_round(self, t):
        flags = (
            t["rope_offsets"] is not None,
            t["k2_slots"] is not None,
            t["cursors"] is not None,
            t["use_pdl"],
            self._wait,
        )
        with torch.cuda.device(self.device):
            fn = self._get("round", t["block_size"], flags)[
                t["request_indices"].numel() > SMALL_BATCH
            ]
            # CTA b % grid owns row b; no KV copy, so at most one CTA per row.
            blocks = max(4, min(t["request_indices"].numel(), self.blocks))
            fn(self._args(t, R_NAMES), t["max_seq_len"], 0, t["mask_token_id"], 0, blocks)

    def commit(self, t):
        mode = (
            "sampled"
            if t["proposals"] is None
            else "prefilled"
            if t["target_ids"] is None
            else "greedy"
        )
        flags = (
            t["stop_token_ids"].numel(),
            mode,
            t["k2_lengths"] is not None,
            t["use_pdl"],
            self._wait,
        )
        with torch.cuda.device(self.device):
            fn = self._get("commit", t["block_size"], flags)[
                t["request_indices"].numel() > SMALL_BATCH
            ]
            # One warp per row (four per CTA); B<=4 is a single CTA.
            blocks = min(-(-t["request_indices"].numel() // 4), self.blocks)
            fn(self._args(t, C_NAMES), t["max_seq_len"], blocks)
