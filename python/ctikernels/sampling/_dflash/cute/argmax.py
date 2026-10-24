# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Single-launch split-row reduction and dynamic-layout CuTe 4.8 launcher."""

from importlib.metadata import version
from threading import RLock

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.runtime import make_fake_stream
from cutlass.utils import SmemAllocator

from ._pdl import PDL

THREADS = 128
# Chain mode skips reads only when a row is long enough to repay the flag read.
CHAIN_MIN_BYTES = 8 << 20
# Chain mode with Q > 32 over logits larger than L2 launches this many waves of
# CTAs (see split_count) and streams with read-only loads. Measured on GB10 and
# RTX 5090 against 1/2/4 waves; Q <= 32 keeps the single-wave configuration.
CHAIN_LONG_WAVES = 8
# Integrated devices use unified-memory DRAM, which streams one long contiguous
# range per CTA better than many short ranges. Choose this from a device property,
# not a device name or architecture. Early positions still split to publish an
# early mismatch while later whole-row CTAs are reading.
LONG_THREADS = 512
LONG_EARLY_POSITIONS = 8
LONG_EARLY_SPLITS = 8
LONG_CHECK_ITERATIONS = 8
_COMPILED = {}
_LOCK = RLock()


@cute.jit
def packed_key(value, index):
    value = Float32(value)
    bits = Uint32(llvm.bitcast(T.i32(), value.ir_value()))
    # Canonicalize both zero signs before the float-to-ordered-uint mapping.
    if value == Float32(0):
        bits = Uint32(0)
    ordered = bits ^ Uint32(0x80000000)
    if (bits & Uint32(0x80000000)) != 0:
        ordered = ~bits
    if value != value:
        ordered = Uint32(0xFFFFFFFF)
    return (Uint64(ordered) << 32) | Uint64(Uint32(0xFFFFFFFF) - Uint32(index))


@cute.jit
def warp_max(key):
    for k in cutlass.range_constexpr(5):
        lo = cute.arch.shuffle_sync_bfly(Uint32(key), offset=1 << k)
        hi = cute.arch.shuffle_sync_bfly(Uint32(key >> 32), offset=1 << k)
        key = max(key, (Uint64(hi) << 32) | Uint64(lo))
    return key


@cute.jit
def atomic64(ptr, value, instruction: cutlass.Constexpr):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value(), Uint64(value).ir_value()],
            f"{instruction} $0, [$1], $2;",
            "=l,l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def atomic32(ptr, value, instruction: cutlass.Constexpr):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint().ir_value(), Uint32(value).ir_value()],
            f"{instruction} $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def max_nan_bf16x2(a, b):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(), Uint32(b).ir_value()],
            "max.NaN.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def max_nan_f32(a, b):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(), Float32(b).ir_value()],
            "max.NaN.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def bits_to_f32(bits):
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value()))


class Kernel:
    def __init__(self, packed=True, chain=False, stream=False, long_rows=False):
        self.packed = packed
        self.chain = chain
        self.stream = stream
        # Long-row chain mode: early positions split, later rows whole.
        self.long_rows = long_rows
        self.threads = LONG_THREADS if long_rows else THREADS
        self.early = LONG_EARLY_POSITIONS
        self.check = LONG_CHECK_ITERATIONS

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        out: cute.Tensor,
        workspace: cute.Tensor,
        proposals: cute.Tensor,
        block: Int32,
        splits: Int32,
        per_split: Int64,
        stream: cuda.CUstream,
    ):
        grid = [logits.shape[0], splits, 1]
        if cutlass.const_expr(self.chain):
            grid = [splits, min(logits.shape[0], Int64(65535)), 1]
        if cutlass.const_expr(self.long_rows):
            rows = logits.shape[0]
            early = rows // Int64(block) * min(Int64(block), Int64(self.early))
            grid = [rows + early * (Int64(splits) - 1), 1, 1]
        self.kernel(
            logits.iterator,
            out.iterator,
            workspace.iterator,
            proposals.iterator,
            logits.shape[0],
            logits.shape[1],
            logits.stride[0],
            proposals.stride[0],
            block,
            splits,
            per_split,
        ).launch(
            grid=grid,
            block=[self.threads, 1, 1],
            use_pdl=PDL,
            stream=stream,
        )

    @cute.jit
    def packed_vector(self, source, vector, words, kept, key):
        aligned = cute.make_ptr(
            Uint32,
            (source + vector * 8).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cutlass.const_expr(self.stream):
            # Single-use stream: read-only path without L1 allocation.
            loaded = cute.arch.inline_ptx(
                "ld.global.nc.L1::no_allocate.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                write_only_types=[Uint32, Uint32, Uint32, Uint32],
                read_only_args=[Int64(aligned.toint())],
            )
            for i in cutlass.range_constexpr(4):
                words[i] = loaded[i]
        else:
            src = cute.make_tensor(aligned, cute.make_layout(4))
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=128)
            cute.copy(atom, src, words)
        pair = max_nan_bf16x2(
            max_nan_bf16x2(words[0], words[1]), max_nan_bf16x2(words[2], words[3])
        )
        vector_max = max_nan_f32(
            bits_to_f32(pair << Uint32(16)), bits_to_f32(pair & Uint32(0xFFFF0000))
        )
        candidate = packed_key(vector_max, vector)
        if candidate > key:
            key = candidate
            for i in cutlass.range_constexpr(4):
                kept[i] = words[i]
        return key

    @cute.jit
    def plain_vector(self, source, vector, values, key):
        vec = cute.size(values)
        base = vector * vec
        aligned = cute.make_ptr(
            source.value_type, (source + base).toint(), cute.AddressSpace.gmem, assumed_align=16
        )
        src = cute.make_tensor(aligned, cute.make_layout(vec))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), source.value_type, num_bits_per_copy=128
        )
        cute.copy(atom, src, values)
        for j in cutlass.range_constexpr(vec):
            key = max(key, packed_key(values[j], base + j))
        return key

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        out: cute.Pointer,
        workspace: cute.Pointer,
        proposals: cute.Pointer,
        rows: Int64,
        vocab: Int64,
        stride: Int64,
        proposal_stride: Int64,
        block: Int32,
        splits: Int32,
        per_split: Int64,
    ):
        if cutlass.const_expr(PDL):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.chain and not self.long_rows):
            split, first, _ = cute.arch.block_idx()
            _, row_step, _ = cute.arch.grid_dim()
            # CTA-uniform, position-major traversal; <= 65535 rows run once.
            # Int64 induction avoids overflowing after the last int32 row.
            linear = Int64(first)
            while linear < rows:
                self.reduce_row(
                    logits,
                    out,
                    workspace,
                    proposals,
                    rows,
                    vocab,
                    stride,
                    proposal_stride,
                    block,
                    linear,
                    split,
                    Int64(splits),
                    per_split,
                )
                # All warps must finish reading seen/partials before reuse.
                cute.arch.sync_threads()
                linear += Int64(row_step)
        else:
            cta, cta_split, _ = cute.arch.block_idx()
            self.reduce_row(
                logits,
                out,
                workspace,
                proposals,
                rows,
                vocab,
                stride,
                proposal_stride,
                block,
                Int64(cta),
                cta_split,
                Int64(splits),
                per_split,
            )

    @cute.jit
    def reduce_row(
        self,
        logits,
        out,
        workspace,
        proposals,
        rows,
        vocab,
        stride,
        proposal_stride,
        block,
        linear,
        split,
        splits,
        per_split,
    ):
        threads = self.threads
        if cutlass.const_expr(self.long_rows):
            # The long-row mode always splits early rows LONG_EARLY_SPLITS ways.
            splits = Int64(LONG_EARLY_SPLITS)
        row = Int32(linear)
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        dtype = logits.value_type
        vec = 128 // dtype.width
        skip = Int32(0)
        request, position, flags = Int64(0), Int32(0), cute.recast_ptr(workspace, dtype=Uint32)
        aborted = None
        if cutlass.const_expr(self.long_rows):
            aborted = SmemAllocator().allocate_tensor(Uint32, cute.make_layout(1))
        if cutlass.const_expr(self.chain):
            # Position-major CTA order: earlier draft positions start first. A row
            # after a request's known first mismatch is never consumed by greedy
            # acceptance, so it skips its reads and reports -1.
            requests = rows // Int64(block)
            if cutlass.const_expr(self.long_rows):
                # Long rows stream best as one contiguous range per CTA; the first
                # LONG_EARLY_POSITIONS positions split so a first mismatch there
                # publishes early. Later rows also stop between check segments.
                cta = linear
                early = requests * min(Int64(block), Int64(self.early))
                if Int64(cta) < early * splits:
                    linear = Int64(cta) // splits
                    split = Int32(Int64(cta) - Int64(linear) * splits)
                else:
                    linear = Int64(cta) - early * (splits - 1)
                    split = Int32(0)
                    splits = Int64(1)
            position = Int32(Int64(linear) // requests)
            request = Int64(linear) - Int64(position) * requests
            row = Int32(request * Int64(block) + Int64(position))
            flags = cute.recast_ptr(workspace + rows * 16 + request * 8, dtype=Uint32)
            seen = SmemAllocator().allocate_tensor(Uint32, cute.make_layout(1))
            if tid == 0:
                seen[0] = atomic32(flags, 0, "atom.relaxed.gpu.global.add.u32")
                if cutlass.const_expr(self.long_rows):
                    aborted[0] = Uint32(0)
            cute.arch.sync_threads()
            if seen[0] > Uint32(block - position):
                skip = Int32(1)
        source = logits + Int64(row) * stride
        # Split at vector boundaries. Scalar tail never reads padding or a neighbour.
        vectors = (vocab + vec - 1) // vec
        if cutlass.const_expr(self.long_rows):
            per_split = (vectors + splits - 1) // splits
        begin = Int64(split) * per_split
        end = min(begin + per_split, vectors)
        key = Uint64(0)
        values = cute.make_rmem_tensor(cute.make_layout(vec), dtype)
        # Vectors entirely inside the row; the last one may be partial.
        full = min(end, vocab // vec)
        scalar_from = begin
        if skip != 0:
            full, end = begin, begin
        # Whole-row CTAs re-read the request's mismatch word between segments
        # (lane 0 per warp, no barrier) and stop once a mismatch precedes the row.
        segment = threads * self.check
        live = Int32(1)
        if cutlass.const_expr(self.packed and dtype == cutlass.BFloat16):
            if (source.toint() & 15) == 0:
                # One NaN-propagating max per eight values, then one packed key per
                # vector: ordered maximum, ties to the smaller vector. The winning
                # vector's words stay in registers to find its first matching element.
                words = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                kept = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                kept.fill(Uint32(0))
                if cutlass.const_expr(self.long_rows):
                    for start in cutlass.range(begin, full, segment):
                        if live != 0:
                            for vector in cutlass.range(
                                start + tid, min(start + segment, full), threads, unroll=4
                            ):
                                key = self.packed_vector(source, vector, words, kept, key)
                            live = self.still_live(flags, splits, block, position, lane, aborted)
                else:
                    for vector in cutlass.range(begin + tid, full, threads, unroll=4):
                        key = self.packed_vector(source, vector, words, kept, key)
                if key != Uint64(0):
                    best_vector = Int64(Uint32(0xFFFFFFFF) - Uint32(key))
                    first, first_value = Int64(0), Float32(0)
                    for j in cutlass.range_constexpr(vec - 1, -1, -1):
                        value = bits_to_f32(kept[j // 2] << Uint32(16))
                        if cutlass.const_expr(j % 2 == 1):
                            value = bits_to_f32(kept[j // 2] & Uint32(0xFFFF0000))
                        if (packed_key(value, 0) >> 32) == (key >> 32):
                            first, first_value = best_vector * vec + j, value
                    key = packed_key(first_value, first)
                scalar_from = max(begin, full)
        elif (source.toint() & 15) == 0:
            # Branch-free body so the unrolled loop keeps several 128-bit loads in flight.
            if cutlass.const_expr(self.long_rows):
                for start in cutlass.range(begin, full, segment):
                    if live != 0:
                        for vector in cutlass.range(
                            start + tid, min(start + segment, full), threads, unroll=4
                        ):
                            key = self.plain_vector(source, vector, values, key)
                        live = self.still_live(flags, splits, block, position, lane, aborted)
            else:
                for vector in cutlass.range(begin + tid, full, threads, unroll=4):
                    key = self.plain_vector(source, vector, values, key)
            scalar_from = max(begin, full)
        for vector in cutlass.range(scalar_from + tid, end, threads):
            base = vector * vec
            for j in cutlass.range_constexpr(vec):
                if base + j < vocab:
                    key = max(key, packed_key(source[base + j], base + j))
        key = warp_max(key)
        partials = SmemAllocator().allocate_tensor(Uint64, cute.make_layout(threads // 32))
        if lane == 0:
            partials[warp] = key
        cute.arch.sync_threads()
        if warp == 0:
            key = Uint64(0)
            if lane < threads // 32:
                key = partials[lane]
            key = warp_max(key)
            if lane == 0:
                slot = cute.recast_ptr(workspace + Int64(row) * 16, dtype=Uint64)
                counter = cute.recast_ptr(workspace + Int64(row) * 16 + 8, dtype=Uint32)
                skipped = cute.recast_ptr(workspace + Int64(row) * 16 + 12, dtype=Uint32)
                if cutlass.const_expr(self.long_rows):
                    if aborted[0] != Uint32(0):
                        skip = Int32(1)
                if cutlass.const_expr(self.chain):
                    if skip != 0:
                        atomic32(skipped, 1, "atom.relaxed.gpu.global.add.u32")
                atomic64(slot, key, "atom.relaxed.gpu.global.max.u64")
                # Device threadfence publishes the slot before announcing arrival.
                llvm.inline_asm(
                    None,
                    [],
                    "membar.gl;",
                    "~{memory}",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                # Acq_rel RMWs form a chain: the last arrival observes every prior
                # publication. No spinning, hence no occupancy-dependent deadlock.
                arrived = atomic32(counter, 1, "atom.acq_rel.gpu.global.add.u32")
                if Int64(arrived) == splits - 1:
                    final = atomic64(slot, 0, "atom.acquire.gpu.global.add.u64")
                    result = Int64(Uint32(0xFFFFFFFF) - Uint32(final))
                    if cutlass.const_expr(self.chain):
                        if atomic32(skipped, 0, "atom.relaxed.gpu.global.exch.b32") != 0:
                            result = Int64(-1)
                    out[row] = result
                    atomic64(slot, 0, "atom.relaxed.gpu.global.exch.b64")
                    atomic32(counter, 0, "atom.release.gpu.global.exch.b32")
                    if cutlass.const_expr(self.chain):
                        if position < block - 1:
                            draft = Int64(proposals[request * proposal_stride + position])
                            if result != draft:
                                atomic32(
                                    flags,
                                    Uint32(block - position),
                                    "atom.relaxed.gpu.global.max.u32",
                                )
                        # The request's last finished row clears its flag words. Every
                        # mismatch publication precedes its row's acq_rel arrival.
                        finished = atomic32(flags + 1, 1, "atom.acq_rel.gpu.global.add.u32")
                        if finished == Uint32(block - 1):
                            atomic32(flags, 0, "atom.relaxed.gpu.global.exch.b32")
                            atomic32(flags + 1, 0, "atom.release.gpu.global.exch.b32")

    @cute.jit
    def still_live(self, flags, splits, block, position, lane, aborted):
        # Only whole-row CTAs stop early; a split row's other CTAs keep reading.
        live = Int32(1)
        if splits == Int64(1):
            seen = Uint32(0)
            if lane == 0:
                seen = atomic32(flags, 0, "atom.relaxed.gpu.global.add.u32")
            seen = cute.arch.shuffle_sync(seen, 0)
            if seen > Uint32(block - position):
                live = Int32(0)
                if lane == 0:
                    aborted[0] = Uint32(1)
        return live


def per_split_vectors(logits, splits):
    """128-bit vectors per split CTA, ceil(ceil(V / vec) / splits)."""
    vectors = -(-logits.shape[1] // (16 // logits.element_size()))
    return -(-vectors // splits)


def split_count(rows, vocab, element_size, sm_count, waves=1):
    """Aim at max(12 CTAs/SM, 1536 CTAs), capped by useful 128-thread vector tiles.

    Small-SM parts (GB10, 48 SMs) need many small CTAs in flight to reach DRAM
    bandwidth; 1536 CTAs is about two waves of 16 resident CTAs/SM there.
    ``waves`` multiplies the target: chain mode with Q > 32 uses several waves so
    that position-major later waves start after earlier positions publish a
    mismatch, and skip their reads. With one wave every CTA is already resident.
    """
    tiles = (vocab + THREADS * (16 // element_size) - 1) // (THREADS * (16 // element_size))
    target = max(12 * sm_count, 1536) * waves
    return max(1, min((target + rows - 1) // rows, tiles, 65535))


def compile_variant(
    dtype,
    out_dtype,
    splits,
    arch,
    packed=True,
    proposal_dtype=None,
    stream=False,
    long_rows=False,
):
    """Offline-capable; R, V, strides, block size, splits and capacity are runtime.

    splits is only a sample runtime argument, retained for offline caller compatibility.

    proposal_dtype selects the chain mode (int32/int64 [B, Q-1] proposals);
    stream selects read-only packed BF16 loads (multi-wave chain mode);
    long_rows selects the long-row chain mode (splits = LONG_EARLY_SPLITS).
    """
    types = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }
    out = cute.runtime.make_fake_tensor(
        types[out_dtype], (cute.sym_int64(),), (1,), assumed_align=types[out_dtype].width // 8
    )
    args = [
        cute.runtime.make_fake_tensor(
            types[dtype],
            (cute.sym_int64(), cute.sym_int64()),
            (cute.sym_int64(), 1),
            assumed_align=types[dtype].width // 8,
        ),
        out,
        cute.runtime.make_fake_tensor(cutlass.Uint8, (cute.sym_int64(),), (1,), assumed_align=8),
        out
        if proposal_dtype is None
        else cute.runtime.make_fake_tensor(
            types[proposal_dtype],
            (cute.sym_int64(), cute.sym_int64()),
            (cute.sym_int64(), 1),
            assumed_align=types[proposal_dtype].width // 8,
        ),
        Int32(2),
        Int32(splits),
        Int64(1),
    ]
    return cute.compile(
        Kernel(packed, proposal_dtype is not None, stream, long_rows),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch} --opt-level=3",
    )


class Provider:
    def __init__(self, device):
        if version("nvidia-cutlass-dsl") != "4.8.0":
            raise RuntimeError("RowArgmax requires nvidia-cutlass-dsl==4.8.0")
        self.device = device
        self.arch = "".join(map(str, torch.cuda.get_device_capability(device)))
        properties = torch.cuda.get_device_properties(device)
        self.sm_count = properties.multi_processor_count
        self.l2_bytes = properties.L2_cache_size
        integrated = getattr(properties, "is_integrated", None)
        if integrated is None:
            # Some torch versions omit this property; CUDA exposes it directly.
            status, integrated = cudart.cudaDeviceGetAttribute(
                cudart.cudaDeviceAttr.cudaDevAttrIntegrated, device.index
            )
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cannot query integrated-device property: {status}")
        self.is_integrated = bool(integrated)

    def _mode(self, logits_bytes, chained, block_size):
        """(chain, stream, long_rows) for a logits size; the compile key besides dtypes."""
        # Below the threshold the plain kernel writes every row's argmax,
        # which the chain contract also allows.
        chain = chained and logits_bytes >= CHAIN_MIN_BYTES
        # Q <= 32 keeps the single-wave configuration measured for Q 4-16. L2-resident
        # logits gain nothing from skipping and lose to the extra CTAs (GB10 B1 Q33).
        stream = chain and block_size > 32 and logits_bytes > self.l2_bytes
        long_rows = stream and self.is_integrated
        return chain, stream and not long_rows, long_rows

    def _get(self, logits, out, proposals=None, block_size=2):
        chained, stream, long_rows = self._mode(
            logits.numel() * logits.element_size(), proposals is not None, block_size
        )
        waves = CHAIN_LONG_WAVES if stream else 1
        splits = (
            LONG_EARLY_SPLITS
            if long_rows
            else split_count(*logits.shape, logits.element_size(), self.sm_count, waves)
        )
        chain = proposals.dtype if chained else None
        key = (self.device, logits.dtype, out.dtype, chain, stream, long_rows)
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("call RowArgmax.warm_up before capture")
            with _LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    compiled = compile_variant(
                        logits.dtype,
                        out.dtype,
                        splits,
                        self.arch,
                        proposal_dtype=chain,
                        stream=stream,
                        long_rows=long_rows,
                    )
                    _COMPILED[key] = compiled
        return compiled, splits, chained

    def warm_up(self, logits, out, proposals=None, block_size=2):
        """Compile every variant reachable at this or any smaller batch.

        The variant depends on the logits size, so a smaller captured batch can
        fall below the chain or L2 threshold. Plain calls (no proposals) have one
        variant. Chained calls with B = R/Q requests reach the modes of 1..B
        requests of the same Q and row width; the modes change only at the two
        thresholds, so checking the batches next to them covers every mode.
        """
        if proposals is None or proposals.numel() == 0:
            # Q == 1 has no drafts and always runs the plain kernel.
            batches = [None]
        else:
            row_bytes = block_size * logits.shape[1] * logits.element_size()
            capacity = logits.shape[0] // block_size
            batches = {1, capacity}
            for limit in (CHAIN_MIN_BYTES, self.l2_bytes):
                below = max(1, min(capacity, limit // row_bytes))
                batches.update((below, min(capacity, below + 1)))
            batches = sorted(batches)
        modes = set()
        with torch.cuda.device(self.device):
            for batch in batches:
                if batch is None:
                    self._get(logits, out)
                    continue
                mode = self._mode(batch * row_bytes, True, block_size)
                if mode in modes:
                    continue
                modes.add(mode)
                rows = batch * block_size
                self._get(logits[:rows], out[:rows], proposals[:batch], block_size)

    @staticmethod
    def _chain(logits, proposals):
        if proposals is None or proposals.numel() == 0:
            return None
        return proposals

    def run(self, logits, out, workspace, proposals=None, block_size=2):
        proposals = self._chain(logits, proposals)
        with torch.cuda.device(self.device):
            compiled, splits, chained = self._get(logits, out, proposals, block_size)
            compiled(
                logits,
                out,
                workspace,
                # Below the chain threshold the plain kernel writes every row's argmax,
                # which the chain contract also allows.
                proposals if chained else out,
                block_size,
                splits,
                # Host-computed vectors per split: no 64-bit division per CTA.
                per_split_vectors(logits, splits),
            )
