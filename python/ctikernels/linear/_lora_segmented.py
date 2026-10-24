# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Two-launch, adapter-grouped LoRA on every CUDA device with compute capability >= 8.9.

The provider is our Triton kernel pair (``triton/_lora_segmented.py``): a split-K shrink
with a deterministic last-arrival fixup, then an expand that adds into the output in FP32
and rounds once. The engine builds stable adapter groups once per batch with
lora_segment_metadata. CSR segments address sorted positions; an optional int32
permutation maps them to original X/output rows. CHUNK (16/32/64) sorted tokens share
A/B reads, including across request boundaries; the helper returns an O(1) chunk table
and chunks=None keeps the CSR scan. The grid bound ceil(M/CHUNK)+S also supports
ungrouped/ragged CSR and empty segments.

Pooled and separate per-slice factors use int64 A/B pointer tables, cached by storage
addresses, geometry and device; no factor copies are made. Dense pool A defaults to
slice*ranks[adapter] (SGLang's layout); padded slice*max_rank is opt-in. Capture never
compiles or builds host pointer tables: warm each operand set/shape eagerly. On cc >= 9.0
both launches use PDL on the current stream.

Validation reads no device values. Callers ensure CSR starts at zero, is monotone and
ends at M, permutation is a bijection of [0,M), adapter IDs are negative or in range,
ranks lie in [0,max_rank], and scalings are finite. Negative IDs/zero ranks skip updates.
Empty M/S is a no-op. max_seq_len is advisory only.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ctikernels.linear import _lora_policy

_LOCK = threading.RLock()
_COMPILED = {}
_POINTER_TABLES = {}
_FACTOR_METADATA = {}
_LAST_CONFIG = threading.local()


def _parse_config(config):
    """Normalize four tuning fields; legacy three-field inputs imply auto CHUNK."""
    if config is None:
        return (None, None, None, None)
    if isinstance(config, str):
        fields = config.split(",")
        config = []
        for raw in fields:
            field = raw.strip()
            if field == "auto":
                config.append(None)
            elif field.isascii() and field.isdecimal():
                config.append(int(field))
            else:
                raise ValueError("config fields must be integers or 'auto'")
    if isinstance(config, (tuple, list)) and len(config) == 3:
        config = (*config, None)
    if not isinstance(config, (tuple, list)) or len(config) != 4:
        raise ValueError("config must contain splits,rank_tile,col_tile,chunk")
    result = []
    for name, value, allowed in zip(
        ("splits", "rank_tile", "col_tile", "chunk"),
        config,
        (range(1, 129), (16, 32), (64, 128, 256), (16, 32, 64)),
        strict=True,
    ):
        if value is None or (isinstance(value, str) and value == "auto"):
            result.append(None)
        elif type(value) is not int or value not in allowed:
            raise ValueError(f"invalid config {name}: expected {tuple(allowed)} or 'auto'")
        else:
            result.append(value)
    return tuple(result)


# Read exactly once, including validation: capture never consults the environment.
_ENV_CONFIG = _parse_config(os.environ.get("CTI_LORA_SEG_CONFIG"))


@lru_cache(maxsize=None)
def _supported(device):
    return _lora_policy.segmented_provider(torch.cuda.get_device_capability(device)) is not None


def _check_device(x):
    if not _supported(x.device):
        raise ValueError("segmented LoRA requires CUDA compute capability 8.9 or newer")


def last_config():
    """Return (splits, rank_tile, col_tile, chunk) for this thread's last delta call.

    None means no successful launch (including an empty call). During graph
    capture this describes the captured launch; graph replay does not run Python.
    """
    return getattr(_LAST_CONFIG, "value", None)


@dataclass(frozen=True, slots=True)
class LoRAEpilogue:
    """``out[:, offset_s : offset_s + width_s] += low[:, s, :rank] . B_s^T`` inside the GEMM.

    ``low`` [M, slices, R >= rank] (unit last stride) is the shrink result, already scaled:
    BF16/FP16 when rounded (TP=1, column-parallel), FP32 for a row-parallel partial, which
    expand takes as hi + lo 16-bit parts (~16 significant bits, not full FP32) instead of
    one rounding (the caller's base all-reduce then sums the deltas too).
    ``b_ptrs`` int64 [slices] holds the data pointers of each ``B_s`` [width_s, rank] (the
    output's dtype, row stride ``b_ld``, unit column stride); ``slices`` int32 [2 * slices]
    holds (column offset, width) pairs. Both are device tables built once, outside capture.
    """

    low: torch.Tensor
    b_ptrs: torch.Tensor
    slices: torch.Tensor
    rank: int
    b_ld: int

    @property
    def rank_tile(self) -> int:
        return max(16, min(64, 1 << (self.rank - 1).bit_length()))

    def check(self, m: int, out: torch.Tensor) -> None:
        """Host-only validation against an [m, N] output."""
        low, count = self.low, self.b_ptrs.numel()
        if low.dtype not in (torch.float32, out.dtype) or low.device != out.device:
            raise ValueError("LoRA low must be FP32 or the output dtype, on the output device")
        if low.ndim != 3 or low.shape[:2] != (m, count) or low.stride(2) != 1:
            raise ValueError("LoRA low must be [M, slices, R] with unit last stride")
        if type(self.rank) is not int or not 1 <= self.rank <= low.shape[2]:
            raise ValueError("LoRA rank must be in [1, R]")
        if type(self.b_ld) is not int or self.b_ld < self.rank:
            raise ValueError("LoRA b_ld must be at least the rank")
        if self.b_ptrs.dtype != torch.int64 or self.slices.dtype != torch.int32:
            raise ValueError("LoRA b_ptrs must be int64 and slices int32")
        if self.slices.shape != (2 * count,) or not count:
            raise ValueError("LoRA slices must hold one (offset, width) pair per B pointer")
        if any(t.device != out.device for t in (self.b_ptrs, self.slices)):
            raise ValueError("LoRA tables must be on the output device")


class LoRAChunkTable(NamedTuple):
    """Sorted <=chunk_tokens intervals, padded to ceil(M/CHUNK)+S with adapter=-1.

    chunk_tokens is immutable host metadata: 16, 32 or 64 (default 16).
    count is a host scheduling hint (an upper bound for device-built metadata),
    never a launch limit. Update all three arrays when changing groups on replay.
    Metadata and weights must be ready before the preceding PDL kernel launches.
    """

    chunk_start: torch.Tensor
    chunk_end: torch.Tensor
    chunk_adapter: torch.Tensor
    count: int
    chunk_tokens: int = 16


def _validate(
    output,
    x,
    lora_a,
    lora_b,
    seg_indptr,
    weight_indices,
    ranks,
    scalings,
    output_sizes,
    max_seq_len,
    permutation=None,
    a_slice_stride="rank",
    chunks=None,
):
    tensors = (output, x, lora_a, lora_b, seg_indptr, weight_indices, ranks, scalings)
    if permutation is not None:
        tensors += (permutation,)
    if chunks is not None:
        if not isinstance(chunks, LoRAChunkTable):
            raise ValueError("chunks must be a LoRAChunkTable from lora_segment_metadata")
        _check_chunk_tokens(chunks.chunk_tokens)
        tensors += chunks[:3]
    if a_slice_stride not in ("rank", "max_rank"):
        raise ValueError("a_slice_stride must be 'rank' or 'max_rank'")
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("all operands and device metadata must be tensors")
    if (
        not isinstance(output_sizes, (tuple, list))
        or not output_sizes
        or any(type(n) is not int or n <= 0 for n in output_sizes)
    ):
        raise ValueError("output_sizes must contain positive integer slice widths")
    sizes = tuple(output_sizes)
    if max_seq_len is not None and (type(max_seq_len) is not int or max_seq_len < 0):
        raise ValueError("max_seq_len must be a nonnegative integer or None")
    if x.ndim != 2 or output.ndim != 2 or lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError("expected x/output matrices and three-dimensional A/B pools")
    m, k = x.shape
    if permutation is not None and (permutation.shape != (m,) or permutation.dtype != torch.int32):
        raise ValueError("permutation must be int32 [M]")
    adapters, n, max_rank = lora_b.shape
    if k < 1 or not 1 <= max_rank <= 512:
        raise ValueError("K must be positive and max_rank must be in [1,512]")
    if n != sum(sizes) or output.shape != (m, n):
        raise ValueError("output and B rows must match M and sum(output_sizes)")
    if lora_a.shape != (adapters, len(sizes) * max_rank, k):
        raise ValueError("A must have shape [adapters, slices*max_rank, K]")
    if weight_indices.ndim != 1 or seg_indptr.shape != (weight_indices.numel() + 1,):
        raise ValueError("seg_indptr must be [S+1] and weight_indices [S]")
    if chunks is not None:
        bound = (m + chunks.chunk_tokens - 1) // chunks.chunk_tokens + weight_indices.numel()
        if any(t.shape != (bound,) or t.dtype != torch.int32 for t in chunks[:3]):
            raise ValueError("chunk arrays must be int32 [ceil(M/CHUNK)+S]")
        if type(chunks.count) is not int or not 0 <= chunks.count <= min(m, bound):
            raise ValueError("chunk count must be a host integer within the chunk bound")
    if ranks.shape != (adapters,) or scalings.shape != (adapters,):
        raise ValueError("ranks and scalings must have one entry per adapter")
    if x.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != x.dtype for t in (output, lora_a, lora_b)
    ):
        raise ValueError("x, output, A and B must share BF16 or FP16 dtype")
    if any(t.dtype != torch.int32 for t in (seg_indptr, weight_indices, ranks)):
        raise ValueError("segment pointers, adapter indices and ranks must be int32")
    if scalings.dtype != torch.float32:
        raise ValueError("scalings must be float32")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all operands must be on the same CUDA device")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("all operands must be strided tensors")
    if any(not t.is_contiguous() for t in tensors[1:]):
        raise ValueError("inputs and metadata must be contiguous")
    if output.stride(1) != 1 or (m > 1 and output.stride(0) < n):
        raise ValueError("output must have unit last stride and nonoverlapping rows")
    if any(t.requires_grad for t in tensors):
        raise ValueError("segmented LoRA is inference-only; detach tensors requiring gradients")
    if m and any(torch._C._overlaps(output, t) for t in tensors[1:]):
        raise ValueError("output must not share storage with inputs or metadata")
    if x.device.type != "cuda":
        raise ValueError("segmented LoRA requires CUDA tensors")
    return m, k, max_rank, sizes


def _check_chunk_tokens(chunk_tokens):
    if type(chunk_tokens) is not int or chunk_tokens not in (16, 32, 64):
        raise ValueError("chunk_tokens must be 16, 32 or 64")


def _launch_chunk(config, chunks):
    override = config[3]
    if chunks is not None:
        _check_chunk_tokens(chunks.chunk_tokens)
        if override is not None and override != chunks.chunk_tokens:
            raise ValueError("config chunk must match chunks.chunk_tokens; rebuild metadata")
        return chunks.chunk_tokens
    return 16 if override is None else override


def lora_segment_metadata(
    adapter_ids_per_request, request_lengths, *, device=None, total_tokens=None, chunk_tokens=16
):
    """Build stable adapter groups and <=chunk_tokens chunks, once per batch.

    chunk_tokens must be 16 (default), 32 or 64; it also selects the kernel
    specialization. Changing it requires rebuilding metadata and recapturing.
    Host lists/CPU tensors produce compact CSR. CUDA request vectors produce
    padded CSR with M slots without a host readback; supply host ``total_tokens``
    for that path. Device lengths/IDs are caller-validated integer vectors with
    nonnegative lengths summing to total_tokens. Returned int32 tensors can be
    passed directly as keyword arguments to the segmented APIs. The chunks entry
    is a LoRAChunkTable with three device int32 arrays and a host scheduling hint;
    reuse it across projections. Negative IDs form groups too, whose rows are
    skipped by the kernels. Device-built count is a conservative upper bound.
    """
    _check_chunk_tokens(chunk_tokens)
    ids, lengths = adapter_ids_per_request, request_lengths
    if isinstance(ids, torch.Tensor) and ids.device.type == "cuda":
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.device != ids.device
            or ids.ndim != 1
            or lengths.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("device request IDs/lengths must be matching integer vectors")
        if type(total_tokens) is not int or total_tokens < 0:
            raise ValueError("device metadata requires nonnegative host total_tokens")
        requested = None if device is None else torch.device(device)
        if requested is not None and (
            requested.type != "cuda"
            or (requested.index is not None and requested.index != ids.device.index)
        ):
            raise ValueError("device metadata must stay on its input device")
        m = total_tokens
        tokens = torch.repeat_interleave(ids, lengths, output_size=m)
        permutation = torch.argsort(tokens, stable=True)
        sorted_ids = tokens[permutation]
        positions = torch.arange(m, device=ids.device, dtype=torch.int64)
        changed = torch.cat(
            (
                torch.ones(min(1, m), device=ids.device, dtype=torch.bool),
                sorted_ids[1:] != sorted_ids[:-1],
            )
        )
        starts = torch.where(changed, positions, 0).cummax(0).values
        chunks = (changed | ((positions - starts) % chunk_tokens == 0)).long().cumsum(0) - 1
        ptr = torch.full((m + 1,), m, device=ids.device, dtype=torch.int64)
        ptr.scatter_reduce_(0, chunks, positions, reduce="amin", include_self=True)
        owners = torch.full((m,), -1, device=ids.device, dtype=ids.dtype)
        # Every member of a chunk has the same adapter; reduce avoids write races.
        owners.scatter_reduce_(0, chunks, sorted_ids, reduce="amin", include_self=False)
        padding = (m + chunk_tokens - 1) // chunk_tokens
        table = LoRAChunkTable(
            torch.cat((ptr[:-1], ptr.new_full((padding,), m))).int(),
            torch.cat((ptr[1:], ptr.new_full((padding,), m))).int(),
            torch.cat((owners, owners.new_full((padding,), -1))).int(),
            min(m, padding + ids.numel()),
            chunk_tokens,
        )
        return {
            "seg_indptr": ptr.int(),
            "weight_indices": owners.int(),
            "permutation": permutation.int(),
            "chunks": table,
        }
    if isinstance(lengths, torch.Tensor) and lengths.device.type != "cpu":
        raise ValueError("host IDs require host request lengths")
    for values in (ids, lengths):
        if isinstance(values, torch.Tensor) and (
            values.device.type != "cpu"
            or values.ndim != 1
            or values.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("host request IDs/lengths must be integer vectors")
    ids, lengths = list(ids), list(lengths)
    if any(isinstance(v, torch.Tensor) for v in ids + lengths):
        ids = [int(v) for v in ids]
        lengths = [int(v) for v in lengths]
    if len(ids) != len(lengths) or any(type(v) is not int for v in ids + lengths):
        raise ValueError("request IDs and lengths must be matching integer sequences")
    if any(v < 0 for v in lengths):
        raise ValueError("request lengths must be nonnegative")
    m = sum(lengths)
    if total_tokens is not None and total_tokens != m:
        raise ValueError("total_tokens must equal the sum of request lengths")
    groups, offset = {}, 0
    for adapter, length in zip(ids, lengths, strict=True):
        groups.setdefault(adapter, []).extend(range(offset, offset + length))
        offset += length
    permutation, ptr, owners = [], [0], []
    for adapter, rows in sorted(groups.items()):
        permutation.extend(rows)
        for lo in range(0, len(rows), chunk_tokens):
            ptr.append(ptr[-1] + min(chunk_tokens, len(rows) - lo))
            owners.append(adapter)
    integer = lambda values: torch.tensor(values, dtype=torch.int32, device=device)
    padding = (m + chunk_tokens - 1) // chunk_tokens
    return {
        "seg_indptr": integer(ptr),
        "weight_indices": integer(owners),
        "permutation": integer(permutation),
        "chunks": LoRAChunkTable(
            integer(ptr[:-1] + [m] * padding),
            integer(ptr[1:] + [m] * padding),
            integer(owners + [-1] * padding),
            len(owners),
            chunk_tokens,
        ),
    }


def _pointer_tables(a_factors, b_factors, device, capturing, *, sizes, offsets=None):
    # Only tiny address tensors are retained, never the factor storage. An event
    # orders initialization across streams without synchronizing the host. The
    # last table holds int32 (output column offset, width) per slice.
    stream = torch.cuda.current_stream(device)
    addresses = tuple(t.data_ptr() for t in (*a_factors, *b_factors))
    shapes = tuple(tuple(t.shape) for t in (*a_factors, *b_factors))
    key = (device, addresses, shapes, offsets, sizes)
    cached = _POINTER_TABLES.get(key)
    if cached is None:
        if capturing:
            raise RuntimeError("new LoRA pointer tables during capture; warm it eagerly first")
        if offsets is None:
            aa = [t.data_ptr() for t in a_factors]
            bb = [t.data_ptr() for t in b_factors]
        else:
            a, b = a_factors[0], b_factors[0]
            a_offsets, b_offsets = offsets
            aa = [a.data_ptr() + v * a.element_size() for v in a_offsets]
            bb = [b.data_ptr() + v * b.element_size() for v in b_offsets]
        cached = (
            torch.tensor(aa, device=device, dtype=torch.int64),
            torch.tensor(bb, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.int32),
            torch.tensor(
                [v for i, n in enumerate(sizes) for v in (sum(sizes[:i]), n)],
                device=device,
                dtype=torch.int32,
            ),
        )
        ready = torch.cuda.Event()
        ready.record(stream)
        _POINTER_TABLES[key] = (cached, ready, {stream.cuda_stream})
    cached, ready, streams = _POINTER_TABLES[key]
    # Capture cannot wait on an uncaptured event; graph capture begins after a
    # device synchronization, which already orders the table upload.
    if stream.cuda_stream not in streams and not capturing:
        stream.wait_event(ready)
        streams.add(stream.cuda_stream)
    return cached


def _launch(
    output,
    x,
    tables,
    ptr,
    ids,
    ranks,
    scales,
    sizes,
    rank,
    permutation,
    b_ld,
    rank_stride,
    chunks=None,
    config=None,
    *,
    row_parallel=False,
):
    from .triton._lora_segmented import launch

    config = _ENV_CONFIG if config is None else _parse_config(config)
    _LAST_CONFIG.value, low = launch(
        output,
        x,
        tables,
        ptr,
        ids,
        ranks,
        scales,
        sizes,
        rank,
        permutation,
        b_ld,
        rank_stride,
        chunks,
        config,
        _launch_chunk(config, chunks),
        row_parallel,
        capturing=torch.cuda.is_current_stream_capturing(),
        compiled=_COMPILED,
    )
    return low if output is None else output


def lora_segmented_delta_(
    output,
    x,
    lora_a,
    lora_b,
    *,
    seg_indptr,
    weight_indices,
    ranks,
    scalings,
    output_sizes,
    max_seq_len=None,
    permutation=None,
    a_slice_stride="rank",
    chunks=None,
    config=None,
    row_parallel=False,
):
    """Add segmented, per-adapter LoRA deltas in place and return ``output``.

    x=[M,K], output=[M,sum(output_sizes)], A=[adapters,slices*max_rank,K],
    B=[adapters,sum(output_sizes),max_rank]. A slice s starts at s*ranks[a]
    by default, or s*max_rank with a_slice_stride="max_rank". Optional permutation
    is int32 [M], mapping sorted positions in CSR to original x/output rows.
    Metadata: int32 CSR [S+1], int32 adapter IDs [S], int32 ranks [adapters],
    float32 scalings [adapters], all contiguous on x's CUDA device. Negative IDs
    skip updates. All slices of an adapter share its rank. BF16/FP16 inputs,
    FP32 accumulation, inference only. Output permits a nonoverlapping row stride.
    Optional chunks is the helper's LoRAChunkTable, padded to ceil(M/CHUNK)+S;
    chunks=None uses the CSR scan. Intervals must partition the same sorted rows
    as CSR in <=CHUNK-token chunks, with adapter=-1 for padded entries. Its count
    only selects tiling; it never truncates the grid. CHUNK comes from the
    table (default 16); an explicit config CHUNK must match chunk_tokens.
    Without a table CHUNK defaults to 16 and the CSR scan uses the override.

    Shape/dtype/layout/device validation never reads device data. Callers ensure
    valid CSR, adapter indices, ranks in [0,max_rank] and finite scalings. Empty
    M/S is a no-op. ``max_seq_len`` is optional advisory metadata, never a launch
    limit. Graph capture requires one eager warm-up per variant; metadata values
    and weights can change in place on replay. For PDL prefetch, factors, pointer
    tables, ranks and grouping metadata must be ready before the preceding PDL
    kernel launches; x/output may be produced by that kernel. Uses the current
    CUDA stream. Keep CSR and chunks consistent when updating groups on replay.

    ``config`` overrides tuning as "splits,rank_tile,col_tile,chunk" or a four-item
    tuple/list. Legacy three-item inputs imply auto CHUNK. Splits are 1..128, rank tiles 16/32 (must fit the rank bucket),
    column tiles 64/128/256, CHUNK 16/32/64; any field may be "auto" (or None in a sequence).
    Omitted config uses CTI_LORA_SEG_CONFIG, read once at import. An explicit
    config replaces that environment setting; "auto,auto,auto,auto" uses host rules.
    Auto splits account for the selected rank tile and CHUNK chunk count. See :func:`last_config`.

    Tensor parallelism. Column-parallel: A is replicated, B and ``output`` hold this rank's
    N shard; the local delta is complete. Row-parallel (``row_parallel=True``): x and A hold
    this rank's K shard, ``output`` is the rank's base partial; the delta is a per-rank
    partial, reduced by the caller's base all-reduce, and the shrink partial is never
    rounded to one 16-bit value before expand (hi + lo parts).
    """
    _LAST_CONFIG.value = None
    _check_row_parallel(row_parallel)
    config = _ENV_CONFIG if config is None else _parse_config(config)
    m, k, rank, sizes = _validate(
        output,
        x,
        lora_a,
        lora_b,
        seg_indptr,
        weight_indices,
        ranks,
        scalings,
        output_sizes,
        max_seq_len,
        permutation,
        a_slice_stride,
        chunks,
    )
    if m == 0 or weight_indices.numel() == 0 or lora_a.shape[0] == 0:
        return output
    with torch.cuda.device(x.device), _LOCK:
        capturing = torch.cuda.is_current_stream_capturing()
        a_offsets, b_offsets = [], []
        for adapter in range(lora_b.shape[0]):
            offset = 0
            for slice_id, size in enumerate(sizes):
                a_offsets.append(
                    (
                        adapter * len(sizes) * rank
                        + (slice_id * rank if a_slice_stride == "max_rank" else 0)
                    )
                    * k
                )
                b_offsets.append((adapter * sum(sizes) + offset) * rank)
                offset += size
        tables = _pointer_tables(
            (lora_a,),
            (lora_b,),
            x.device,
            capturing,
            sizes=sizes,
            offsets=(tuple(a_offsets), tuple(b_offsets)),
        )
        _check_device(x)
        return _launch(
            output,
            x,
            tables,
            seg_indptr,
            weight_indices,
            ranks,
            scalings,
            sizes,
            rank,
            permutation,
            rank,
            int(a_slice_stride == "rank"),
            chunks,
            config,
            row_parallel=row_parallel,
        )


def _check_row_parallel(row_parallel):
    if type(row_parallel) is not bool:
        raise ValueError("row_parallel must be a bool")


def lora_delta_factors_(
    output, x, factors, *, scaling, output_sizes, config=None, row_parallel=False
):
    """Add a single adapter using separate (A0,B0,A1,B1,...) tensors.

    All slices share rank in [1,512]. Scaling is a device float32 scalar (or a
    Python number cached as one); in-place factor/scalar updates survive replay.
    No packing/copying of factors occurs. Auto CHUNK is 16; a config override (32/64)
    rebuilds the cached table for that size.
    Warm each M/CHUNK before capture.
    ``config`` has the same tuning override semantics as lora_segmented_delta_;
    ``row_parallel`` has its tensor-parallel contract.
    """
    return _factors(output, x, factors, scaling, output_sizes, config, row_parallel)


def lora_shrink(x, factors, *, scaling, output_sizes, row_parallel=False):
    """Run only the shrink of :func:`lora_delta_factors_` and return the
    ``LoRAEpilogue`` that a base GEMM adds before its output rounding.

    The low-rank tensor is scaled; rounded to x's dtype, or FP32 with ``row_parallel``.
    It is per-call scratch (or the active GraphWorkspace slot): consume it before the next
    LoRA call. The pointer tables are cached like the delta's; warm each M before capture.
    """
    return _factors(None, x, factors, scaling, output_sizes, None, row_parallel)


def _factors(output, x, factors, scaling, output_sizes, config, row_parallel):
    _LAST_CONFIG.value = None
    _check_row_parallel(row_parallel)
    config = _ENV_CONFIG if config is None else _parse_config(config)
    if not isinstance(factors, tuple) or not factors or len(factors) % 2:
        raise ValueError("factors must be a nonempty tuple (A0,B0,A1,B1,...)")
    if (
        not isinstance(output_sizes, (tuple, list))
        or not output_sizes
        or any(type(n) is not int or n <= 0 for n in output_sizes)
        or len(factors) != 2 * len(output_sizes)
    ):
        raise ValueError("output_sizes must contain one positive integer per factor pair")
    sizes = tuple(output_sizes)
    shrink_only = output is None
    tensors = (x, *factors) if shrink_only else (output, x, *factors)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("all operands must be tensors")
    if x.ndim != 2 or (not shrink_only and output.ndim != 2) or factors[0].ndim != 2:
        raise ValueError("expected matrices for x, output and factors")
    m, k = x.shape
    rank = factors[0].shape[0]
    if k < 1 or not 1 <= rank <= 512:
        raise ValueError("K must be positive and rank must be in [1,512]")
    if not shrink_only and output.shape != (m, sum(sizes)):
        raise ValueError("output must match M and sum(output_sizes)")
    for s, n in enumerate(sizes):
        if factors[2 * s].shape != (rank, k) or factors[2 * s + 1].shape != (n, rank):
            raise ValueError("factors must be A [rank,K], B [slice_width,rank] with shared rank")
    if x.dtype not in (torch.bfloat16, torch.float16) or any(t.dtype != x.dtype for t in tensors):
        raise ValueError("all operands must share BF16 or FP16 dtype")
    if isinstance(scaling, torch.Tensor):
        if scaling.dtype != torch.float32 or scaling.numel() != 1 or scaling.ndim > 1:
            raise ValueError("scaling must be a float32 scalar tensor")
        tensors += (scaling,)
    elif type(scaling) not in (float, int):
        raise ValueError("scaling must be a float32 scalar tensor or number")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all operands must be on the same CUDA device")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("all operands must be strided tensors")
    if any(not t.is_contiguous() for t in tensors[0 if shrink_only else 1 :]):
        raise ValueError("inputs and metadata must be contiguous")
    if any(t.requires_grad for t in tensors):
        raise ValueError("LoRA is inference-only; detach tensors requiring gradients")
    if not shrink_only and (output.stride(1) != 1 or (m > 1 and output.stride(0) < sum(sizes))):
        raise ValueError("output must have unit last stride and nonoverlapping rows")
    if not shrink_only and m and any(torch._C._overlaps(output, t) for t in tensors[1:]):
        raise ValueError("output must not share storage with inputs or metadata")
    if x.device.type != "cuda":
        raise ValueError("LoRA requires CUDA tensors")
    if m == 0 and not shrink_only:
        return output
    with torch.cuda.device(x.device), _LOCK:
        capturing = torch.cuda.is_current_stream_capturing()
        tables = _pointer_tables(factors[0::2], factors[1::2], x.device, capturing, sizes=sizes)
        _check_device(x)
        chunk = 16 if config[3] is None else config[3]
        key = (
            x.device,
            m,
            chunk,
            rank,
            None if isinstance(scaling, torch.Tensor) else float(scaling),
        )
        metadata = _FACTOR_METADATA.get(key)
        if metadata is None:
            if capturing:
                raise RuntimeError("new factor metadata during capture; warm it eagerly first")
            metadata = (
                torch.tensor([0, m], device=x.device, dtype=torch.int32),
                torch.zeros(1, device=x.device, dtype=torch.int32),
                torch.tensor([rank], device=x.device, dtype=torch.int32),
                None
                if isinstance(scaling, torch.Tensor)
                else torch.tensor([scaling], device=x.device, dtype=torch.float32),
                LoRAChunkTable(
                    torch.tensor([*range(0, m, chunk), m], device=x.device, dtype=torch.int32),
                    torch.tensor(
                        [*range(chunk, m, chunk), m, m], device=x.device, dtype=torch.int32
                    ),
                    torch.tensor(
                        [0] * ((m + chunk - 1) // chunk) + [-1], device=x.device, dtype=torch.int32
                    ),
                    (m + chunk - 1) // chunk,
                    chunk,
                ),
            )
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(x.device))
            _FACTOR_METADATA[key] = (
                metadata,
                ready,
                {torch.cuda.current_stream(x.device).cuda_stream},
            )
        metadata, ready, streams = _FACTOR_METADATA[key]
        stream = torch.cuda.current_stream(x.device)
        # Capture cannot wait on an uncaptured event; graph capture begins after
        # a device synchronization, which already orders the metadata uploads.
        if stream.cuda_stream not in streams and not capturing:
            stream.wait_event(ready)
            streams.add(stream.cuda_stream)
        ptr, ids, ranks, scale, chunks = metadata
        if isinstance(scaling, torch.Tensor):
            scale = scaling.view(1)
        result = _launch(
            output,
            x,
            tables,
            ptr,
            ids,
            ranks,
            scale,
            sizes,
            rank,
            None,
            rank,
            0,
            chunks,
            config,
            row_parallel=row_parallel,
        )
        return result if not shrink_only else LoRAEpilogue(result, tables[1], tables[3], rank, rank)


@lru_cache(maxsize=1)
def _base_linear():
    from .inference import Linear

    return Linear()


def lora_segmented_projection(
    x,
    weight,
    bias,
    lora_a,
    lora_b,
    *,
    seg_indptr,
    weight_indices,
    ranks,
    scalings,
    output_sizes,
    max_seq_len=None,
    permutation=None,
    a_slice_stride="rank",
    chunks=None,
    row_parallel=False,
):
    """Return a fresh base projection with the segmented delta applied in place.

    BF16 contiguous x/weight with no bias use ``ctikernels.linear.Linear``;
    other supported base projections use F.linear. The delta uses the Triton segmented kernels.
    Tensor, metadata and tensor-parallel contracts match :func:`lora_segmented_delta_`.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise ValueError("x and weight must be tensors")
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[1] != x.shape[1]:
        raise ValueError("expected x [M,K] and weight [N,K]")
    if weight.device != x.device or weight.dtype != x.dtype or weight.requires_grad:
        raise ValueError("base weight must match x device/dtype and be frozen")
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.shape != (weight.shape[0],)
        or bias.device != x.device
        or bias.dtype != x.dtype
        or bias.requires_grad
    ):
        raise ValueError("bias must be frozen [N] on x device with x dtype")
    # Validate before invoking a base kernel, including malformed CUDA metadata.
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _validate(
        output,
        x,
        lora_a,
        lora_b,
        seg_indptr,
        weight_indices,
        ranks,
        scalings,
        output_sizes,
        max_seq_len,
        permutation,
        a_slice_stride,
        chunks,
    )
    if x.dtype == torch.bfloat16 and x.is_contiguous() and weight.is_contiguous() and bias is None:
        _base_linear()(x, weight, out=output)
    else:
        with torch.no_grad():
            output = F.linear(x, weight, bias)
    return lora_segmented_delta_(
        output,
        x,
        lora_a,
        lora_b,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        ranks=ranks,
        scalings=scalings,
        output_sizes=output_sizes,
        max_seq_len=max_seq_len,
        permutation=permutation,
        a_slice_stride=a_slice_stride,
        chunks=chunks,
        row_parallel=row_parallel,
    )
