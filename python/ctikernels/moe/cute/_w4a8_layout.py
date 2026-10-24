# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# _moe_utils/moe_kernel_builder.py, _moe_utils/sm12x_blockscaled_layout.py).
# Local modifications: equal K32 scales, one mixed family, explicit resource
# accounting, producer/consumer staging, and no target-dependent import-time objects.

"""CPU resource/address model and lazy SM12x mixed-MMA layouts."""

from dataclasses import dataclass
from functools import lru_cache

from ._w4a8_schedule import ORDERS


def align(value, alignment=128):
    return (value + alignment - 1) // alignment * alignment


def scale_capacity(rows, experts=128):
    return (rows + 3 * experts) // 4 * 4


def scale_offset(row_offset, expert):
    return (row_offset + 3 * expert) // 4 * 4


def activation_scale_address(group, row, capacity):
    """Byte offset in int32 [K/128,P], independent of data-row offsets."""
    return 4 * ((group // 4) * capacity + row) + group % 4


def sf_cosize(rows, stages):
    # A small (swapped) activation tile still starts each TMA stage at 128 B.
    return (stages - 1) * max(rows, 32) + rows


def scale_lane_row(side, lane):
    """SM120 atom's scale-word owner, before tiled-MMA permutations."""
    if side == "a":
        return lane // 4 + (lane % 2) * 8
    if side == "b":
        return lane // 4
    raise ValueError("side must be a or b")


@dataclass(frozen=True)
class Tile:
    stage: int
    rows: int
    stages: int
    bn: int = 128
    bk: int = 128
    math_warps: int = 8
    watchdog: bool = False
    order: str = "expert_mtile_ntile"
    # L2 prefetch run of canonical weight rows, in K tiles ahead (0: off); per device.
    prefetch: int = 0

    def __post_init__(self):
        if not isinstance(self.watchdog, bool):
            raise TypeError("watchdog must be bool")
        if self.stage not in (1, 2):
            raise ValueError("stage must be FC1 or FC2")
        if self.rows not in ((8, 32, 64, 128) if self.stage == 1 else (32, 64, 128)):
            raise ValueError("unsupported row tile")
        if self.order not in ORDERS:
            raise ValueError("unsupported schedule order")
        if self.stages not in (1, 2, 3, 4):
            raise ValueError("stages must be 1, 2, 3 or 4")
        if type(self.prefetch) is not int or not 0 <= self.prefetch <= 8:
            raise ValueError("prefetch must be 0..8 K tiles")
        if self.bn != 128:
            raise ValueError("W4A8 BN must be 128 (one interleaved scale block per tile)")
        if (self.bk, self.math_warps) != (128, 8):
            raise ValueError("W4A8 requires BK=128 and eight math warps")

    @property
    def swapped(self):
        return self.rows == 8

    @property
    def alias_epilogue(self):
        # Only the 128-row tactic trades next-work prefetch for a smaller arena.
        return self.rows == 128

    @property
    def register_q_scales(self):
        # Two BN128 FC1 stages plus all shared scales consume 99 KiB, with
        # no room for barriers/work records. Load Q scale words into registers
        # for both widths at rows=128; smaller row tiles retain scale TMA.
        return self.rows == 128

    @property
    def physical(self):
        return (self.bn, self.rows, self.bk) if self.swapped else (self.rows, self.bn, self.bk)

    @property
    def weights(self):
        return 2 if self.stage == 1 else 1

    @property
    def accumulator_registers(self):
        """FP32 accumulators per math thread, excluding operand fragments."""
        return self.rows * self.bn * self.weights // (32 * self.math_warps)

    @property
    def transaction_bytes(self):
        # FP4 TMA unpack counts packed global bytes, not expanded shared bytes.
        return (
            self.rows * self.bk
            + self.weights * self.bn * self.bk // 2
            + 4 * ((0 if self.register_q_scales else self.rows) + self.weights * self.bn)
        )

    @property
    def data_bytes(self):
        # TMA expands FP4 to byte storage; transactions still count packed bytes.
        return (self.rows + self.weights * self.bn) * self.bk * self.stages

    @property
    def fields(self):
        """Actual struct fields: (name, bytes, alignment), including tail padding."""
        fields = [
            ("full", self.stages * 8, 8),
            ("empty", self.stages * 8, 8),
            ("work_full", 2 * 8, 8),
            ("work_empty", 2 * 8, 8),
            ("work", 2 * 6 * 4, 4),
            ("data", self.data_bytes, 128),
        ]
        if not self.register_q_scales:
            fields.append(("sq", 4 * sf_cosize(self.rows, self.stages), 128))
        fields.append(("sw", 4 * sf_cosize(self.bn, self.stages) * self.weights, 128))
        if not self.alias_epilogue:
            fields.append(("epilogue", 2 * self.rows * self.bn, 128))
        return tuple(fields)

    @property
    def smem_bytes(self):
        size = 0
        for _, extent, alignment in self.fields:
            size = align(size, alignment) + extent
        return align(size)

    def check_smem(self, limit):
        if self.smem_bytes > limit:
            raise ValueError(f"{self}: needs {self.smem_bytes} shared bytes; device limit {limit}")
        return self


def choose_tile(
    stage,
    rows,
    limit,
    stages=None,
    *,
    bn=128,
    watchdog=False,
    order="expert_mtile_ntile",
    prefetch=0,
):
    for count in (stages,) if stages is not None else (4, 3, 2):
        tile = Tile(stage, rows, count, bn=bn, watchdog=watchdog, order=order, prefetch=prefetch)
        if tile.smem_bytes <= limit:
            return tile
    raise ValueError(f"FC{stage} rows={rows} has no stage count within {limit} shared bytes")


@lru_cache(maxsize=1)
def dsl_layouts():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import cpasync
    from cutlass.tensor_utils import LayoutEnum
    from cutlass.utils import blackwell_helpers, hopper_helpers

    class Layouts:
        def __init__(self, tile):
            self.tile = tile
            self.q = self.data_layout(tile.rows, cutlass.Float8E4M3FN)
            self.w = self.data_layout(tile.bn, cutlass.Int8)
            self.sq = self.scale_layout(tile.rows)
            # Weight scales arrive in FlashInfer's 128x4 interleave (BN = 128): a 128-row
            # block's K128 words are one 512-B run, row n at word 4 * (n % 32) + (n % 128) // 32;
            # the logical index stays n.
            self.sw = cute.make_layout(((32, 4), 1, tile.stages), stride=((4, 1), 128, 128))

        def data_layout(self, rows, dtype):
            kind = hopper_helpers.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, 128)
            atom = cute.nvgpu.warpgroup.make_smem_layout_atom(kind, dtype)
            return cute.tile_to_shape(atom, (rows, 128, self.tile.stages), order=(0, 1, 2))

        def scale_layout(self, rows):
            return cute.make_layout((rows, 1, self.tile.stages), stride=(1, rows, max(rows, 32)))

        def mma(self):
            tile = self.tile
            a, b = cutlass.Float8E4M3FN, cutlass.Float4E2M1FN
            wm = 4 if tile.rows >= 64 else 2
            permutation = blackwell_helpers.get_permutation_mnk(tile.physical, 32, True)
            if tile.swapped:
                a, b = b, a
                wm = 8
                permutation = tile.physical[:2] + (32,)
            op = cute.nvgpu.warp.MmaMXF8F6F4Op(a, b, cutlass.Float32, cutlass.Float8E8M0FNU)
            warps = cute.make_ordered_layout((wm, 8 // wm, 1), order=(0, 1, 2))
            return cute.make_tiled_mma(op, warps, permutation_mnk=permutation)

        def lora_mma(self):
            """BF16 m16n8k16 with the W4A8 MMA's warps and M/N permutation (non-swapped):
            its C fragments are the W4A8 accumulators', so it can seed them."""
            tile = self.tile
            wm = 4 if tile.rows >= 64 else 2
            permutation = blackwell_helpers.get_permutation_mnk(tile.physical, 32, True)
            op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
            warps = cute.make_ordered_layout((wm, 8 // wm, 1), order=(0, 1, 2))
            return cute.make_tiled_mma(op, warps, permutation_mnk=permutation[:2] + (16,))

        def data_copy(self, mma, side):
            weight = (side == "a") == self.tile.swapped
            nwarps = 1 if self.tile.swapped else (2 if self.tile.rows >= 64 else 4)
            matrices = 4 if side == "a" or self.tile.physical[1] // nwarps >= 16 else 2
            if weight:
                op = cute.nvgpu.warp.LdMatrix8x16x8bOp(
                    transpose=False, num_matrices=matrices, unpack_bits=4
                )
                dtype = cutlass.Int8
            else:
                op = cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=matrices)
                dtype = cutlass.Float8E4M3FN
            atom = cute.make_copy_atom(op, dtype)
            return (cute.make_tiled_copy_A if side == "a" else cute.make_tiled_copy_B)(atom, mma)

        @staticmethod
        def tma(tensor, layout, rows, *, weight=False, scale=False):
            return cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                tensor,
                cute.slice_(layout, (None, None, 0)),
                (rows, 1 if scale else 128),
                num_multicast=1,
                internal_type=cutlass.Int8 if weight else None,
            )

        @staticmethod
        def scale_threads(layout, mma, side):
            # Collapse K to one packed int32 word. The MMA lanes broadcast a
            # row's word; the four bytes are selected separately below.
            axis = 0 if side == "a" else 1
            shape = ((2, 2, 8), 1) if side == "a" else ((4, 8), 1)
            stride = ((8, 0, 1), 16) if side == "a" else ((0, 1), 8)
            perm = mma.permutation_mnk
            warps = mma.thr_layout_vmnk
            tiled = cute.logical_divide(layout, (perm[axis], cute.make_layout(1)))
            atoms = cute.zipped_divide(
                tiled, (cute.make_layout(mma.shape_mnk[axis]), cute.make_layout(1))
            )
            lanes = cute.composition(atoms, (cute.make_layout(shape, stride=stride), None))
            return cute.zipped_divide(
                lanes,
                (
                    None,
                    (
                        cute.make_layout(cute.size(warps[axis + 1])),
                        cute.make_layout(cute.size(warps[3])),
                    ),
                ),
            )

        def scale_copy(self, mma, side):
            axis = 0 if side == "a" else 1
            warps = mma.thr_layout_vmnk
            extent = cute.size(mma.permutation_mnk[axis])
            layout = self.scale_threads(cute.make_layout((extent, 1)), mma, side)
            broadcast = cute.make_layout(
                (cute.size(warps[1]), cute.size(warps[2])),
                stride=(1, 0) if side == "a" else (0, 1),
            )
            # Thread modes are (lane, (warp-M/N, warp-K)); keep warp-K nested
            # while introducing the other operand's broadcast warp axis.
            tv = cute.composition(layout, ((None, (broadcast, None)), None))
            tv = cute.composition(tv, (cute.right_inverse(warps), None))
            return cute.make_tiled_copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32), tv, (extent, 1)
            )

        def scale_fragment(self, tensor, mma, tid, side):
            axis = 0 if side == "a" else 1
            layout = self.scale_threads(tensor.layout, mma, side)
            view = cute.make_tensor(tensor.iterator, layout)
            coord = mma.thr_layout_vmnk.get_flat_coord(tid)
            owner = (coord[0], (coord[axis + 1], coord[3]))
            rest = (None, (None,) * cute.rank(layout, mode=[1, 1]))
            return cute.make_rmem_tensor_like(view[owner, rest])

        @staticmethod
        def scale_bytes(fragment):
            # Packed-SF view: (vector, MN, (atoms/SF, SF/tile), tiles/pack).
            # Equal K32 gives four bytes per tile and exactly one tile/pack.
            return cute.make_tensor(
                cute.recast_ptr(fragment.iterator, dtype=cutlass.Float8E8M0FNU),
                cute.make_layout(
                    (32, cute.size(fragment, mode=[1]), (1, 4), (1, 1)),
                    stride=(0, 4, (0, 1), (0, 4)),
                ),
            )

    return Layouts
