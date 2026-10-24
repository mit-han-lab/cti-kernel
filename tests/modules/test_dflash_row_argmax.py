# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU provider semantics and metadata-only admission checks."""

import pytest
import torch

from ctikernels.sampling import RowArgmax

OP = RowArgmax("cpu")


def new_workspace(capacity):
    return torch.zeros(RowArgmax.workspace_size(capacity), dtype=torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_dflash_row_argmax_reference(dtype, out_dtype):
    storage = torch.empty(5, 10, dtype=dtype)
    logits = storage[:, 1:8]
    logits.copy_(
        torch.tensor(
            [
                [-float("inf")] * 7,
                [-0.0, +0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [-1.0, 4.0, 2.0, 4.0, -3.0, 4.0, -5.0],
                [float("inf"), float("nan"), 0.0, float("nan"), 0.0, 0.0, 0.0],
                [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            ]
        )
    )
    out = torch.empty(5, dtype=out_dtype)
    work = new_workspace(8)
    assert OP.provider == "reference"
    OP.warm_up(logits, out, work)
    assert not work.any()
    for _ in range(3):
        OP(logits, out, work)
        assert out.tolist() == [0, 0, 1, 1, 6]
        assert torch.equal(out, logits.argmax(-1))
        assert not work.any()


def test_dflash_row_argmax_empty_and_capacity():
    OP(torch.empty(0, 7), torch.empty(0, dtype=torch.int64), new_workspace(0))
    assert RowArgmax.workspace_size(3) == OP.workspace_size(3) == 72
    for capacity in (-1, True, 1.5, 2**31):
        with pytest.raises(ValueError):
            RowArgmax.workspace_size(capacity)


@pytest.mark.parametrize(
    "case,error",
    [
        ("grad", ValueError),
        ("dtype", TypeError),
        ("rank", ValueError),
        ("empty_vocab", ValueError),
        ("large_vocab", ValueError),
        ("inner_stride", ValueError),
        ("overlapping_rows", ValueError),
        ("out_dtype", TypeError),
        ("out_shape", ValueError),
        ("out_stride", ValueError),
        ("work_dtype", TypeError),
        ("work_shape", ValueError),
        ("work_stride", ValueError),
        ("capacity", ValueError),
        ("alignment", ValueError),
        ("alias", ValueError),
        ("device", ValueError),
        ("not_tensor", TypeError),
    ],
)
def test_dflash_row_argmax_validation(case, error):
    x, out, work = (
        torch.randn(4, 7),
        torch.empty(4, dtype=torch.int64),
        new_workspace(4),
    )
    if case == "grad":
        x.requires_grad_()
    elif case == "dtype":
        x = x.double()
    elif case == "rank":
        x = x.flatten()
    elif case == "empty_vocab":
        x = torch.empty(4, 0)
    elif case == "large_vocab":
        x = torch.as_strided(torch.empty(1), (4, 2**31), (0, 0))
    elif case == "inner_stride":
        x = torch.randn(4, 14)[:, ::2]
    elif case == "overlapping_rows":
        x = torch.randn(1, 7).expand(4, 7)
    elif case == "out_dtype":
        out = out.float()
    elif case == "out_shape":
        out = out.view(2, 2)
    elif case == "out_stride":
        out = torch.empty(8, dtype=torch.int64)[::2]
    elif case == "work_dtype":
        work = work.to(torch.int8)
    elif case == "work_shape":
        work = work.view(4, 24)
    elif case == "work_stride":
        work = torch.zeros(192, dtype=torch.uint8)[::2]
    elif case == "capacity":
        work = new_workspace(3)
    elif case == "alignment":
        work = torch.zeros(97, dtype=torch.uint8)[1:]
    elif case == "alias":
        out = work[:32].view(torch.int64)
    elif case == "device":
        out = torch.empty(4, dtype=torch.int64, device="meta")
    elif case == "not_tensor":
        x = [[1.0]]
    with pytest.raises(error):
        OP(x, out, work)


def test_dflash_row_argmax_devices():
    with pytest.raises(ValueError, match="CPU and CUDA"):
        RowArgmax("meta")


@pytest.mark.parametrize("proposal_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block", [2, 5, 16, 32, 33, 64, 65, 128])
def test_dflash_row_argmax_reference_chain(proposal_dtype, out_dtype, block):
    rows = 3 * block
    x = torch.randn(rows, 40, generator=torch.Generator().manual_seed(block))
    out = torch.empty(rows, dtype=out_dtype)
    work = new_workspace(rows + 3)
    proposals = torch.empty(6, block + 2, dtype=proposal_dtype)[::2, 1:block]
    proposals.copy_(x.argmax(-1).view(3, block)[:, :-1])
    proposals[0, 0] = (proposals[0, 0] + 1) % 40
    proposals[1, block - 2] = (proposals[1, block - 2] + 1) % 40
    chain = {"proposals": proposals, "block_size": block}
    OP.warm_up(x, out, work, **chain)
    assert not work.any()
    for kwargs in (chain, {}, chain):
        OP(x, out, work, **kwargs)
        assert torch.equal(out, x.argmax(-1))
        assert not work.any()


@pytest.mark.parametrize("method", ["__call__", "warm_up"])
@pytest.mark.parametrize(
    "case,error,message",
    [
        ("missing_block", ValueError, "both proposals and block_size"),
        ("missing_proposals", ValueError, "both proposals and block_size"),
        ("shape", ValueError, "unit inner stride"),
        ("rank", ValueError, "unit inner stride"),
        ("dtype", TypeError, "int32 or int64"),
        ("inner_stride", ValueError, "unit inner stride"),
        ("rows", ValueError, "multiple of block_size"),
        ("device", ValueError, "device"),
        ("not_tensor", TypeError, "must be a tensor"),
    ],
)
def test_dflash_row_argmax_chain_validation(method, case, error, message):
    x = torch.randn(10, 7)
    out = torch.empty(10, dtype=torch.int64)
    work = new_workspace(10)
    proposals, block = torch.zeros(2, 4, dtype=torch.int32), 5
    if case == "missing_block":
        block = None
    elif case == "missing_proposals":
        proposals = None
    elif case == "shape":
        proposals = proposals[:, :3]
    elif case == "rank":
        proposals = proposals.flatten()
    elif case == "dtype":
        proposals = proposals.float()
    elif case == "inner_stride":
        proposals = torch.zeros(2, 8, dtype=torch.int32)[:, ::2]
    elif case == "rows":
        x, out = x[:-1], out[:-1]
    elif case == "device":
        proposals = proposals.to("meta")
    elif case == "not_tensor":
        proposals = [[0] * 4] * 2
    with pytest.raises(error, match=message):
        getattr(OP, method)(x, out, work, proposals=proposals, block_size=block)


@pytest.mark.parametrize("block", [-1, 0, 2**31, True, 2.0])
@pytest.mark.parametrize("method", ["__call__", "warm_up"])
def test_dflash_row_argmax_chain_block_validation(method, block):
    with pytest.raises(ValueError, match="block_size must be an int >= 1"):
        getattr(OP, method)(
            torch.zeros(4, 7),
            torch.empty(4, dtype=torch.int64),
            new_workspace(4),
            proposals=torch.zeros(2, 1, dtype=torch.int32),
            block_size=block,
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_dflash_row_argmax_q1_empty_strides(dtype):
    logits = torch.tensor([[0.0, 2.0, 2.0], [3.0, 3.0, -1.0]])
    proposals = torch.empty_strided((2, 0), (17, 9), dtype=dtype)
    out = torch.empty(2, dtype=dtype)
    work = new_workspace(2)
    OP.warm_up(logits, out, work, proposals=proposals, block_size=1)
    OP(logits, out, work, proposals=proposals, block_size=1)
    assert out.tolist() == [1, 0]
    assert not work.any()
