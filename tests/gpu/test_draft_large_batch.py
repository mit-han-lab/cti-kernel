# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Large-batch qualification; select the 21 CI rows with -m 'not exhaustive'.

Every arm has B=5/64/300 in CI. Table Q=1/33/64 rows document its narrower
public contract with explicit skips; valid auto rows never skip provider errors.
The existing dense FP32 oracle and window-suite tolerances remain unchanged.
"""

import sys
from dataclasses import dataclass
from itertools import product

import pytest
import torch

from ctikernels.attention import draft
from ctikernels.attention._draft import cute, flashinfer, flashinfer_mask
from ctikernels.attention.inference import CacheTableDraftAttention, DraftAttention

from ._dflash import SENTINEL, Fixture
from ._exhaustive import ci_cases
from .test_dflash_draft_window import capture, compare_replay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BATCHES = (5, 8, 16, 32, 48, 64, 191, 300)
ARMS = ("auto", "cute", "spread", "flashinfer", "table", "table-cute", "table-flashinfer")
MASKS = ((False, None), (True, None), (True, 1000), (False, 1000))
CAPACITY, PAGE = 2048, 16
# Isolate tuning without inspecting or reporting unrelated environment variables.
TUNING = (
    "PROVIDER",
    "SPREAD",
    "MERGE",
    "INLINE",
    "SPLITS",
    "SHORT",
    "ROWS",
    "CHUNK_CAP",
    "MIN_CHUNK",
    "TARGET_CTAS",
    "KEY_TILE",
    "KEY_GROUPS",
    "STAGES",
    "PIECES",
    "SPREAD_SLACK",
    "COOP_PDL",
    "EARLY_TRIGGER",
    "IKET",
)


@dataclass(frozen=True)
class Arm:
    name: str
    b: int
    d: int
    q: int
    causal: bool
    window: int | None
    hq: int = 8
    hk: int = 2

    @property
    def table(self):
        return self.name.startswith("table")

    @property
    def provider(self):
        if self.name in ("auto", "table"):
            return None
        return "flashinfer" if self.name.endswith("flashinfer") else "cute"

    @property
    def id(self):
        mask = "causal" if self.causal else "noncausal"
        mask += "_full" if self.window is None else f"_w{self.window}"
        suffix = ""
        if self.name == "spread":
            suffix = "-wide_spread" if self.b > 32 else "-spread_grid"
        if self.hk == 1:
            suffix += "-mqa_h64_k1"
        return f"{self.name}-b{self.b}-hd{self.d}-q{self.q}-{mask}{suffix}"

    def geometry(self):
        geometry = {
            "batch_size": self.b,
            "query_length": self.q,
            "num_qo_heads": self.hq,
            "num_kv_heads": self.hk,
            "head_dim": self.d,
            "dtype": torch.bfloat16,
            "causal": self.causal,
            "window": self.window,
            "device": torch.device("cuda"),
        }
        if self.table:
            # All C logical tokens fit in the table, including the query block.
            # M=C keeps frontier=C-Q valid; the API reserves another Q guard keys.
            geometry["max_seq_len"] = CAPACITY
        else:
            geometry.update(max_kv_tokens=CAPACITY, page_size=PAGE)
        return geometry


def cases():
    grid = [
        Arm(name, b, d, q, causal, window)
        for name, b, d, q, (causal, window) in product(
            ARMS, BATCHES, (64, 128, 256), (1, 2, 5, 16, 32, 33, 64), MASKS
        )
    ]
    grid += [
        Arm("auto", b, d, q, False, max(1, q // 2))
        for b, d, q in product(BATCHES, (64, 128, 256), (2, 5, 16, 33, 64))
    ]
    grid += [Arm("auto", b, 256, q, True, 1000, 64, 1) for b, q in product(BATCHES, (1, 2, 8, 16))]
    core = []
    for index, name in enumerate(ARMS):
        for offset, b in enumerate((5, 64, 300)):
            d = (64, 128, 256)[(index + offset) % 3]
            q = (2, 16, 32)[offset] if name.startswith("table") else (1, 33, 16)[offset]
            causal, window = MASKS[(index + offset) % len(MASKS)]
            core.append((Arm(name, b, d, q, causal, window),))
    return [
        pytest.param(*case.values, marks=case.marks, id=case.values[0].id)
        for case in ci_cases(grid, core=core)
    ]


def clear_compiled_ops():
    cute.compile_kernels.cache_clear()
    # Avoid importing the CuTe table compiler on devices that cannot use it.
    module = sys.modules.get("ctikernels.attention._draft.flashinfer._table_csr")
    if module is not None:
        module.compile_table_csr.cache_clear()
    for name, kernels in (
        ("ctikernels.attention._flashinfer_schedule", ("prepare_draft", "prepare_draft_chunked")),
        ("ctikernels.attention._draft.flashinfer_mask.prepare", ("pack_mask",)),
    ):
        module = sys.modules.get(name)
        if module is not None:
            for kernel in kernels:
                getattr(module, kernel).device_caches.clear()


@pytest.fixture(autouse=True)
def isolated_case(monkeypatch):
    for name in TUNING:
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    clear_compiled_ops()
    yield
    clear_compiled_ops()


def configure(arm, monkeypatch):
    if arm.name == "cute" or arm.table:
        monkeypatch.setenv("CTI_DRAFT_SPREAD", "0")
    elif arm.name == "spread":
        monkeypatch.setenv("CTI_DRAFT_SPREAD", "1")
    # A legacy merge disables spread. Tables also require inline merging.
    if arm.name == "spread" or arm.table:
        monkeypatch.setenv("CTI_DRAFT_INLINE", "1")


def check_support(arm):
    geometry = arm.geometry()
    cls = CacheTableDraftAttention if arm.table else DraftAttention
    cls.workspace_size(**geometry, provider=arm.provider)
    configuration = draft._table_configuration(**geometry) if arm.table else geometry
    implementation = draft._provider(arm.provider, configuration)
    if implementation in (flashinfer, flashinfer_mask):
        flashinfer.check_support(configuration["device"])


class RoutedFixture(Fixture):
    def check_metadata(self):
        if self.op.provider == "flashinfer-fa2-custom-mask":
            # Auto on non-SM12x trims whole pages before packing a narrow mask.
            from .test_draft_custom_mask import check_trimmed_metadata

            check_trimmed_metadata(self, self.p)
        else:
            super().check_metadata()


class TableFixture(Fixture):
    """Adapt the same fragmented K/V and oracle to the engine table interface.

    The original CSR op only prepares Fixture's metadata for its existing checks;
    all measured output comes from the table op. Each op is constructed once.
    Done rows use zero guard K/V and finite Q, as required by the table contract.
    """

    def __init__(self, arm):
        super().__init__(
            arm.b,
            arm.q,
            PAGE,
            arm.hq,
            arm.hk,
            arm.causal,
            arm.window,
            c=CAPACITY,
            provider=arm.provider,
            head_dim=arm.d,
        )
        self.csr_op = self.op
        self.guard = self.k.shape[0] * PAGE
        # Keep the original paged layout for Fixture's oracle and preparation.
        guard_pages = (arm.b * arm.q + PAGE - 1) // PAGE
        zeros = torch.zeros(
            (guard_pages, PAGE, arm.hk, arm.d), dtype=self.k.dtype, device=self.k.device
        )
        self.k = torch.cat((self.k, zeros))
        self.v = torch.cat((self.v, zeros))
        self.table_kv = (self.k.flatten(0, 1).unsqueeze(1), self.v.flatten(0, 1).unsqueeze(1))
        self.requests = torch.empty_like(self.slots)
        self.frontier = torch.empty(arm.b, dtype=torch.int64, device="cuda")
        self.done = torch.empty(arm.b, dtype=torch.bool, device="cuda")
        self.table_backing = torch.empty((arm.b, CAPACITY + 7), dtype=torch.int32, device="cuda")
        self.table = self.table_backing[:, :CAPACITY]
        geometry = arm.geometry()
        self.table_workspaces = [
            torch.zeros(size, dtype=torch.uint8, device="cuda")
            for size in CacheTableDraftAttention.workspace_size(**geometry, provider=arm.provider)
        ]
        arguments = {k: v for k, v in geometry.items() if k not in ("batch_size", "device")}
        self.op = CacheTableDraftAttention(
            *self.table_workspaces,
            request_indices=self.requests,
            frontier_table=self.frontier,
            done_table=self.done,
            cache_index_table=self.table,
            guard_start=self.guard,
            provider=arm.provider,
            **arguments,
        )

    def stage(self, contexts, slots, turn=0):
        super().stage(contexts, slots, turn)
        requests = torch.arange(self.b, dtype=torch.int32).roll(turn)
        frontier = torch.empty(self.b, dtype=torch.int64)
        done = torch.empty(self.b, dtype=torch.bool)
        table = torch.full(self.table_backing.shape, SENTINEL, dtype=torch.int32)
        ptr, indices = [0], []
        for row, request in enumerate(requests.tolist()):
            frontier[request] = contexts[row]
            done[request] = slots[row] == -1
            start, end = self.cpu_token_ptr[row : row + 2].tolist()
            table[request, : end - start] = self.cpu_token_idx[start:end]
            if slots[row] == -1:
                # Table done rows still read Q and attend to the guard keys.
                self.query[row * self.q : (row + 1) * self.q].zero_()
                keys = torch.arange(
                    self.guard + row * self.q, self.guard + (row + 1) * self.q, dtype=torch.int32
                )
            else:
                keys = self.cpu_token_idx[start:end]
            # Same CSR as table_csr, using vector slices instead of a Python
            # iteration over up to 614,400 individual CPU tensor scalars.
            indices.append(keys)
            ptr.append(ptr[-1] + keys.numel())
        self.table_ptr = torch.tensor(ptr, dtype=torch.int32)
        self.table_indices = torch.cat(indices)
        self.table_metadata = (
            (self.requests, requests),
            (self.frontier, frontier),
            (self.done, done),
            (self.table_backing, table),
        )
        for actual, expected in self.table_metadata:
            actual.copy_(expected)

    def run(self):
        self.csr_op.prepare()
        return self.op.run(q=self.query, paged_kv_cache=self.table_kv, out=self.out)

    def check_metadata(self):
        super().check_metadata()
        for actual, expected in self.table_metadata:
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        if self.op.provider == "flashinfer-fa2-table-csr":
            inputs = self.op._implementation._inputs
            torch.testing.assert_close(inputs[4].cpu(), self.table_ptr, atol=0, rtol=0)
            torch.testing.assert_close(
                inputs[5][: self.table_indices.numel()].cpu(), self.table_indices, atol=0, rtol=0
            )


def assert_provider(f, arm):
    known = (
        ("cute-table", "flashinfer-fa2-table-csr")
        if arm.table
        else ("cute-splitkv", "flashinfer-fa2-device-schedule", "flashinfer-fa2-custom-mask")
    )
    assert f.op.provider in known
    assert isinstance(f.op.pdl, bool)
    assert f.op.merge in ("none", "legacy", "tree", "distributed", "flashinfer")
    if arm.provider == "flashinfer":
        assert f.op.provider == (
            "flashinfer-fa2-table-csr" if arm.table else "flashinfer-fa2-device-schedule"
        )
    elif arm.provider == "cute":
        assert f.op.provider == ("cute-table" if arm.table else "cute-splitkv")
        assert isinstance(f.op._implementation, cute.Provider)
        assert f.op._implementation.spread == (arm.b if arm.name == "spread" else 0)
        if arm.name == "spread" and arm.b > cute.SPREAD_MAX_BATCH:
            assert f.op.merge == "tree"
            assert f.op._implementation.fallbacks == ()


@pytest.mark.parametrize("arm", cases())
def test_large_batch(arm, monkeypatch):
    if arm.table and not 2 <= arm.q <= 32:
        pytest.skip("CacheTableDraftAttention requires 2 <= Q <= 32")
    configure(arm, monkeypatch)
    # Skip only explicit provider admission failures, never compilation or replay
    # failures. Valid auto geometries must construct and run on the routed provider.
    if arm.provider is None:
        check_support(arm)
    else:
        try:
            check_support(arm)
        except NotImplementedError as error:
            pytest.skip(str(error))
    f = (
        TableFixture(arm)
        if arm.table
        else RoutedFixture(
            arm.b,
            arm.q,
            PAGE,
            arm.hq,
            arm.hk,
            arm.causal,
            arm.window,
            c=CAPACITY,
            provider=arm.provider,
            head_dim=arm.d,
        )
    )
    assert_provider(f, arm)
    slots = list(range(arm.b))
    maximum = CAPACITY - arm.q
    f.stage([maximum] * arm.b, slots)
    graph = capture([f])
    compare_replay([f], graph)
    patterns = (
        (
            [(0, 7, CAPACITY // 2, maximum)[row % 4] for row in slots],
            # Start at row zero so B=5 also exercises inactive -> active replay.
            [-1 if row % 7 == 0 else row for row in slots],
        ),
        ([7] * (arm.b - 1) + [maximum], slots),
        ([maximum] * arm.b, slots),
    )
    for turn, (contexts, active) in enumerate(patterns, start=1):
        f.stage(contexts, active, turn)
        compare_replay([f], graph)
        assert_provider(f, arm)
