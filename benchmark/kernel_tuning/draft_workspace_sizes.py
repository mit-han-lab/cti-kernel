# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Report 16,632 construction-time workspace bounds without allocating GPU tensors.

Run with --out report.jsonl, or omit --out to write JSONL to stdout. C is the
logical token capacity (including Q), as in test_draft_large_batch. Tables use
M=C so frontier=C-Q is legal, and internally reserve C+Q keys with page size 1;
CSR uses C keys and page size 16. Both capacities are recorded explicitly.
Only routing, admission and workspace sizing run: no plans or kernels compile.
"""

import argparse
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from itertools import product
from pathlib import Path

import torch

from ctikernels.attention import draft
from ctikernels.attention._draft import cute, flashinfer, flashinfer_mask
from ctikernels.attention.inference import CacheTableDraftAttention, DraftAttention

ARMS = ("auto", "cute", "spread", "flashinfer", "table", "table-cute", "table-flashinfer")
BATCHES = (1, 2, 4, 5, 8, 16, 32, 48, 64, 191, 300)
MASKS = ((False, None), (True, None), (True, 1000), (False, 1000))
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


@contextmanager
def arm_environment(arm):
    names = tuple(f"CTI_DRAFT_{name}" for name in TUNING)
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        if arm == "cute" or arm.startswith("table"):
            os.environ["CTI_DRAFT_SPREAD"] = "0"
        elif arm == "spread":
            os.environ["CTI_DRAFT_SPREAD"] = "1"
        if arm == "spread" or arm.startswith("table"):
            os.environ["CTI_DRAFT_INLINE"] = "1"
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@lru_cache(maxsize=None)
def flashinfer_support_error(device):
    # workspace_size checks capacity, whereas construction also checks the ABI.
    # Distribution metadata and hardware are constant for this report.
    try:
        flashinfer.check_support(device)
    except NotImplementedError as error:
        return f"{type(error).__name__}: {error}"
    return None


def selected_name(implementation, table):
    if table:
        return "cute-table" if implementation is cute else "flashinfer-fa2-table-csr"
    return {
        cute: "cute-splitkv",
        flashinfer: "flashinfer-fa2-device-schedule",
        flashinfer_mask: "flashinfer-fa2-custom-mask",
    }[implementation]


def records(device):
    device_name = torch.cuda.get_device_name(device)
    for arm in ARMS:
        table = arm.startswith("table")
        provider = (
            None
            if arm in ("auto", "table")
            else "flashinfer"
            if arm.endswith("flashinfer")
            else "cute"
        )
        cls = CacheTableDraftAttention if table else DraftAttention
        with arm_environment(arm):
            for b, d, q, (causal, window), (hq, hk), capacity in product(
                BATCHES,
                (64, 128, 256),
                (1, 16, 64),
                MASKS,
                ((8, 2), (32, 8)),
                (2048, 32768, 262144),
            ):
                geometry = {
                    "batch_size": b,
                    "query_length": q,
                    "num_qo_heads": hq,
                    "num_kv_heads": hk,
                    "head_dim": d,
                    "dtype": torch.bfloat16,
                    "causal": causal,
                    "window": window,
                    "device": device,
                }
                if table:
                    geometry["max_seq_len"] = capacity
                else:
                    geometry.update(max_kv_tokens=capacity, page_size=16)
                row = {
                    "device": device_name,
                    "arm": arm,
                    "geometry": {
                        key: value
                        for key, value in geometry.items()
                        if key not in ("device", "dtype")
                    },
                    "capacity": capacity,
                    "dtype": "bfloat16",
                    "page_size": 1 if table else 16,
                    "provider_max_kv_tokens": capacity + q if table else capacity,
                    "provider": None,
                    "float_bytes": None,
                    "int_bytes": None,
                    "error": None,
                }
                try:
                    configuration = (
                        draft._table_configuration(**geometry) if table else geometry.copy()
                    )
                    if not table:
                        # Match DraftAttention.workspace_size's normalization exactly.
                        configuration["window"] = (
                            None if window is not None and window >= capacity else window
                        )
                    implementation = draft._provider(provider, configuration)
                    row["provider"] = selected_name(implementation, table)
                    row["float_bytes"], row["int_bytes"] = cls.workspace_size(
                        **geometry, provider=provider
                    )
                    if implementation in (flashinfer, flashinfer_mask):
                        row["error"] = flashinfer_support_error(device)
                except (NotImplementedError, ValueError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="JSONL destination; defaults to stdout")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a current CUDA device is required")
    device = torch.device("cuda", torch.cuda.current_device())
    destination = args.out.open("w", encoding="utf-8") if args.out else nullcontext(sys.stdout)
    with destination as output:
        for row in records(device):
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
