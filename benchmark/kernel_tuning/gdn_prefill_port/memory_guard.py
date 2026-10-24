# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Conservative admission checks; on GB10 (unified memory) host memory is also checked."""

import json


def check(label, estimated_gib=12):
    import torch

    fields = {line.split(":")[0]: int(line.split()[1]) for line in open("/proc/meminfo")}
    available = fields["MemAvailable"] / 2**20
    torch.cuda.empty_cache()  # blocks a previous test left cached are free for this one
    free_gpu = torch.cuda.mem_get_info()[0] / 2**30
    spark = torch.cuda.get_device_capability() == (12, 1)
    admitted = free_gpu > estimated_gib + 8
    if spark:
        admitted = admitted and available - estimated_gib >= 32 and estimated_gib <= available * 0.4
    record = {
        "label": label,
        "available_gib": available,
        "cuda_free_gib": free_gpu,
        "estimated_gib": estimated_gib,
        "admitted": admitted,
        "spark": spark,
    }
    print("MEMORY_CHECK " + json.dumps(record), flush=True)
    return admitted


def pytest_runtest_setup(item):
    import pytest

    if not check(item.nodeid):
        pytest.skip("Insufficient free memory for conservative 12 GiB test allowance + margin")
